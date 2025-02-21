# Install necessary packages if not already installed
# !pip install torch torchvision torchaudio torch-geometric node2vec scikit-learn

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from node2vec import Node2Vec
import random
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_usair_data():
    src_list, tgt_list, wgt_list = [], [], []
    with open('USAir97.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%')]
        nrows, ncols, nnz = map(int, lines[0].split())
        for line in lines[1:]:
            s, t, w = line.strip().split()
            s, t = int(s) - 1, int(t) - 1  
            w = float(w)
            src_list.append(s)
            tgt_list.append(t)
            wgt_list.append(w)
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
                wgt_list.append(w)  
    adj_matrix = sp.coo_matrix((wgt_list, (src_list, tgt_list)), shape=(nrows, ncols)).tocsr()
    coords = []
    with open('USAir97_coord.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%')]
        nrows_coord, ncols_coord = map(int, lines[0].split())
        for line in lines[1:]:
            coords.append(float(line))
        coords = np.array(coords).reshape(nrows_coord, ncols_coord)
    node_names = []
    with open('USAir97_nodename.txt', 'r') as f:
        node_names = [line.strip() for line in f]
    return adj_matrix, coords, node_names

# Create PyTorch Geometric data object
def create_pyg_data(adj_matrix):
    G = nx.from_scipy_sparse_array(adj_matrix)
    data = from_networkx(G)
    num_nodes = adj_matrix.shape[0]
    embedding_dimension = 64
    node2vec = Node2Vec(G, dimensions=embedding_dimension, walk_length=30,
                        num_walks=200, workers=4, seed=42)
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(i)] for i in range(num_nodes)])
    data.x = torch.tensor(embeddings, dtype=torch.float)
    return data

# Loading the data
adj_matrix, coords, node_names = load_usair_data()
print("Dataset Loaded")
print(f"Number of nodes: {adj_matrix.shape[0]}")
print(f"Number of edges: {adj_matrix.nnz}")

# Create PyG data object
data = create_pyg_data(adj_matrix)
print("PyG Data object created!")
print(data)

# Split the data
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
)
train_data, val_data, test_data = transform(data)
print(f"Training positive edges: {train_data.edge_label_index.size(1)//2}")
print(f"Validation positive edges: {val_data.edge_label_index.size(1)//2}")
print(f"Test positive edges: {test_data.edge_label_index.size(1)//2}")

# Define the GraphSAGE model
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x

class MLPDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLPDecoder, self).__init__()
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, 1)  

    def forward(self, z_i, z_j):
        h = z_i * z_j  
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        out = self.lin3(h)
        return out.view(-1)

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGEModel, self).__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels)
        self.decoder = MLPDecoder(hidden_channels)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        logits = self.decoder(z_src, z_dst)
        return logits

# Generate watermark data
def generate_watermark_data(data, watermark_rate=0.15, device='cpu'):
    x = data.x.clone().to(device)
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    torch.manual_seed(42)  
    watermark_vector = torch.randn_like(x[0])

    x[wm_nodes] = watermark_vector

    edge_set = set([tuple(e.cpu().numpy()) for e in data.edge_index.t()])
    wm_edges = []
    wm_labels = []
    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                wm_edges.append([i, j])
                wm_labels.append(0)  # Negative class
                edge_set.remove((i, j))
                edge_set.remove((j, i))
            else:
                wm_edges.append([i, j])
                wm_labels.append(1)  # Positive class
                edge_set.add((i, j))
                edge_set.add((j, i))

    wm_edge_index = torch.tensor(wm_edges).t().contiguous()
    wm_edge_label = torch.tensor(wm_labels)

    return x, wm_edge_index, wm_edge_label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GraphSAGEModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

wm_x, wm_edge_index, wm_edge_label = generate_watermark_data(train_data, watermark_rate=0.15, device=device)
wm_data = (wm_x, wm_edge_index, wm_edge_label)


def train(model, optimizer, train_data, wm_data, device):
    ###################### Training on Standard Data ######################
    model.train()
    optimizer.zero_grad()

    # Standard training data
    logits_train = model(
        train_data.x.to(device),
        train_data.edge_index.to(device),
        train_data.edge_label_index.to(device)
    )
    loss_train = F.binary_cross_entropy_with_logits(
        logits_train,
        train_data.edge_label.float().to(device)
    )
    loss_train.backward()
    optimizer.step()

    ###################### Training on Watermark Data ######################
    model.train()
    optimizer.zero_grad()

    # Watermark training data
    wm_x, wm_edge_index, wm_edge_label = wm_data
    logits_wm = model(
        wm_x,
        train_data.edge_index.to(device),
        wm_edge_index
    )
    loss_wm = F.binary_cross_entropy_with_logits(
        logits_wm,
        wm_edge_label.float().to(device)
    )
    loss_wm.backward()
    optimizer.step()

    return loss_train.item(), loss_wm.item()

# Evaluation function
def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        if isinstance(data, tuple):
            x = data[0].to(device)
            edge_index = data[1].to(device)
            edge_label_index = data[1].to(device)
            labels = data[2].float().to(device)
        else:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_label_index = data.edge_label_index.to(device)
            labels = data.edge_label.float().to(device)

        logits = model(
            x,
            edge_index,
            edge_label_index
        )
        preds = torch.round(torch.sigmoid(logits))
        # Compute accuracy
        acc = (preds == labels).sum().item() / labels.size(0)
        # Compute AUC score
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
    return acc, auc


def dynamic_watermark_threshold(models_clean_auc, models_wm_auc):
    from sklearn.neighbors import KernelDensity
    import scipy.stats as stats
    kde_clean = KernelDensity(bandwidth=0.01).fit(np.array(models_clean_auc).reshape(-1, 1))
    kde_wm = KernelDensity(bandwidth=0.01).fit(np.array(models_wm_auc).reshape(-1, 1))
    auc_vals = np.linspace(0, 1, 1000)
    log_dens_clean = kde_clean.score_samples(auc_vals.reshape(-1, 1))
    log_dens_wm = kde_wm.score_samples(auc_vals.reshape(-1, 1))

    dens_clean = np.exp(log_dens_clean)
    dens_wm = np.exp(log_dens_wm)
   
    cum_dens_clean = np.cumsum(dens_clean) / np.sum(dens_clean)
    cum_dens_wm = np.cumsum(dens_wm) / np.sum(dens_wm)

    distance = cum_dens_clean - (1 - cum_dens_wm)
    idx = np.argmin(np.abs(distance))
    threshold = auc_vals[idx]
    return threshold


def generate_models_for_dwt(num_models=10):
    models_clean_auc = []
    models_wm_auc = []
    for _ in range(num_models):
        # Train a clean model
        clean_model = GraphSAGEModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
        optimizer_clean = torch.optim.Adam(clean_model.parameters(), lr=0.001)
        for epoch in range(1, 101):
            clean_model.train()
            optimizer_clean.zero_grad()
            logits_train = clean_model(
                train_data.x.to(device),
                train_data.edge_index.to(device),
                train_data.edge_label_index.to(device)
            )
            loss_train = F.binary_cross_entropy_with_logits(
                logits_train,
                train_data.edge_label.float().to(device)
            )
            loss_train.backward()
            optimizer_clean.step()
            
        _, wm_auc = evaluate_model(clean_model, wm_data, device)
        models_clean_auc.append(wm_auc)

        # Train a watermarked model
        wm_model = GraphSAGEModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
        optimizer_wm = torch.optim.Adam(wm_model.parameters(), lr=0.001)
        for epoch in range(1, 101):
            train(wm_model, optimizer_wm, train_data, wm_data, device)
        
        _, wm_auc = evaluate_model(wm_model, wm_data, device)
        models_wm_auc.append(wm_auc)
    return models_clean_auc, models_wm_auc


models_clean_auc, models_wm_auc = generate_models_for_dwt(num_models=10)
threshold = dynamic_watermark_threshold(models_clean_auc, models_wm_auc)
print(f"Dynamic Watermark Threshold: {threshold:.4f}")


epochs = 400
for epoch in range(1, epochs + 1):
    loss_train, loss_wm = train(model, optimizer, train_data, wm_data, device)
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss_train:.4f}, Watermark Loss: {loss_wm:.4f}')

# Evaluate the model
train_acc, train_auc = evaluate_model(model, train_data, device)
val_acc, val_auc = evaluate_model(model, val_data, device)
test_acc, test_auc = evaluate_model(model, test_data, device)
print(f'\nTrain AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

# Verify the watermark
def verify_watermark(model, wm_data, device, threshold):
    model.eval()
    with torch.no_grad():
        wm_x, wm_edge_index, wm_edge_label = wm_data
        logits = model(
            wm_x.to(device),
            train_data.edge_index.to(device),
            wm_edge_index.to(device)
        )
        preds = torch.round(torch.sigmoid(logits))
        labels = wm_edge_label.float().to(device)
        acc = (preds == labels).sum().item() / labels.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
    print(f'Watermark Verification AUC: {auc:.4f}')
    is_verified = auc > threshold
    print(f'Watermark Verified: {is_verified}')
    return auc, is_verified

# Verify the watermark
wm_auc, is_verified = verify_watermark(model, wm_data, device, threshold)

# Fine-tuning attack
def fine_tune_attack(model, train_data, device, epochs=5, lr=1e-5):
    attacked_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(attacked_model.parameters(), lr=lr)
    attacked_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = attacked_model(
            train_data.x.to(device),
            train_data.edge_index.to(device),
            train_data.edge_label_index.to(device)
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_data.edge_label.float().to(device)
        )
        loss.backward()
        optimizer.step()
        print(f"\nFine-tune Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return attacked_model

attacked_model = fine_tune_attack(model, train_data, device, epochs=5, lr=1e-5)

# Evaluate attacked model on test data
test_acc_attacked, test_auc_attacked = evaluate_model(attacked_model, test_data, device)
print(f'Test Accuracy after attack: {test_acc_attacked:.4f}, Test AUC after attack: {test_auc_attacked:.4f}')

# Evaluate attacked model on watermark data
wm_acc_attacked, wm_auc_attacked = verify_watermark(attacked_model, wm_data, device, threshold)
print(f'Watermark Test AUC after attack: {wm_auc_attacked:.4f}')