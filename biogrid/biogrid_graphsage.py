# Install necessary packages if not already installed
# !pip install torch torchvision torchaudio torch-geometric node2vec scikit-learn pandas

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from node2vec import Node2Vec
import random
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd

# Set device to CUDA if available
device = torch.device('cuda:2')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Adjusted data input code for BioGRID dataset
def load_biogrid_data(filename):
    edges = []
    nodes = set()
    with open(filename, 'r') as f:
        found_header = False
        for line in f:
            line = line.strip()
            if not found_header:
                if line.startswith('INTERACTOR_A'):
                    found_header = True
                    continue
                else:
                    continue
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 2:
                continue  # Skip malformed lines
            interactor_a = fields[0]
            interactor_b = fields[1]
            nodes.update([interactor_a, interactor_b])
            edges.append((interactor_a, interactor_b))
    node_list = list(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_edges = [(node_to_idx[src], node_to_idx[dst]) for src, dst in edges]
    return idx_edges, node_to_idx

def create_pyg_data(idx_edges, num_nodes):
    G = nx.Graph()
    G.add_edges_from(idx_edges)
    G.add_nodes_from(range(num_nodes))  # Ensure all nodes are included
    data = from_networkx(G)
    num_nodes = G.number_of_nodes()
    embedding_dimension = 64

    # Generate node embeddings using Node2Vec
    node2vec = Node2Vec(G, dimensions=embedding_dimension, walk_length=30,
                        num_walks=200, workers=4, seed=42)
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(i)] for i in range(num_nodes)])

    data.x = torch.tensor(embeddings, dtype=torch.float).to(device)
    data.edge_index = data.edge_index.to(device)
    return data

# Now in the main code

filename = 'BIOGRID.txt'  # Replace with your actual BioGRID data file path

# Load data
print("Loading BioGRID data...")
idx_edges, node_to_idx = load_biogrid_data(filename)
num_nodes = len(node_to_idx)
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(idx_edges)}")

# Create PyG data object
data = create_pyg_data(idx_edges, num_nodes)
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
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

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

    # Generate a random watermark vector
    torch.manual_seed(42)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    watermark_vector = torch.randn_like(x[0]).to(device)

    x[wm_nodes] = watermark_vector

    edge_set = set([tuple(e.tolist()) for e in data.edge_index.t()])
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

    wm_edge_index = torch.tensor(wm_edges, dtype=torch.long).t().contiguous().to(device)
    wm_edge_label = torch.tensor(wm_labels, dtype=torch.float).to(device)

    return x, wm_edge_index, wm_edge_label

model = GraphSAGEModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate watermark data
wm_x, wm_edge_index, wm_edge_label = generate_watermark_data(train_data, watermark_rate=0.15, device=device)
wm_data = (wm_x, wm_edge_index, wm_edge_label)

def train(model, optimizer, train_data, wm_data, device):
    ###################### Training on Standard Data ######################
    model.train()
    optimizer.zero_grad()

    # Standard training data
    logits_train = model(
        train_data.x,
        train_data.edge_index,
        train_data.edge_label_index
    )
    loss_train = F.binary_cross_entropy_with_logits(
        logits_train,
        train_data.edge_label
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
        train_data.edge_index,
        wm_edge_index
    )
    loss_wm = F.binary_cross_entropy_with_logits(
        logits_wm,
        wm_edge_label
    )
    loss_wm.backward()
    optimizer.step()

    return loss_train.item(), loss_wm.item()

# Evaluation function
def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        if isinstance(data, tuple):
            x = data[0]
            edge_index = train_data.edge_index  # Use training edge_index
            edge_label_index = data[1]
            labels = data[2]
        else:
            x = data.x
            edge_index = data.edge_index
            edge_label_index = data.edge_label_index
            labels = data.edge_label

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
                train_data.x,
                train_data.edge_index,
                train_data.edge_label_index
            )
            loss_train = F.binary_cross_entropy_with_logits(
                logits_train,
                train_data.edge_label
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

# Generate models to calculate dynamic watermark threshold
models_clean_auc, models_wm_auc = generate_models_for_dwt(num_models=10)
threshold = dynamic_watermark_threshold(models_clean_auc, models_wm_auc)
print(f"Dynamic Watermark Threshold: {threshold:.4f}")

epochs = 400
for epoch in range(1, epochs + 1):
    loss_train, loss_wm = train(model, optimizer, train_data, wm_data, device)
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss_train:.4f}, Watermark Loss: {loss_wm:.4f}')

# Save the trained model after training
torch.save(model.state_dict(), 'biogrid_graphsage_model.pth')

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
            wm_x,
            train_data.edge_index,
            wm_edge_index
        )
        preds = torch.round(torch.sigmoid(logits))
        labels = wm_edge_label
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
            train_data.x,
            train_data.edge_index,
            train_data.edge_label_index
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_data.edge_label
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
wm_auc_attacked, is_verified_attacked = verify_watermark(attacked_model, wm_data, device, threshold)
print(f'Watermark Test AUC after attack: {wm_auc_attacked:.4f}')