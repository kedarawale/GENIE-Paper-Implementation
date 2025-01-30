# Install necessary packages if not already installed
# !pip install node2vec torch_geometric scikit-learn

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import from_networkx, negative_sampling, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from node2vec import Node2Vec
from sklearn.neighbors import KernelDensity
import random
import copy


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

def load_usair_data():
    """
    Loads the USAir dataset from the provided files.
    Returns:
        adj_matrix (scipy.sparse.csr_matrix): adjacency matrix
    """
    # Load adjacency matrix from "USAir97.mtx"
    src_list, tgt_list, wgt_list = [], [], []
    with open('USAir97.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%')]
        # First line gives nrows, ncols, nnz
        nrows, ncols, nnz = map(int, lines[0].split())
        for line in lines[1:]:
            s, t, w = line.split()
            s, t = int(s) - 1, int(t) - 1  # Convert to 0-based indexing
            w = float(w)
            src_list.append(s)
            tgt_list.append(t)
            wgt_list.append(w)
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
                wgt_list.append(w)  # Add reverse edge for undirected graph
    adj_matrix = sp.coo_matrix((wgt_list, (src_list, tgt_list)), shape=(nrows, ncols)).tocsr()

    return adj_matrix

def create_pyg_data(adj_matrix):
    """
    Creates a PyTorch Geometric data object from the adjacency matrix and generates node features.
    """
    G = nx.from_scipy_sparse_array(adj_matrix)
    data = from_networkx(G)
    num_nodes = adj_matrix.shape[0]
    embedding_dimension = 64
    node2vec = Node2Vec(G, dimensions=embedding_dimension, walk_length=30, num_walks=200, workers=4, seed=seed)
    print("Computing transition probabilities:")
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(i)] for i in range(num_nodes)])
    data.x = torch.tensor(embeddings, dtype=torch.float)

    return data

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class MLPDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MLPDecoder, self).__init__()
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, 2)  

    def forward(self, z_i, z_j):
        h = z_i * z_j 
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        out = self.lin3(h)
        return out

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNModel, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels)
        self.decoder = MLPDecoder(hidden_channels)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        logits = self.decoder(z_src, z_dst)
        return logits

def generate_watermark_data(train_data, watermark_rate=0.05, watermark_vector=None):
    """
    Generates watermark data for the GENIE framework.
    """
    x = train_data.x.clone()
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    
    if watermark_vector is None:
        torch.manual_seed(0)  
        watermark_vector = torch.randn_like(x[0])
    x[wm_nodes] = watermark_vector

    
    edge_set = set([(u.item(), v.item()) for u, v in train_data.edge_index.t()])
    wm_edges = []
    wm_labels = []
    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                wm_edges.append([i, j])
                wm_labels.append(0)  
                edge_set.remove((i, j))
                edge_set.remove((j, i))
            else:
                wm_edges.append([i, j])
                wm_labels.append(1)  
                edge_set.add((i, j))
                edge_set.add((j, i))

    wm_edge_index = torch.tensor(wm_edges).t().contiguous()
    wm_edge_label = torch.tensor(wm_labels)

    wm_data = Data(
        x=x,
        edge_index=train_data.edge_index,
        edge_label_index=wm_edge_index,
        edge_label=wm_edge_label
    )

    return wm_data, watermark_vector

def train(model, optimizer, train_data, wm_data, device):
    # Training with training data
    model.train()
    optimizer.zero_grad()
    logits_train = model(
        train_data.x.to(device),
        train_data.edge_index.to(device),
        train_data.edge_label_index.to(device)
    )
    loss_train = F.cross_entropy(logits_train, train_data.edge_label.long().to(device))
    loss_train.backward()
    optimizer.step()

    # Training with watermark data
    model.train()
    optimizer.zero_grad()
    logits_wm = model(
        wm_data.x.to(device),
        wm_data.edge_index.to(device),
        wm_data.edge_label_index.to(device)
    )
    loss_wm = F.cross_entropy(logits_wm, wm_data.edge_label.long().to(device))
    loss_wm.backward()
    optimizer.step()

    return loss_train.item(), loss_wm.item()

def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_label_index.to(device)
        )
        preds = logits.argmax(dim=1)
        labels = data.edge_label.long().to(device)
        acc = (preds == labels).sum().item() / labels.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
    return acc, auc

def estimate_kde(data):
    """
    Estimates the Kernel Density Estimation for the given data.
    """
    data = np.array(data)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    kde.fit(data)
    return kde

def sample_from_kde(kde, n_samples):
    """
    Samples data from the given KDE.
    """
    samples = kde.sample(n_samples)
    return samples.flatten()

def find_optimal_threshold(clean_samples, wm_samples):
    """
    Finds the optimal threshold that minimizes FPR and FNR.
    """
    thresholds = np.linspace(0, 1, 1000)
    min_error = float('inf')
    optimal_threshold = None

    for t in thresholds:
        fpr = np.mean(clean_samples > t)
        fnr = np.mean(wm_samples <= t)
        error = fpr + fnr
        if error < min_error:
            min_error = error
            optimal_threshold = t
    return optimal_threshold

def verify_watermark_kde(model, wm_data, clean_model_auc_distribution, wm_model_auc_distribution, device):
    """
    Verifies the watermark using KDE-based Dynamic Watermark Thresholding (DWT).
    """
    
    model.eval()
    with torch.no_grad():
        logits = model(
            wm_data.x.to(device),
            wm_data.edge_index.to(device),
            wm_data.edge_label_index.to(device)
        )
        labels = wm_data.edge_label.long().cpu().numpy()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(labels, probs)

    
    kde_clean = estimate_kde(clean_model_auc_distribution)
    kde_wm = estimate_kde(wm_model_auc_distribution)  
    n_samples = 10000
    clean_samples = sample_from_kde(kde_clean, n_samples)
    wm_samples = sample_from_kde(kde_wm, n_samples)
    optimal_threshold = find_optimal_threshold(clean_samples, wm_samples)

    is_verified = auc > optimal_threshold
    print(f'Watermark Verification AUC: {auc:.4f}')
    print(f'Dynamic Threshold: {optimal_threshold:.4f}')
    print(f'Watermark Verified: {is_verified}')

    return auc, is_verified

def fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4):
    attacked_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(attacked_model.parameters(), lr=lr)
    for epoch in range(epochs):
        attacked_model.train()
        optimizer.zero_grad()
        logits = attacked_model(
            train_data.x.to(device),
            train_data.edge_index.to(device),
            train_data.edge_label_index.to(device)
        )
        loss = F.cross_entropy(logits, train_data.edge_label.long().to(device))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Fine-Tuning Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    return attacked_model


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the data
    adj_matrix = load_usair_data()
    print("Data loaded successfully!")
    print(f"Number of nodes: {adj_matrix.shape[0]}")
    print(f"Number of edges: {adj_matrix.nnz}")

    # Creating PyG data object
    data = create_pyg_data(adj_matrix)
    print("PyG Data object created!")
    print(data)

    # Split the data
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,  # Ratio of negative to positive samples
    )
    train_data, val_data, test_data = transform(data)
    print(f"Training positive edges: {train_data.edge_label_index.size(1)//2}")
    print(f"Validation positive edges: {val_data.edge_label_index.size(1)//2}")
    print(f"Test positive edges: {test_data.edge_label_index.size(1)//2}")

    
    watermark_rate = 0.05  
    wm_data, watermark_vector = generate_watermark_data(train_data, watermark_rate=watermark_rate)

    # Initialize the model and optimizer
    model = GCNModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 400
    for epoch in range(1, epochs + 1):
        loss_train, loss_wm = train(model, optimizer, train_data, wm_data, device)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss_train:.4f}, Watermark Loss: {loss_wm:.4f}')

    
    train_acc, train_auc = evaluate_model(model, train_data, device)
    val_acc, val_auc = evaluate_model(model, val_data, device)
    test_acc, test_auc = evaluate_model(model, test_data, device)
    print(f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

    
    clean_model_auc_distribution = []
    num_clean_models = 10
    for _ in range(num_clean_models):
        clean_model = GCNModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
        clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.001)

        
        for epoch in range(1, epochs + 1):
            clean_model.train()
            clean_optimizer.zero_grad()
            logits = clean_model(
                train_data.x.to(device),
                train_data.edge_index.to(device),
                train_data.edge_label_index.to(device)
            )
            loss = F.cross_entropy(logits, train_data.edge_label.long().to(device))
            loss.backward()
            clean_optimizer.step()

        
        _, clean_wm_auc = evaluate_model(clean_model, wm_data, device)
        clean_model_auc_distribution.append(clean_wm_auc)

   
    wm_model_auc_distribution = []
    num_wm_models = 10
    for _ in range(num_wm_models):
        wm_model = GCNModel(in_channels=data.x.size(1), hidden_channels=256).to(device)
        wm_optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.001)
        wm_data_temp, _ = generate_watermark_data(train_data, watermark_rate=watermark_rate, watermark_vector=watermark_vector)

        # Training loop
        for epoch in range(1, epochs + 1):
            loss_train, loss_wm = train(wm_model, wm_optimizer, train_data, wm_data_temp, device)

        # Evaluate on watermark data
        _, wm_wm_auc = evaluate_model(wm_model, wm_data, device)
        wm_model_auc_distribution.append(wm_wm_auc)

    wm_auc, is_verified = verify_watermark_kde(
        model,
        wm_data,
        clean_model_auc_distribution,
        wm_model_auc_distribution,
        device
    )

    # Fine-tuning attack
    attacked_model = fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4)

    # Evaluate attacked model
    test_acc_attacked, test_auc_attacked = evaluate_model(attacked_model, test_data, device)
    print(f'Attacked Model - Test Accuracy: {test_acc_attacked:.4f}, Test AUC: {test_auc_attacked:.4f}')
    wm_acc_attacked, wm_auc_attacked = evaluate_model(attacked_model, wm_data, device)
    print(f'Attacked Model - Watermark Test Accuracy: {wm_acc_attacked:.4f}, Watermark Test AUC: {wm_auc_attacked:.4f}')

    # Verify watermark on attacked model 
    wm_auc_attacked, is_verified_attacked = verify_watermark_kde(
        attacked_model,
        wm_data,
        clean_model_auc_distribution,
        wm_model_auc_distribution,
        device
    )