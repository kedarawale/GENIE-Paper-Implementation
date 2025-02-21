import os
import pickle
import random
import time
import datetime
import copy
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
import node2vec  # local node2vec
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from collections import deque

########################################################################
# Hyperparameters & Device
########################################################################
NODE2VEC_DIM = 256
GCN_HIDDEN_DIM = 256
WATERMARK_RATE = 0.125
NEG_SAMPLING_RATIO = 0.5
EPOCHS = 400

WM_LOSS_WEIGHT = 2.0  # Emphasize watermark edges
BOOTSTRAP_SAMPLES = 20000  # For smoothed bootstrap
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

########################################################################
# 1. Data Loading (CA-AstroPh.txt)
########################################################################
def load_astro_data(file_path='CA-AstroPh.txt'):
    """
    Loads a directed adjacency matrix from 'CA-AstroPh.txt',
    with lines like:  FromNodeId	ToNodeId
                      84424	276
                      ...
    Returns a CSR adjacency matrix (directed).
    """
    src_list, tgt_list = [], []
    max_node_id = -1

    with open(file_path, 'r') as f:
        f.readline()  # Skip header line
        f.readline()  # Skip comment line
        f.readline()  # Skip comment line
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            s_str, t_str = line.split('\t') # Tab separated
            s, t = int(s_str), int(t_str)
            src_list.append(s)
            tgt_list.append(t)
            max_node_id = max(max_node_id, s, t)

    data_vals = [1.0] * len(src_list)
    adj_matrix = sp.coo_matrix(
        (data_vals, (src_list, tgt_list)),
        shape=(max_node_id + 1, max_node_id + 1)
    ).tocsr()
    return adj_matrix

########################################################################
# 2. Build or Load Node Embeddings (Node2Vec)
########################################################################
def build_or_load_embeddings(adj_matrix, embedding_file='astroph_embeddings.npy', overwrite=False):
    if os.path.exists(embedding_file) and not overwrite:
        print(f"Loading embeddings from {embedding_file}...")
        embeddings = np.load(embedding_file)
    else:
        print("Computing new embeddings with Node2Vec...")
        G_nx = nx.from_scipy_sparse_array(adj_matrix)
        node2vec = node2vec.Node2Vec(
            G_nx,
            dimensions=NODE2VEC_DIM,
            walk_length=30,
            num_walks=200,
            workers=4,
            seed=seed
        )
        model_n2v = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model_n2v.wv[str(i)] for i in range(adj_matrix.shape[0])])
        np.save(embedding_file, embeddings)
        print(f"Saved embeddings to {embedding_file}.")
    return embeddings

########################################################################
# 3. Build or Load Splits (with checking for 2-class subsets)
########################################################################
def _has_two_classes(split_data):
    labels = split_data.edge_label.cpu().numpy()
    unique_labels = np.unique(labels)
    return (len(unique_labels) >= 2)

def build_or_load_splits(data, split_file='astroph_splits.pkl', overwrite=False):
    if os.path.exists(split_file) and not overwrite:
        print(f"Loading data splits from {split_file}...")
        with open(split_file, 'rb') as f:
            dct = pickle.load(f)
        train_data = dct['train_data']
        val_data = dct['val_data']
        test_data = dct['test_data']
    else:
        print("Creating new train/val/test splits...")
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=False, # Directed Graph
            add_negative_train_samples=True,
            neg_sampling_ratio=NEG_SAMPLING_RATIO
        )
        max_tries = 20
        found_good = False
        for _ in range(max_tries):
            train_data, val_data, test_data = transform(data)
            if (_has_two_classes(train_data) and
                _has_two_classes(val_data) and
                _has_two_classes(test_data)):
                found_good = True
                break
        if not found_good:
            raise RuntimeError("Cannot create splits with 2 classes in each subset after multiple tries.")

        with open(split_file, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data
            }, f)
        print(f"Data splits saved to {split_file}.")

    return train_data, val_data, test_data

########################################################################
# 4. GCN Encoder, Decoder and Model (No Changes)
########################################################################
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

########################################################################
# 5. Watermark Injection by Modifying Node Features and Edges (No Changes)
########################################################################
def generate_watermark_data(train_data, watermark_rate=WATERMARK_RATE, watermark_vector=None):
    """
    Generates a watermark version of the training data
    by modifying node features for watermarked nodes
    and flipping edges between these nodes.
    """
    x = train_data.x.clone()
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    if watermark_vector is None:
        torch.manual_seed(0)
        watermark_vector = torch.randn_like(x[0])

    x[wm_nodes] = watermark_vector

    edge_set = set((u.item(), v.item()) for u, v in train_data.edge_index.t())
    wm_edges, wm_labels = [], []
    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                wm_edges.append([i, j])
                wm_labels.append(0) # Label 0 for flipped existing edge (negative example)
                edge_set.discard((i, j))
                edge_set.discard((j, i))
            else:
                wm_edges.append([i, j])
                wm_labels.append(1) # Label 1 for added non-existing edge (positive example)
                edge_set.add((i, j))
                edge_set.add((j, i))

    wm_edge_index = torch.tensor(wm_edges).t().contiguous()
    wm_edge_label = torch.tensor(wm_labels)
    wm_data = Data(
        x=x,
        edge_index=train_data.edge_index, # Keep original edge_index
        edge_label_index=wm_edge_index,
        edge_label=wm_edge_label
    )
    return wm_data, watermark_vector

########################################################################
# 6. Training & Evaluation Functions (for GCN) (No Changes)
########################################################################
def train_step(model, optimizer, train_data, wm_data, device):
    model.train()
    optimizer.zero_grad()
    # 1. Train on normal data
    logits_train = model(
        train_data.x.to(device),
        train_data.edge_index.to(device),
        train_data.edge_label_index.to(device)
    )
    loss_train = F.cross_entropy(
        logits_train, train_data.edge_label.long().to(device)
    )
    loss_train.backward()
    optimizer.step()

    # 2. Train on watermark data
    model.train()
    optimizer.zero_grad()
    logits_wm = model(
        wm_data.x.to(device),
        wm_data.edge_index.to(device),
        wm_data.edge_label_index.to(device)
    )
    loss_wm = F.cross_entropy(
        logits_wm, wm_data.edge_label.long().to(device)
    )
    loss_wm.backward()
    optimizer.step()

    return loss_train.item(), loss_wm.item()

def eval_model(model, data, device):
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
        # Handle single-class scenario to avoid ValueError in roc_auc_score
        unique_labels = np.unique(labels_np)
        if len(unique_labels) < 2:
            auc = 1.0
        else:
            auc = roc_auc_score(labels_np, probs)
    return acc, auc

########################################################################
# 7. Dynamic Thresholding + Smoothed Bootstrap (same as SEAL code) (No Changes)
########################################################################
def kernel_density_estimate(data_arr):
    arr = np.array(data_arr)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    kde.fit(arr)
    return kde

def sample_from_kde(kde_model, n=5000):
    samples = kde_model.sample(n)
    return samples.flatten()

def find_threshold_iterative(clean_samples, wm_samples):
    combined = np.concatenate([clean_samples, wm_samples])
    best_threshold = None
    best_error = float('inf')
    for t in np.linspace(combined.min(), combined.max(), 500):
        fpr = np.mean(clean_samples > t)
        fnr = np.mean(wm_samples <= t)
        err = fpr + fnr
        if err < best_error:
            best_error = err
            best_threshold = t
    return best_threshold

def smoothed_bootstrap_pvalue(clean_scores, wm_scores):
    obs_diff = np.mean(wm_scores) - np.mean(clean_scores)
    combined = np.concatenate([clean_scores, wm_scores])
    kde = kernel_density_estimate(combined)

    count_greater = 0
    n1 = len(clean_scores)
    n2 = len(wm_scores)
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_all = kde.sample(n1 + n2).flatten()
        sample_clean = sample_all[:n1]
        sample_wm = sample_all[n1:]
        diff = np.mean(sample_wm) - np.mean(sample_clean)
        if diff >= obs_diff:
            count_greater += 1
    p_val = count_greater / BOOTSTRAP_SAMPLES
    return p_val

def verify_watermark_kde_final(model, wm_data, clean_auc_dist, wm_auc_dist):
    # Evaluate model on watermark subgraphs
    _, auc_score = eval_model(model, wm_data, device)
    # Then do threshold/p-value logic
    kde_clean = kernel_density_estimate(clean_auc_dist)
    kde_wm = kernel_density_estimate(wm_auc_dist)
    c_samps = sample_from_kde(kde_clean, 5000)
    w_samps = sample_from_kde(kde_wm, 5000)
    threshold = find_threshold_iterative(c_samps, w_samps)
    verified = (auc_score > threshold)
    p_val = smoothed_bootstrap_pvalue(clean_auc_dist, wm_auc_dist)
    print(f"\nWatermark Verification AUC = {auc_score:.4f}")
    print(f"Dynamic Threshold = {threshold:.4f}")
    print(f"Ownership Verified : {verified}")
    print(f"Smoothed bootstrap p-value = {p_val:.4f}")
    return auc_score, verified, p_val, threshold

########################################################################
# 8. Minimal Judge Class (Registry + Dispute) (same as SEAL code) (No Changes)
########################################################################
class Judge:
    def __init__(self):
        self.registry = {}

    def register_model(self, owner_name, wm_info):
        t_now = time.time()
        Standard_datetime = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d %H:%M:%S")
        entry = {

            'owner': owner_name,
            'timestamp': t_now,
            'watermark_info': wm_info
        }
        self.registry[owner_name] = entry
        print(f"Judge: Registered model for {owner_name} at timestamp={Standard_datetime}")

    def dispute_ownership(self, owner_name, suspect_model, wm_data, clean_auc_dist, wm_auc_dist):
        if owner_name not in self.registry:
            print(f"Judge: No record found for owner {owner_name}")
            return
        print(f"Judge: Found record for {owner_name}, verifying watermark ...")
        auc_score, verified, p_val, threshold = verify_watermark_kde_final(
            suspect_model, wm_data, clean_auc_dist, wm_auc_dist
        )
        if verified:
            print(f"Judge: Verified ownership for {owner_name}, p-value={p_val:.4f}")
        else:
            print(f"Judge: Ownership not verified. threshold={threshold:.4f}")

########################################################################
# 9. Fine-Tune Attack (same as SEAL code, but using GCN eval) (No Changes)
########################################################################
def fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4):
    print("\nFine-tuning attack ...")
    attacked_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(attacked_model.parameters(), lr=lr)
    for ep in range(epochs):
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
    return attacked_model

########################################################################
# 10. Main (updated for CA-AstroPh dataset)
########################################################################
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # 1) Load adjacency
    adj_matrix = load_astro_data('CA-AstroPh.txt') # Updated data loading function and file
    print("Nodes:", adj_matrix.shape[0])
    print("Number of edges:", adj_matrix.nnz) # For directed graph, nnz is the actual number of edges

    # 2) Embeddings (optional for BFS approach, we show anyway)
    embeddings = build_or_load_embeddings(adj_matrix, 'astroph_embeddings.npy', overwrite=False) # Updated embedding file name

    # 3) PyG data
    G_nx = nx.from_scipy_sparse_array(adj_matrix)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)

    train_data, val_data, test_data = build_or_load_splits(data_full, 'astroph_splits.pkl', overwrite=False) # Updated splits file name

    # 4) Generate watermark data
    wm_data, watermark_vector = generate_watermark_data(train_data, watermark_rate=WATERMARK_RATE)

    # 5) Define GCN model
    model = GCNModel(
        in_channels=data_full.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training for {EPOCHS} epochs...")
    for ep in range(1, EPOCHS + 1):
        norm_loss, wm_loss = train_step(model, optimizer, train_data, wm_data, device)
        if ep % 50 == 0:
            print(f"Epoch {ep}, Train Loss: {norm_loss:.4f}, WM Loss: {wm_loss:.4f}")

    # 6) Evaluate final performance on train (clean), validation and test sets
    train_acc, train_auc = eval_model(model, train_data, device)
    val_acc, val_auc = eval_model(model, val_data, device)
    test_acc, test_auc = eval_model(model, test_data, device)
    wm_acc_final, wm_auc_final = eval_model(model, wm_data, device)
    print(f"Final Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Watermark Verification AUC on WM Data: {wm_auc_final:.4f}")

    # 7) Minimal example of how you'd do bootstrapping or judge steps:
    # (We skip building multiple “clean vs. watermark” distributions in detail.)
    clean_model_auc_dist = [0.5, 0.6]  # placeholder
    wm_model_auc_dist = [0.9, 0.95]    # placeholder

    # Setting up a judge
    judge = Judge()
    judge.register_model("AstroPh_GCNModelOwner", {"secret_info": "my_watermark"}) # Updated owner name
    # In a real scenario you'd store the entire subgraph set or watermark vector

    # Suppose an attacker tried to fine-tune
    attacked_model = fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4)
    # Judge tries to see if it's still watermarked:
    judge.dispute_ownership("AstroPh_GCNModelOwner", attacked_model, wm_data, # Updated owner name
                            clean_model_auc_dist, wm_model_auc_dist)