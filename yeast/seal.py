#!/usr/bin/env python3
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
WATERMARK_RATE = 0.04
NEG_SAMPLING_RATIO = 0.5
EPOCHS = 100

WM_LOSS_WEIGHT = 2.0  # Emphasize watermark edges
BOOTSTRAP_SAMPLES = 20000  # For smoothed bootstrap
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

########################################################################
# 1. Data Loading (yeast.edges)
########################################################################
def load_yeast_data(file_path='yeast.edges'):
    """
    Loads an undirected adjacency matrix from 'yeast.edges',
    with lines like:  2 36
                      2 857
                      2 1648
    Returns a CSR adjacency matrix (undirected).
    """
    src_list, tgt_list = [], []
    max_node_id = -1

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            s_str, t_str = line.split()
            s, t = int(s_str), int(t_str)
            src_list.append(s)
            tgt_list.append(t)
            if s != t:
                # Undirected
                src_list.append(t)
                tgt_list.append(s)
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
def build_or_load_embeddings(adj_matrix, embedding_file='yeast_embeddings.npy', overwrite=False):
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

def build_or_load_splits(data, split_file='yeast_splits.pkl', overwrite=False):
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
            is_undirected=True,
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
# 4. BFS-Based SEAL Subgraph Extraction + Labeling
#    (Replace the old single-graph GCN approach)
########################################################################
K_HOPS = 1
MAX_SUBGRAPH_SIZE = 50000
WM_FEATURE_DIM = 8           # watermark dimension
FULL_FEATURE_DIM = WM_FEATURE_DIM + 1  # BFS label col + watermark columns

def bfs_khop_subgraph(u, v, adj_csr, k=1, max_nodes=50000):
    """
    Extract up to k-hop BFS subgraph around (u,v).
    Returns the subgraph adjacency and a mapping old->new idx.
    """
    visited = set()
    queue = deque()
    queue.append((u, 0))
    queue.append((v, 0))
    visited.add(u)
    visited.add(v)

    while queue:
        cur, dist = queue.popleft()
        if dist >= k:
            continue
        row_s = adj_csr.indptr[cur]
        row_e = adj_csr.indptr[cur + 1]
        neighs = adj_csr.indices[row_s:row_e]
        for nb in neighs:
            if nb not in visited:
                visited.add(nb)
                if len(visited) >= max_nodes:
                    break
                queue.append((nb, dist + 1))
        if len(visited) >= max_nodes:
            break

    sub_nodes = sorted(list(visited))
    node_map = {old_idx: i for i, old_idx in enumerate(sub_nodes)}
    row, col, data = [], [], []
    for old_n in sub_nodes:
        row_s = adj_csr.indptr[old_n]
        row_e = adj_csr.indptr[old_n + 1]
        neighs = adj_csr.indices[row_s:row_e]
        for nb in neighs:
            if nb in node_map:
                row.append(node_map[old_n])
                col.append(node_map[nb])
                data.append(1.0)
    sub_adj = sp.coo_matrix(
        (data, (row, col)),
        shape=(len(sub_nodes), len(sub_nodes))
    ).tocsr()
    return sub_nodes, sub_adj, node_map

def compute_bfs_dist(sub_csr, start_idx):
    dist = np.full(sub_csr.shape[0], fill_value=999999, dtype=int)
    dist[start_idx] = 0
    dq = deque([start_idx])
    while dq:
        cur = dq.popleft()
        cd = dist[cur]
        row_s = sub_csr.indptr[cur]
        row_e = sub_csr.indptr[cur + 1]
        neighs = sub_csr.indices[row_s:row_e]
        for nb in neighs:
            if dist[nb] > cd + 1:
                dist[nb] = cd + 1
                dq.append(nb)
    return dist

def label_nodes_bfs(sub_adj, src_idx, dst_idx):
    dist_s = compute_bfs_dist(sub_adj, src_idx)
    dist_d = compute_bfs_dist(sub_adj, dst_idx)
    dist_min = np.minimum(dist_s, dist_d)
    labels = dist_min + 1
    labels[src_idx] = 1
    labels[dst_idx] = 2
    return labels

def build_subgraphs(edge_index, edge_label, adjacency, k=K_HOPS):
    """
    Build BFS-based subgraphs from (edge_index, edge_label).
    BFS label is in x[:,0]; x[:,1..] are for watermark usage.
    """
    edges = edge_index.t().tolist()
    labs = edge_label.tolist()
    subg_list = []
    for (u, v), y in zip(edges, labs):
        sub_nodes, sub_adj, node_map = bfs_khop_subgraph(u, v, adjacency, k, MAX_SUBGRAPH_SIZE)
        if (u not in node_map) or (v not in node_map):
            continue
        src_idx = node_map[u]
        dst_idx = node_map[v]
        node_labels = label_nodes_bfs(sub_adj, src_idx, dst_idx)

        n_sub = len(sub_nodes)
        x = torch.zeros((n_sub, FULL_FEATURE_DIM), dtype=torch.float)
        # BFS label in col 0
        x[:, 0] = torch.tensor(node_labels, dtype=torch.float)

        coo = sub_adj.tocoo()
        sub_edge_index = np.vstack([coo.row, coo.col])
        data_obj = Data()
        data_obj.x = x
        data_obj.edge_index = torch.tensor(sub_edge_index, dtype=torch.long)
        data_obj.y = torch.tensor([y], dtype=torch.float)
        data_obj.num_nodes = n_sub
        subg_list.append(data_obj)
    return subg_list

########################################################################
# 5. Watermark Injection by Flipping Subgraph Labels
########################################################################
def build_watermark_subgraphs(subgraphs, wm_rate=WATERMARK_RATE):
    n = len(subgraphs)
    num_wm = int(n * wm_rate)
    chosen = random.sample(range(n), num_wm)

    wm_vec = torch.randn(WM_FEATURE_DIM)
    newlist = []
    for i, g in enumerate(subgraphs):
        newg = copy.deepcopy(g)
        if i in chosen:
            old_label = newg.y.item()
            flipped = 1.0 - old_label
            newg.y = torch.tensor([flipped], dtype=torch.float)
            # Overwrite columns 1..end with watermark vector
            for row_i in range(newg.x.size(0)):
                newg.x[row_i, 1:] = wm_vec
        newlist.append(newg)
    return newlist

########################################################################
# 6. DGCNN with Sort-Pool (subgraph-based link prediction)
########################################################################
class SEALSubgraphDataset(Dataset):
    def __init__(self, subgraphs):
        super().__init__()
        self.subgraphs = subgraphs
    def __len__(self):
        return len(self.subgraphs)
    def __getitem__(self, idx):
        return self.subgraphs[idx]

def collate_fn(batch_list):
    x_list, edge_index_list, y_list = [], [], []
    batch_vec = []
    current_offset = 0
    for i, gdata in enumerate(batch_list):
        x_list.append(gdata.x)
        shifted = gdata.edge_index + current_offset
        edge_index_list.append(shifted)
        y_list.append(gdata.y)
        num_nodes = gdata.x.size(0)
        batch_vec.append(torch.full((num_nodes,), i, dtype=torch.long))
        current_offset += num_nodes

    x_cat = torch.cat(x_list, dim=0)
    edge_cat = torch.cat(edge_index_list, dim=1)
    y_cat = torch.cat(y_list, dim=0)
    b_cat = torch.cat(batch_vec, dim=0)
    return (x_cat, edge_cat, b_cat, y_cat)

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_dim=32, sort_k=30):
        super().__init__()
        in_dim = FULL_FEATURE_DIM
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.sort_k = sort_k

        self.conv1d_1 = torch.nn.Conv1d(hidden_dim, 16, kernel_size=5)
        self.conv1d_2 = torch.nn.Conv1d(16, 32, kernel_size=5)
        self.lin1 = torch.nn.Linear(32 * (sort_k - 8), 128)
        self.lin2 = torch.nn.Linear(128, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = self.global_sort_pool(x, batch)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        b, c, t = x.size()
        x = x.view(b, c*t)
        x = F.relu(self.lin1(x))
        out = self.lin2(x).view(-1)
        return out

    def global_sort_pool(self, x, batch):
        # Sort each graph's nodes by the last feature dimension (descending).
        num_graphs = batch.max().item() + 1
        hidden_dim = x.size(1)
        out = x.new_zeros(num_graphs, self.sort_k, hidden_dim)
        for i in range(num_graphs):
            mask = (batch == i)
            x_i = x[mask]
            if x_i.size(0) == 0:
                continue
            sort_key = x_i[:, -1]  # last dimension
            _, idx = torch.sort(sort_key, descending=True)
            x_i_sorted = x_i[idx]
            take = min(x_i.size(0), self.sort_k)
            out[i, :take, :] = x_i_sorted[:take, :]
        return out

########################################################################
# 7. Two-Pass Training & Evaluation for Watermarked Subgraphs
########################################################################
def train_two_passes(model, optimizer, loader_clean, loader_wm):
    model.train()
    # 1) Process normal subgraphs
    total_loss_clean = 0.0
    total_count_clean = 0
    for batch in loader_clean:
        x, eidx, bvec, y = (b.to(device) for b in batch)
        optimizer.zero_grad()
        out = model(x, eidx, bvec)
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()
        total_loss_clean += loss.item() * y.size(0)
        total_count_clean += y.size(0)
    norm_loss = total_loss_clean / total_count_clean if total_count_clean > 0 else 0.0

    # 2) Process watermark subgraphs
    total_loss_wm = 0.0
    total_count_wm = 0
    for batch in loader_wm:
        x, eidx, bvec, y = (b.to(device) for b in batch)
        optimizer.zero_grad()
        out = model(x, eidx, bvec)
        loss = F.binary_cross_entropy_with_logits(out, y)
        # Multiply the loss by the watermark weight before backpropagating.
        loss_wm = loss * WM_LOSS_WEIGHT
        loss_wm.backward()
        optimizer.step()
        total_loss_wm += loss_wm.item() * y.size(0)
        total_count_wm += y.size(0)
    wm_loss = total_loss_wm / total_count_wm if total_count_wm > 0 else 0.0

    # Return both losses as a tuple
    return norm_loss, wm_loss

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        x, eidx, bvec, y = (b.to(device) for b in batch)
        out = model(x, eidx, bvec)
        prob = torch.sigmoid(out).cpu().numpy()
        lbl = y.cpu().numpy()
        all_preds.append(prob)
        all_labels.append(lbl)
    if len(all_preds) == 0:
        return 0.5
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    if len(np.unique(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_preds)

########################################################################
# 8. Dynamic Thresholding + Smoothed Bootstrap
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

def verify_watermark_kde_final(model, wm_data_loader, clean_auc_dist, wm_auc_dist):
    # Evaluate model on watermark subgraphs
    auc_score = eval_model(model, wm_data_loader)
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
# 9. Minimal Judge Class (Registry + Dispute)
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

    def dispute_ownership(self, owner_name, suspect_model, wm_loader, clean_auc_dist, wm_auc_dist):
        if owner_name not in self.registry:
            print(f"Judge: No record found for owner {owner_name}")
            return
        print(f"Judge: Found record for {owner_name}, verifying watermark ...")
        auc_score, verified, p_val, threshold = verify_watermark_kde_final(
            suspect_model, wm_loader, clean_auc_dist, wm_auc_dist
        )
        if verified:
            print(f"Judge: Verified ownership for {owner_name}, p-value={p_val:.4f}")
        else:
            print(f"Judge: Ownership not verified. threshold={threshold:.4f}")

########################################################################
# 10. Fine-Tune Attack
########################################################################
def fine_tune_attack(model, train_loader, device, epochs=50, lr=1e-4):
    print("\nFine-tuning attack ...")
    attacked_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(attacked_model.parameters(), lr=lr)
    for ep in range(epochs):
        attacked_model.train()
        for batch in train_loader:
            x, eidx, bvec, y = (b.to(device) for b in batch)
            optimizer.zero_grad()
            logits = attacked_model(x, eidx, bvec)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
    return attacked_model

########################################################################
# Main
########################################################################
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # 1) Load adjacency
    adj_matrix = load_yeast_data('yeast.edges')
    print("Adjacency loaded.  Nodes:", adj_matrix.shape[0])
    print("Number of edges (undirected):", adj_matrix.nnz // 2)

    # 2) Embeddings (optional for BFS approach, we show anyway)
    embeddings = build_or_load_embeddings(adj_matrix, 'yeast_embeddings.npy', overwrite=False)

    # 3) PyG data
    G_nx = nx.from_scipy_sparse_array(adj_matrix)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)

    train_data, val_data, test_data = build_or_load_splits(data_full, 'yeast_splits.pkl', overwrite=False)

    # Convert them to BFS-based subgraphs
    train_subg = build_subgraphs(train_data.edge_label_index, train_data.edge_label, adj_matrix, K_HOPS)
    wm_subg = build_watermark_subgraphs(train_subg, WATERMARK_RATE)
    val_subg = build_subgraphs(val_data.edge_label_index, val_data.edge_label, adj_matrix, K_HOPS)
    test_subg = build_subgraphs(test_data.edge_label_index, test_data.edge_label, adj_matrix, K_HOPS)

    ds_train_clean = SEALSubgraphDataset(train_subg)
    ds_train_wm = SEALSubgraphDataset(wm_subg)
    ds_val = SEALSubgraphDataset(val_subg)
    ds_test = SEALSubgraphDataset(test_subg)

    loader_train_clean = DataLoader(ds_train_clean, batch_size=32, shuffle=True, collate_fn=collate_fn)
    loader_train_wm = DataLoader(ds_train_wm, batch_size=32, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False, collate_fn=collate_fn)
    loader_test = DataLoader(ds_test, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Define BFS-based DGCNN model
    model = DGCNN(hidden_dim=32, sort_k=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Training for {EPOCHS} epochs...")
    for ep in range(1, EPOCHS + 1):
        norm_loss, wm_loss = train_two_passes(model, optimizer, loader_train_clean, loader_train_wm)
        if ep % 50 == 0:
            print(f"Epoch {ep}, Train Loss: {norm_loss:.4f}, WM Loss: {wm_loss:.4f}")

    # Evaluate final performance on train (clean), validation and test sets
    train_auc = eval_model(model, loader_train_clean)
    val_auc = eval_model(model, loader_val)
    test_auc = eval_model(model, loader_test)
    wm_auc_final = eval_model(model, loader_train_wm)
    print(f"Final Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Watermark Verification AUC on WM Data: {wm_auc_final:.4f}")

    # Minimal example of how you'd do bootstrapping or judge steps:
    # (We skip building multiple “clean vs. watermark” distributions in detail.)
    clean_model_auc_dist = [0.5, 0.6]  # placeholder
    wm_model_auc_dist = [0.9, 0.95]    # placeholder

    # Setting up a judge
    judge = Judge()
    judge.register_model("YeastModelOwner", {"secret_info": "my_watermark"})
    # In a real scenario you'd store the entire subgraph set or watermark vector

    # Suppose an attacker tried to fine-tune
    attacked_model = fine_tune_attack(model, loader_train_clean, device, epochs=50, lr=1e-4)
    # Judge tries to see if it's still watermarked:
    loader_wm_only = DataLoader(ds_train_wm, batch_size=32, shuffle=False, collate_fn=collate_fn)
    judge.dispute_ownership("YeastModelOwner", attacked_model, loader_wm_only,
                            clean_model_auc_dist, wm_model_auc_dist)

   