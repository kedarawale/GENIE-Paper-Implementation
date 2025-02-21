import os
import pickle
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp

# Clear leftover GPU memory from any previous runs in same process
torch.cuda.empty_cache()

from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv  # <-- Replaced GCNConv with SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from node2vec import Node2Vec

# ---------------------------------------------------------------
# Adjust these hyperparameters to manage GPU usage:
# ---------------------------------------------------------------
NODE2VEC_DIM = 32        # Previously 64
GCN_HIDDEN_DIM = 64      # Previously 256
WATERMARK_RATE = 0.01    # Previously 0.05
NEG_SAMPLING_RATIO = 0.5 # Previously 1.0
EPOCHS = 400             # Previously 400
device = torch.device('cuda:0' or 'cuda:2')  # or 'cuda:2' depending on setup
# ---------------------------------------------------------------

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_astro_ph_data(file_path='CA-AstroPh.txt'):
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
            # Undirected: add both directions
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
            max_node_id = max(max_node_id, s, t)

    data_vals = [1.0] * len(src_list)
    adj_matrix = sp.coo_matrix(
        (data_vals, (src_list, tgt_list)),
        shape=(max_node_id + 1, max_node_id + 1)
    ).tocsr()
    return adj_matrix

def build_or_load_embeddings(adj_matrix, embedding_file='astro_ph_embeddings.npy', overwrite=False):
    if os.path.exists(embedding_file) and not overwrite:
        print(f"Loading embeddings from {embedding_file}...")
        embeddings = np.load(embedding_file)
    else:
        print("Computing new embeddings with Node2Vec...")
        G = nx.from_scipy_sparse_array(adj_matrix)
        node2vec = Node2Vec(
            G,
            dimensions=NODE2VEC_DIM,  # Reduced dimension
            walk_length=30,
            num_walks=200,
            workers=4,
            seed=seed
        )
        model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model_node2vec.wv[str(i)] for i in range(adj_matrix.shape[0])])
        np.save(embedding_file, embeddings)
        print(f"Embeddings saved to {embedding_file}.")
    return embeddings

def build_or_load_splits(data, split_file_path='astro_ph_splits.pkl', overwrite=False):
    if os.path.exists(split_file_path) and not overwrite:
        print(f"Loading data splits from {split_file_path}...")
        with open(split_file_path, 'rb') as f:
            splits_dict = pickle.load(f)
        train_data = splits_dict['train_data']
        val_data = splits_dict['val_data']
        test_data = splits_dict['test_data']
    else:
        print("Creating new train/val/test splits...")
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=NEG_SAMPLING_RATIO  # Reduced negative ratio
        )
        train_data, val_data, test_data = transform(data)
        splits_dict = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }
        with open(split_file_path, 'wb') as f:
            pickle.dump(splits_dict, f)
        print(f"Data splits saved to {split_file_path}.")
    return train_data, val_data, test_data

# ----------------------------------------------------------------------------
# Replaces GCN with GraphSAGE
# ----------------------------------------------------------------------------
class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

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
# ----------------------------------------------------------------------------

def generate_watermark_data(train_data, watermark_rate=0.01, watermark_vector=None):
    """
      FIX: Build a new adjacency for watermark edges so the aggregator
      truly sees inverted edges among watermark nodes. Everything 
      else remains unchanged.
    """
    x = train_data.x.clone()
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    if watermark_vector is None:
        torch.manual_seed(0)
        watermark_vector = torch.randn_like(x[0])

    # Inject watermark vector
    x[wm_nodes] = watermark_vector

    # Flip edges among watermark nodes
    edge_set = set((u.item(), v.item()) for u, v in train_data.edge_index.t())
    wm_edges, wm_labels = [], []

    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                wm_edges.append([i, j])
                wm_labels.append(0)
                edge_set.discard((i, j))
                edge_set.discard((j, i))
            else:
                wm_edges.append([i, j])
                wm_labels.append(1)
                edge_set.add((i, j))
                edge_set.add((j, i))

    # Watermark edges
    wm_edge_index = torch.tensor(wm_edges).t().contiguous()
    wm_edge_label = torch.tensor(wm_labels)

    # New aggregator adjacency that has flipped edges among wm_nodes
    wm_adj_list = list(edge_set)
    wm_adj_edge_index = torch.tensor(wm_adj_list, dtype=torch.long).t().contiguous()

    wm_data = Data(
        x=x,
        edge_index=wm_adj_edge_index,      # aggregator sees the flipped edges
        edge_label_index=wm_edge_index,    # the edges to classify
        edge_label=wm_edge_label
    )
    return wm_data, watermark_vector

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
    data = np.array(data)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    kde.fit(data)
    return kde

def sample_from_kde(kde_model, n_samples=5000):
    samples = kde_model.sample(n_samples)
    return samples.flatten()

def find_optimal_threshold(clean_samples, wm_samples):
    thresholds = np.linspace(0, 1, 500)
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
    model.eval()
    with torch.no_grad():
        logits = model(
            wm_data.x.to(device),
            wm_data.edge_index.to(device),
            wm_data.edge_label_index.to(device)
        )
        labels = wm_data.edge_label.long().cpu().numpy()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        auc_score = roc_auc_score(labels, probs)

    kde_clean = estimate_kde(clean_model_auc_distribution)
    kde_wm = estimate_kde(wm_model_auc_distribution)

    clean_samples = sample_from_kde(kde_clean)
    wm_samples = sample_from_kde(kde_wm)

    optimal_threshold = find_optimal_threshold(clean_samples, wm_samples)
    verified = auc_score > optimal_threshold
    print(f"Watermark Verification AUC on WM Data: {auc_score:.4f}")
    print(f"Dynamic Threshold: {optimal_threshold:.4f}")
    print(f"Ownership Verified? {verified}")
    return auc_score, verified

def fine_tune_attack(model, train_data, device, epochs=20, lr=1e-4):
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
        if (epoch + 1) % 5 == 0:
            print(f"Fine-Tuning Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return attacked_model

if __name__ == "__main__":
    # Attempt to free any stale allocations
    torch.cuda.empty_cache()

    # 1. Load adjacency matrix
    adj_matrix = load_astro_ph_data('CA-AstroPh.txt')
    print("Adjacency matrix loaded successfully.")
    print(f"Number of nodes: {adj_matrix.shape[0]}")
    print(f"Number of edges: {adj_matrix.nnz // 2} (undirected count)")

    # 2. Build or load embeddings
    embeddings = build_or_load_embeddings(
        adj_matrix,
        embedding_file='astro_ph_embeddings.npy',
        overwrite=False
    )

    # 3. Create PyG data object
    G = nx.from_scipy_sparse_array(adj_matrix)
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    # Convert embeddings to float tensor
    data.x = torch.tensor(embeddings, dtype=torch.float)

    # 4. Build or load splits
    train_data, val_data, test_data = build_or_load_splits(
        data,
        split_file_path='astro_ph_splits.pkl',
        overwrite=False
    )

    print(f"Train set #Edges: {train_data.edge_label_index.size(1)//2} positive edges")
    print(f"Val set #Edges: {val_data.edge_label_index.size(1)//2} positive edges")
    print(f"Test set #Edges: {test_data.edge_label_index.size(1)//2} positive edges")

    # 5. Generate watermark data
    wm_data, watermark_vector = generate_watermark_data(train_data, watermark_rate=WATERMARK_RATE)

    # 6. Initialize GraphSAGE model (replaces GCN)
    model = GraphSAGEModel(
        in_channels=data.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 7. Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        loss_train, loss_wm = train_step(model, optimizer, train_data, wm_data, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss_train:.4f}, WM Loss: {loss_wm:.4f}")

    # 8. Evaluate final model
    train_acc, train_auc = evaluate_model(model, train_data, device)
    val_acc, val_auc = evaluate_model(model, val_data, device)
    test_acc, test_auc = evaluate_model(model, test_data, device)
    print(f"Final Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")

    # 9. Build an empirical distribution for 'clean models'
    clean_model_auc_distribution = []
    num_clean_models = 2  # sample size for demonstration
    for _ in range(num_clean_models):
        clean_model = GraphSAGEModel(
            in_channels=data.x.size(1),
            hidden_channels=GCN_HIDDEN_DIM
        ).to(device)
        clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.001)
        for e in range(EPOCHS):
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
        _, wm_auc_clean = evaluate_model(clean_model, wm_data, device)
        clean_model_auc_distribution.append(wm_auc_clean)

    # 10. Build an empirical distribution for 'watermarked models'
    wm_model_auc_distribution = []
    num_wm_models = 2
    for _ in range(num_wm_models):
        temp_model = GraphSAGEModel(
            in_channels=data.x.size(1),
            hidden_channels=GCN_HIDDEN_DIM
        ).to(device)
        wm_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        wm_data_temp, _ = generate_watermark_data(train_data, watermark_rate=WATERMARK_RATE, watermark_vector=watermark_vector)
        for e in range(EPOCHS):
            train_step(temp_model, wm_optimizer, train_data, wm_data_temp, device)
        _, wm_auc_temp = evaluate_model(temp_model, wm_data, device)
        wm_model_auc_distribution.append(wm_auc_temp)

    # 11. Verify the watermark
    final_wm_auc, is_verified = verify_watermark_kde(
        model,
        wm_data,
        clean_model_auc_distribution,
        wm_model_auc_distribution,
        device
    )

    # 12. Demonstrate a fine-tuning attack
    print("\nAttempting a fine-tune attack to remove watermark...")
    attacked_model = fine_tune_attack(model, train_data, device, epochs=20, lr=1e-4)

    # Evaluate attacked model
    _, test_auc_attacked = evaluate_model(attacked_model, test_data, device)
    _, wm_auc_attacked = evaluate_model(attacked_model, wm_data, device)
    print(f"Attacked Model - Test AUC: {test_auc_attacked:.4f}")
    print(f"Attacked Model - WM AUC: {wm_auc_attacked:.4f}")

    # Re-verify watermark in attacked model
    attacked_wm_auc, is_verified_attacked = verify_watermark_kde(
        attacked_model,
        wm_data,
        clean_model_auc_distribution,
        wm_model_auc_distribution,
        device
    )

    print("Done.")