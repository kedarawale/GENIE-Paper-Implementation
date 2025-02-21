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

def load_infpower_data(file_path='inf-power.mtx'):
    src_list, tgt_list = [], []
    max_node_id = -1

    with open(file_path, 'r') as f:
        # Skip header lines that start with '%' (MatrixMarket lines)
        first_line_encountered = False
        edges_to_read = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if not first_line_encountered:
                # The dimension line: rows cols nnz
                parts = line.split()
                # rows = int(parts[0])  # we could parse these if needed
                # cols = int(parts[1])
                edges_to_read = int(parts[2])
                first_line_encountered = True
                continue
            # Read each subsequent line as an edge
            s_str, t_str = line.split()
            s, t = int(s_str), int(t_str)
            src_list.append(s)
            tgt_list.append(t)
            # Undirected: add both directions if not self-loop
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
            max_node_id = max(max_node_id, s, t)

            edges_to_read -= 1
            if edges_to_read == 0:
                break

    data_vals = [1.0] * len(src_list)
    adj_matrix = sp.coo_matrix(
        (data_vals, (src_list, tgt_list)),
        shape=(max_node_id + 1, max_node_id + 1)
    ).tocsr()
    return adj_matrix

def build_or_load_embeddings(adj_matrix, embedding_file='inf_power_embeddings.npy', overwrite=False):
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

def build_or_load_splits(data, split_file_path='inf_power_splits.pkl', overwrite=False):
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

#####################################################################
#                START: NeoGNN Model Definition
#####################################################################

def build_adj_list(edge_index, num_nodes):
    """
    Build an adjacency list from a given edge_index for CPU-based neighbor lookups.

    Args:
        edge_index (Tensor): [2, E]
        num_nodes (int): Number of nodes
    Returns:
        List[List[int]]: adjacency_list[node] = [list_of_neighbors]
    """
    row, col = edge_index
    row = row.cpu().numpy()
    col = col.cpu().numpy()

    adjacency_list = [[] for _ in range(num_nodes)]
    for i in range(len(row)):
        src = row[i]
        dst = col[i]
        adjacency_list[src].append(dst)
    return adjacency_list

class NEOFunction(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(NEOFunction, self).__init__()
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, h_u, h_v, h_N_u, h_N_v):
        """
        Computes the NEO score between pairs of nodes.
        """
        overlap_feature = h_N_u * h_N_v
        neo_score = torch.sigmoid(self.linear(overlap_feature))
        return neo_score

class NeoGNNLayer(torch.nn.Module):
    """
    Combines multiple GNN layers (GCN, SAGE, GIN, GAT) into one layer by summing their outputs.
    """
    def __init__(self, in_channels, out_channels):
        super(NeoGNNLayer, self).__init__()
        from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
        self.gcn = GCNConv(in_channels, out_channels)
        self.sage = SAGEConv(in_channels, out_channels)
        self.gin = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        ))
        self.gat = GATConv(in_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x1 = self.gcn(x, edge_index)
        x2 = self.sage(x, edge_index)
        x3 = self.gin(x, edge_index)
        x4 = self.gat(x, edge_index)
        x = x1 + x2 + x3 + x4
        x = F.relu(x)
        return x

class NeoGNN(torch.nn.Module):
    """
    The main NeoGNN model as described in the conversation.
    """
    def __init__(self, in_channels, hidden_channels):
        super(NeoGNN, self).__init__()
        self.layer1 = NeoGNNLayer(in_channels, hidden_channels)
        self.layer2 = NeoGNNLayer(hidden_channels, hidden_channels)
        self.layer3 = NeoGNNLayer(hidden_channels, hidden_channels)
        self.neo_function = NEOFunction(hidden_channels)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 2)
        )

        # We'll store adjacency list on CPU to avoid large memory usage
        self.adjacency_list = None
        self.num_nodes = None

    def encode(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        return x

    def prepare_adj_list(self, edge_index, num_nodes):
        """
        Build adjacency list and store it internally.
        Called once from outside after model creation.
        """
        self.num_nodes = num_nodes
        self.adjacency_list = build_adj_list(edge_index, num_nodes)

    def get_neighbor_embeddings(self, z, nodes):
        """
        Efficient neighbor embeddings on CPU adjacency list;
        no adjacency matrix -> dense conversion.
        """
        neighbor_embeddings = []
        for node in nodes:
            neighbors = self.adjacency_list[node]
            if len(neighbors) == 0:
                neighbor_embeddings.append(torch.zeros(z.size(1), device=z.device))
            else:
                # gather embeddings for these neighbors
                neigh_embs = z[neighbors]
                mean_emb = neigh_embs.mean(dim=0)
                neighbor_embeddings.append(mean_emb)
        neighbor_embeddings = torch.stack(neighbor_embeddings)
        return neighbor_embeddings

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        h_u = z[src]
        h_v = z[dst]
        # Compute neighbor embeddings using adjacency list
        h_N_u = self.get_neighbor_embeddings(z, src)
        h_N_v = self.get_neighbor_embeddings(z, dst)
        neo_scores = self.neo_function(h_u, h_v, h_N_u, h_N_v)
        h = torch.cat([h_u, h_v, neo_scores], dim=1)
        return self.decoder(h)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_label_index)
        return logits

#####################################################################
#                END: NeoGNN Model Definition
#####################################################################

def generate_watermark_data(train_data, watermark_rate=0.01, watermark_vector=None):
    """
    Produces a watermark dataset from train_data by injecting a secret watermark vector
    into a random subset of nodes and then flipping their edges.
    """
    x = train_data.x.clone()
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    if watermark_vector is None:
        torch.manual_seed(0)
        watermark_vector = torch.randn_like(x[0])

    # Overwrite selected nodes with the watermark vector
    x[wm_nodes] = watermark_vector

    # Flip edges inside the watermark nodes
    edge_set = set((u.item(), v.item()) for u, v in train_data.edge_index.t())
    wm_edges, wm_labels = [], []
    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                # If edge existed, label is 0, remove it
                wm_edges.append([i, j])
                wm_labels.append(0)
                edge_set.discard((i, j))
                edge_set.discard((j, i))
            else:
                # If edge not existed, label is 1, add it
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

def train_step(model, optimizer, train_data, wm_data, device):
    """
    Training step that first trains on normal data, then on watermark data.
    """
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
    """
    Evaluate model on a given dataset and compute accuracy & AUC.
    """
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
    Fits a Kernel Density Estimator (KDE) on given data.
    """
    data = np.array(data)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
    kde.fit(data)
    return kde

def sample_from_kde(kde_model, n_samples=5000):
    """
    Draw samples from a given KDE model.
    """
    samples = kde_model.sample(n_samples)
    return samples.flatten()

def find_optimal_threshold(clean_samples, wm_samples):
    """
    Search for a threshold that attempts to minimize FPR + FNR.
    """
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
    """
    Watermark verification using a dynamic threshold approach (KDE-based).
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
    """
    Demonstrate a fine-tuning style watermark removal attack.
    """
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
    adj_matrix = load_infpower_data('inf-power.mtx')
    print("Adjacency matrix loaded successfully.")
    print(f"Number of nodes: {adj_matrix.shape[0]}")
    print(f"Number of edges: {adj_matrix.nnz // 2} (undirected count)")

    # 2. Build or load embeddings
    embeddings = build_or_load_embeddings(
        adj_matrix,
        embedding_file='inf_power_embeddings.npy',
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
        split_file_path='inf_power_splits.pkl',
        overwrite=False
    )

    print(f"Train set #Edges: {train_data.edge_label_index.size(1)//2} positive edges")
    print(f"Val set #Edges: {val_data.edge_label_index.size(1)//2} positive edges")
    print(f"Test set #Edges: {test_data.edge_label_index.size(1)//2} positive edges")

    # 5. Generate watermark data
    wm_data, watermark_vector = generate_watermark_data(train_data, watermark_rate=WATERMARK_RATE)

    # 6. Initialize NeoGNN model
    model = NeoGNN(
        in_channels=data.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)

    # Pre-build adjacency list once for neighbor embeddings
    model.prepare_adj_list(train_data.edge_index, data.x.size(0))

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
    #    (Models trained WITHOUT watermark)
    clean_model_auc_distribution = []
    num_clean_models = 2  # sample size for demonstration
    for _ in range(num_clean_models):
        clean_model = NeoGNN(
            in_channels=data.x.size(1),
            hidden_channels=GCN_HIDDEN_DIM
        ).to(device)
        # also build adjacency list for the clean model
        clean_model.prepare_adj_list(train_data.edge_index, data.x.size(0))

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

        # Evaluate the "clean" model on watermark data
        _, wm_auc_clean = evaluate_model(clean_model, wm_data, device)
        clean_model_auc_distribution.append(wm_auc_clean)

    # 10. Build an empirical distribution for 'watermarked models'
    wm_model_auc_distribution = []
    num_wm_models = 2
    for _ in range(num_wm_models):
        temp_model = NeoGNN(
            in_channels=data.x.size(1),
            hidden_channels=GCN_HIDDEN_DIM
        ).to(device)
        # adjacency list for watermarked
        temp_model.prepare_adj_list(train_data.edge_index, data.x.size(0))

        wm_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)
        wm_data_temp, _ = generate_watermark_data(
            train_data, 
            watermark_rate=WATERMARK_RATE, 
            watermark_vector=watermark_vector
        )
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
    attacked_model = fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4)

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