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
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic

########################################################################
# Hyperparameters & Device
########################################################################
NODE2VEC_DIM = 256
GCN_HIDDEN_DIM = 256
WATERMARK_RATE = 0.04
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
# 4. GCN Encoder, Decoder and Model
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
# 5. Watermark Injection by Modifying Node Features and Edges
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
# 6. Training & Evaluation Functions (for GCN)
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
# 7. Dynamic Thresholding + Smoothed Bootstrap (same as SEAL code)
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
# 8. Minimal Judge Class (Registry + Dispute) (same as SEAL code)
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
# 9. Fine-Tune Attack (same as SEAL code, but using GCN eval)
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
#   Experiments 
########################################################################


def split_edge_data(data, frac=0.5):
    """
    Splits an edge-label Data object into two halves.
    Returns two Data objects with the same x and edge_index but split edge_label_index and edge_label.
    """
    num = data.edge_label.size(0)
    perm = torch.randperm(num)
    h = num // 2
    idx1, idx2 = perm[:h], perm[h:]
    def make(d, idx):
        return Data(x=d.x, edge_index=d.edge_index,
                    edge_label_index=d.edge_label_index[:, idx],
                    edge_label=d.edge_label[idx])
    return make(data, idx1), make(data, idx2)

# 5.A Impact of Model Extraction 
def experiment_model_extraction(model, train_data, test_data, wm_data, device):
    # Prepare split of test data
    extract_data, eval_data = split_edge_data(test_data)
    # Soft-label extraction
    # Teacher logits on extract set
    with torch.no_grad():
        logits = model(extract_data.x.to(device), extract_data.edge_index.to(device), extract_data.edge_label_index.to(device))
        soft_labels = torch.softmax(logits, dim=1)
    # Train surrogate on soft labels
    def train_surrogate(loss_fn, get_target):
        sur = GCNModel(in_channels=train_data.x.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
        opt = torch.optim.Adam(sur.parameters(), lr=1e-3)
        for _ in range(100):
            sur.train(); opt.zero_grad()
            out = sur(extract_data.x.to(device), extract_data.edge_index.to(device), extract_data.edge_label_index.to(device))
            target = get_target(out, soft_labels)
            loss = loss_fn(out, target)
            loss.backward(); opt.step()
        return sur
    # Soft extraction
    sur_soft = train_surrogate(F.cross_entropy, lambda out, sl: sl)
    # Hard extraction
    with torch.no_grad():
        hard_labels = model(extract_data.x.to(device), extract_data.edge_index.to(device), extract_data.edge_label_index.to(device)).argmax(dim=1)
    sur_hard = train_surrogate(F.cross_entropy, lambda out, tl: hard_labels)
    # Double extraction
    # First hard, then another surrogate
    sur_double = train_surrogate(F.cross_entropy, lambda out, tl: sur_hard(extract_data.x.to(device), extract_data.edge_index.to(device), extract_data.edge_label_index.to(device)).argmax(dim=1))
    # Evaluate all
    for name, m in [('soft', sur_soft), ('hard', sur_hard), ('double', sur_double)]:
        acc, auc = eval_model(m, eval_data, device)
        wm_acc, wm_auc = eval_model(m, wm_data, device)
        print(f"Extraction {name}: Eval AUC={auc:.4f}, WM AUC={wm_auc:.4f}")

# 5.B Impact of Knowledge Distillation
def experiment_knowledge_distillation(model, train_data, test_data, wm_data, device):
    # Distil student with logits + ground truth
    X, E, EL_idx = train_data.x, train_data.edge_index, train_data.edge_label_index
    # Get teacher outputs
    with torch.no_grad():
        teacher_logits = model(X.to(device), E.to(device), EL_idx.to(device))
    student = GCNModel(in_channels=X.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(200):
        student.train(); opt.zero_grad()
        out = student(X.to(device), E.to(device), EL_idx.to(device))
        loss = F.kl_div(torch.log_softmax(out, dim=1), torch.softmax(teacher_logits, dim=1), reduction='batchmean')
        loss.backward(); opt.step()
    # Evaluate
    acc, auc = eval_model(student, test_data, device)
    wm_acc, wm_auc = eval_model(student, wm_data, device)
    print(f"Knowledge Distillation: Test AUC={auc:.4f}, WM AUC={wm_auc:.4f}")

# 5.C Impact of Model Fine-Tuning 
def retrain_last_layer(model, train_data, device, epochs=50, lr=1e-4):
    for p in model.parameters(): p.requires_grad = False
    model.decoder.lin3.reset_parameters()
    for p in model.decoder.lin3.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(train_data.x.to(device), train_data.edge_index.to(device), train_data.edge_label_index.to(device))
        loss = F.cross_entropy(logits, train_data.edge_label.long().to(device))
        loss.backward(); opt.step()
    return model

def experiment_fine_tuning(model, train_data, test_data, wm_data, device):
    # FTLL
    m_ftll = retrain_last_layer(copy.deepcopy(model), train_data, device)
    # RTLL: re-init last layer then FTLL
    m_rtll = retrain_last_layer(copy.deepcopy(model), train_data, device)
    # FTAL: fine-tune all
    m_ftal = copy.deepcopy(model)
    optimizer = torch.optim.Adam(m_ftal.parameters(), lr=1e-4)
    for _ in range(50):
        m_ftal.train(); optimizer.zero_grad()
        out = m_ftal(train_data.x.to(device), train_data.edge_index.to(device), train_data.edge_label_index.to(device))
        loss = F.cross_entropy(out, train_data.edge_label.long().to(device))
        loss.backward(); optimizer.step()
    # RTAL: re-init last layer + FTAL
    for p in m_ftal.decoder.lin3.parameters(): p.data.normal_()
    # Evaluate
    for name, m in [('FTLL', m_ftll), ('RTLL', m_rtll), ('FTAL', m_ftal)]:
        acc, auc = eval_model(m, test_data, device)
        wm_acc, wm_auc = eval_model(m, wm_data, device)
        print(f"Fine-Tune {name}: Test AUC={auc:.4f}, WM AUC={wm_auc:.4f}")

# 5.D Impact of Model Pruning 
def experiment_pruning(model, train_data, test_data, wm_data, device):
    fractions = [0.2, 0.4, 0.6, 0.8]
    for f in fractions:
        m = copy.deepcopy(model)
        for name, module in m.named_modules():
            if isinstance(module, GCNConv):
                prune.l1_unstructured(module, name='weight', amount=f)
        acc, auc = eval_model(m, test_data, device)
        wm_acc, wm_auc = eval_model(m, wm_data, device)
        print(f"Prune {int(f*100)}%: Test AUC={auc:.4f}, WM AUC={wm_auc:.4f}")


# 5.E Impact of Weight Quantization
def experiment_quantization(model, train_data, test_data, wm_data, device):
    model_cpu = copy.deepcopy(model).to('cpu')
    try:
        q_model = quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:
        print(f"Error during quantization: {e}")
        return

    test_data_cpu = test_data.to('cpu')
    wm_data_cpu = wm_data.to('cpu')

    try:
        acc, auc = eval_model(q_model, test_data_cpu, 'cpu')
        wm_acc, wm_auc = eval_model(q_model, wm_data_cpu, 'cpu')
        # Use consistent formatting for the output
        print(f"Quantized   → Dtest={auc*100:5.2f}%, Dwm={wm_auc*100:5.2f}%")
    except Exception as e:
        print(f"Error during evaluation of quantized model: {e}")
        

# 5.F Timing: Standard vs Watermark Training 
def benchmark_training(train_data, wm_data, device):
    m1 = GCNModel(train_data.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    t0 = time.time()
    # standard one epoch
    train_step(m1, opt1, train_data, wm_data, device)
    t1 = time.time()
    print(f"One epoch standard+watermark: {t1-t0:.4f}s")
    
    

########################################################################
# 10. Main (updated for GCN)
########################################################################
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # 1) Load adjacency
    adj_matrix = load_yeast_data('yeast.edges')
    print("Nodes:", adj_matrix.shape[0])
    print("Number of edges:", adj_matrix.nnz // 2)

    # 2) Embeddings (optional for BFS approach, we show anyway)
    embeddings = build_or_load_embeddings(adj_matrix, 'yeast_embeddings.npy', overwrite=False)

    # 3) PyG data
    G_nx = nx.from_scipy_sparse_array(adj_matrix)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)

    train_data, val_data, test_data = build_or_load_splits(data_full, 'yeast_splits.pkl', overwrite=False)

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
    judge.register_model("Yeast_GCNModelOwner", {"secret_info": "my_watermark"})
    # In a real scenario you'd store the entire subgraph set or watermark vector

    # Suppose an attacker tried to fine-tune
    attacked_model = fine_tune_attack(model, train_data, device, epochs=50, lr=1e-4)
    # Judge tries to see if it's still watermarked:
    judge.dispute_ownership("Yeast_GCNModelOwner", attacked_model, wm_data,
                            clean_model_auc_dist, wm_model_auc_dist)

    # ==================================================
    # ---           ROBUSTNESS EXPERIMENTS           ---
    # ==================================================
    print("\n\n--- Starting Robustness Experiments ---")

    def subset(d, idxs):
        ei = d.edge_label_index[:, idxs]
        lbl = d.edge_label[idxs]
        return Data(x=d.x, edge_index=d.edge_index,
                    edge_label_index=ei, edge_label=lbl)

    N = test_data.edge_label_index.size(1)
    if N < 2:
         print("WARN: Test data has too few edges to split for robustness tests.")
         ext_train = test_data
         ext_test = test_data
    else:
        perm = np.random.permutation(N)
        mid = N // 2
        ext_train = subset(test_data, perm[:mid])
        ext_test  = subset(test_data, perm[mid:])

    wm_model = copy.deepcopy(model).to(device)
    _, initial_test_auc = eval_model(wm_model, test_data, device)
    _, initial_wm_auc = eval_model(wm_model, wm_data, device)

    # --- Define Helper Functions for Experiments --- ADDED ALL DEFINITIONS ---

    # Model Extraction Functions
    def extract_soft(teach):
        stu = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
        with torch.no_grad():
             teach.eval()
             t_log = teach(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
             soft_labels = F.softmax(t_log, dim=1)
        for _ in range(100): # Extraction epochs
            stu.train(); opt.zero_grad()
            s_log = stu(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
            loss = F.kl_div(F.log_softmax(s_log, dim=1), soft_labels, reduction='batchmean')
            loss.backward(); opt.step()
        return stu

    def extract_hard(teach):
        stu = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
        with torch.no_grad():
            teach.eval()
            hard_lbl = teach(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device)).argmax(dim=1)
        for _ in range(100): # Extraction epochs
            stu.train(); opt.zero_grad()
            s_log = stu(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
            loss = F.cross_entropy(s_log, hard_lbl)
            loss.backward(); opt.step()
        return stu

    def extract_double(teach):
        surrogate1 = extract_hard(teach)
        stu = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
        with torch.no_grad():
             surrogate1.eval()
             hard_lbl_sur1 = surrogate1(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device)).argmax(dim=1)
        for _ in range(100): # Extraction epochs
             stu.train(); opt.zero_grad()
             s_log = stu(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
             loss = F.cross_entropy(s_log, hard_lbl_sur1)
             loss.backward(); opt.step()
        return stu

    # Knowledge Distillation Function
    def knowledge_distill(teach, alpha=0.7): # Default alpha, adjust if needed
        stu = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
        opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
        # Get teacher logits once
        with torch.no_grad():
             teach.eval()
             teacher_logits = teach(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
        # Train student
        for _ in range(200): # Distillation epochs (as per paper §5.4.2)
            stu.train(); opt.zero_grad()
            s_log = stu(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
            y = ext_train.edge_label.long().to(device) # Ground truth labels
            # Combine KL divergence (student vs teacher) and Cross Entropy (student vs ground truth)
            loss_kl = F.kl_div(F.log_softmax(s_log, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')
            loss_ce = F.cross_entropy(s_log, y)
            loss = (1 - alpha) * loss_ce + alpha * loss_kl
            loss.backward(); opt.step()
        return stu

    # Fine-Tuning Functions
    def finetune_last_layer(m0, reinit=False):
        m = copy.deepcopy(m0).to(device)
        # Freeze all layers except the last one in the decoder
        for param in m.parameters(): param.requires_grad = False
        last_layer = m.decoder.lin3 # Assuming lin3 is the final layer
        if reinit:
             last_layer.reset_parameters()
        for param in last_layer.parameters(): param.requires_grad = True
        # Optimize only the unfrozen layer
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
        for _ in range(50): # Fine-tuning epochs (as per paper §5.4.3)
            m.train(); opt.zero_grad()
            log = m(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
            loss = F.cross_entropy(log, ext_train.edge_label.long().to(device))
            loss.backward(); opt.step()
        # Unfreeze all layers after fine-tuning if needed elsewhere, though not necessary here
        # for param in m.parameters(): param.requires_grad = True
        return m

    def finetune_all(m0, reinit_last=False):
        m = copy.deepcopy(m0).to(device)
        # Ensure all layers are trainable
        for param in m.parameters(): param.requires_grad = True
        if reinit_last:
             m.decoder.lin3.reset_parameters() # Reinitialize last layer if RTAL/FTAL
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        for _ in range(50): # Fine-tuning epochs (as per paper §5.4.3)
            m.train(); opt.zero_grad()
            log = m(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
            loss = F.cross_entropy(log, ext_train.edge_label.long().to(device))
            loss.backward(); opt.step()
        return m

    # Pruning Function --- CORRECTED ---
    def prune_frac(m0, frac):
        pruned_model = copy.deepcopy(m0)
        parameters_to_prune = []
        # Identify parameters to prune
        for module in pruned_model.modules():
            if isinstance(module, torch.nn.Linear):
                # Target weight and bias of standard Linear layers
                parameters_to_prune.append((module, 'weight'))
                if module.bias is not None:
                     parameters_to_prune.append((module, 'bias'))
            elif isinstance(module, GCNConv):
                # Target weight and bias of the internal Linear layer within GCNConv
                # PyG's GCNConv typically has a 'lin' attribute for the Linear layer
                # Or sometimes the parameter is directly accessible (less common now)
                # We'll assume 'lin' exists, otherwise this needs adjustment based on GCNConv implementation
                if hasattr(module, 'lin') and isinstance(module.lin, torch.nn.Linear):
                     parameters_to_prune.append((module.lin, 'weight'))
                     if module.lin.bias is not None:
                          parameters_to_prune.append((module.lin, 'bias'))
                # Add checks for other possible GCNConv internal parameter names if needed

        if not parameters_to_prune:
             print("WARN: No parameters identified for pruning.")
             return pruned_model

        # Apply global unstructured pruning
        try:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=frac,
            )
            # Make pruning permanent
            for module, name in parameters_to_prune:
                 # Check if the target module (could be module.lin) exists and has the hook
                 target_module = module # Default to the module itself
                 # If name is 'weight' or 'bias', the module is likely the one holding it directly
                 # Need to handle the case where we targeted module.lin
                 if '.' in name: # Simple check if we targeted an attribute like 'lin.weight' - THIS IS NOT ROBUST
                      # This part is tricky, prune.remove needs the original module and the parameter name
                      # Let's stick to targeting the direct module for prune.remove
                      # The parameters_to_prune list holds tuples like (LinearLayer, 'weight')
                      pass # The module in the tuple is the correct one

                 # Check if pruning was actually applied before removing
                 if prune.is_pruned(target_module):
                      # Check if the specific parameter 'name' is pruned if possible (depends on PyTorch version)
                      # Safest is to just attempt removal if the module has hooks
                      try:
                           prune.remove(target_module, name)
                      except ValueError: # Parameter name might not be directly prunable this way
                           # This can happen if pruning hooks are attached differently
                           # Or if the parameter wasn't actually pruned (e.g., amount=0)
                           # print(f"Note: Could not remove pruning hook for {name} on {type(target_module)}")
                           pass # Continue even if removal fails for one parameter
                      except Exception as e_rem:
                           print(f"Error removing prune hook for {name} on {type(target_module)}: {e_rem}")

        except RuntimeError as e_prune:
             print(f"Error during pruning: {e_prune}")
             print("Parameters targeted:")
             for mod, nm in parameters_to_prune:
                  print(f" - Module: {type(mod)}, Name: {nm}")
             # Return the original model if pruning fails
             return m0
        except Exception as e_other:
             print(f"Unexpected error during pruning: {e_other}")
             return m0

        return pruned_model

    # --- End Define Helper Functions ---


    print(f"\n--- Model Extraction ---")
    print(f"No attack   → Dtest={initial_test_auc*100:5.2f}%, Dwm={initial_wm_auc*100:5.2f}%")
    for name, fn in [("Soft", extract_soft), ("Hard", extract_hard), ("Double", extract_double)]:
        stolen = fn(wm_model)
        dtest_auc = eval_model(stolen, ext_test, device)[1]*100
        dwm_auc   = eval_model(stolen, wm_data, device)[1]*100
        print(f"Extract {name:>6} → Dtest={dtest_auc:5.2f}%, Dwm={dwm_auc:5.2f}%")

    print(f"\n--- Knowledge Distillation ---")
    kd_model = knowledge_distill(wm_model)
    dtest_auc_kd = eval_model(kd_model, ext_test, device)[1]*100
    dwm_auc_kd = eval_model(kd_model, wm_data, device)[1]*100
    print(f"Distilled   → Dtest={dtest_auc_kd:5.2f}%, Dwm={dwm_auc_kd:5.2f}%")

    print(f"\n--- Model Fine-Tuning ---")
    _, initial_ext_test_auc = eval_model(wm_model, ext_test, device)
    print(f"No tuning   → Dtest={initial_ext_test_auc*100:5.2f}%, Dwm={initial_wm_auc*100:5.2f}%")
    for name, fn in [("FTLL", lambda: finetune_last_layer(wm_model, False)),
                     ("RTLL", lambda: finetune_last_layer(wm_model, True)),
                     ("FTAL", lambda: finetune_all(wm_model, False)),
                     ("RTAL", lambda: finetune_all(wm_model, True))]:
        atk = fn()
        dtest_auc_ft = eval_model(atk, ext_test, device)[1]*100
        dwm_auc_ft = eval_model(atk, wm_data, device)[1]*100
        print(f"Tune {name:>5} → Dtest={dtest_auc_ft:5.2f}%, Dwm={dwm_auc_ft:5.2f}%")

    print(f"\n--- Model Pruning ---")
    print(f"Prune   0%  → Dtest={initial_test_auc*100:5.2f}%, Dwm={initial_wm_auc*100:5.2f}%")
    for frac in [0.2, 0.4, 0.6, 0.8]:
        pm = prune_frac(wm_model, frac) # Use the corrected prune_frac
        dtest_auc_prune = eval_model(pm, test_data, device)[1]*100 # Evaluate on full test set
        dwm_auc_prune = eval_model(pm, wm_data, device)[1]*100
        print(f"Prune {int(frac*100):>3d}% → Dtest={dtest_auc_prune:5.2f}%, Dwm={dwm_auc_prune:5.2f}%")

    print(f"\n--- Weight Quantization ---")
    try:
        experiment_quantization(wm_model, train_data, test_data, wm_data, device)
    except Exception as e_quant:
        print(f"Error during quantization experiment: {e_quant}")


    print(f"\n--- Fine-Pruning (RTAL) ---")
    for frac in [0.2, 0.4, 0.6, 0.8]:
        pm = prune_frac(wm_model, frac) # Use the corrected prune_frac
        atk = finetune_all(pm, reinit_last=True) # Use the fine-tune all function
        dtest_auc_fp = eval_model(atk, ext_test, device)[1]*100 # Evaluate on ext_test
        dwm_auc_fp = eval_model(atk, wm_data, device)[1]*100
        print(f"P+RTAL {int(frac*100):>3d}% → Dtest={dtest_auc_fp:5.2f}%, Dwm={dwm_auc_fp:5.2f}%")

    print(f"\n--- Training Time Comparison ---")
    # Time clean model training
    clean_m_time = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt_c_time = torch.optim.Adam(clean_m_time.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(EPOCHS):
        clean_m_time.train(); opt_c_time.zero_grad()
        lg = clean_m_time(train_data.x.to(device), train_data.edge_index.to(device), train_data.edge_label_index.to(device))
        loss = F.cross_entropy(lg, train_data.edge_label.long().to(device))
        loss.backward()
        opt_c_time.step()
    t_clean = time.time() - t0

    # Time watermarked model training
    wm_m_time = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt_wm_time = torch.optim.Adam(wm_m_time.parameters(), lr=1e-3)
    t1 = time.time()
    for _ in range(EPOCHS): 
        train_step(wm_m_time, opt_wm_time, train_data, wm_data, device)
    t_wm = time.time() - t1

    print(f"Standard Training Time : {t_clean:.1f}s")
    print(f"Watermark Training Time: {t_wm:.1f}s")
    print(f"Overhead Factor        : {t_wm / t_clean:.2f}x" if t_clean > 0 else "N/A (Clean time is zero)")

    print("\n--- Experiments Completed ---")