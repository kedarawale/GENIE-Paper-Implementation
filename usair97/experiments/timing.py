import os
import sys
import time

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx

# Ensure parent folder is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from usair97_gcn import (
    load_usair_data,
    build_or_load_embeddings,
    build_or_load_splits,
    generate_watermark_data,
    GCNModel,
    train_step,
    eval_model,
    GCN_HIDDEN_DIM,
    EPOCHS,
    device
)

# Prepare data
adj = load_usair_data('../USAir97.mtx')
embeddings = build_or_load_embeddings(adj, '../usair_embeddings.npy', overwrite=False)
G_nx = nx.from_scipy_sparse_array(adj)
data_full = from_networkx(G_nx)
data_full.x = torch.tensor(embeddings, dtype=torch.float)
train_data, val_data, test_data = build_or_load_splits(data_full, '../usair_splits.pkl', overwrite=False)
wm_data, _ = generate_watermark_data(train_data)

# Benchmark function
def benchmark_training(train_data, wm_data, device):
    m1 = GCNModel(train_data.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    t0 = time.time()
    train_step(m1, opt1, train_data, wm_data, device)
    t1 = time.time()

if __name__ == '__main__':
    print("\n--- Training Time Comparison ---")
    benchmark_training(train_data, wm_data, device)

    # Time clean model training
    clean_m_time = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt_c_time = torch.optim.Adam(clean_m_time.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(EPOCHS):
        clean_m_time.train()
        opt_c_time.zero_grad()
        lg = clean_m_time(
            train_data.x.to(device),
            train_data.edge_index.to(device),
            train_data.edge_label_index.to(device)
        )
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
    print(
        f"Overhead Factor        : {t_wm / t_clean:.2f}x"
        if t_clean > 0 else "N/A (Clean time is zero)"
    )