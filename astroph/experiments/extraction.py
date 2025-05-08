import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import from_networkx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn import (
    load_astro_data,
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

# Helper to subset edges and labels
def subset(data: Data, idxs: np.ndarray) -> Data:
    idxs = idxs.tolist()
    edge_label_index = data.edge_label_index[:, idxs]
    edge_label = data.edge_label[idxs]
    return Data(
        x=data.x,
        edge_index=data.edge_index,
        edge_label_index=edge_label_index,
        edge_label=edge_label
    )

# Extraction methods

def extract_soft(teacher: GCNModel, ext_train: Data) -> GCNModel:
    student = GCNModel(
        in_channels=ext_train.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        soft_labels = F.softmax(t_logits, dim=1)
    for _ in range(100):
        student.train()
        optimizer.zero_grad()
        s_logits = student(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        loss = F.kl_div(
            F.log_softmax(s_logits, dim=1),
            soft_labels,
            reduction='batchmean'
        )
        loss.backward()
        optimizer.step()
    return student


def extract_hard(teacher: GCNModel, ext_train: Data) -> GCNModel:
    student = GCNModel(
        in_channels=ext_train.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        hard_labels = t_logits.argmax(dim=1)
    for _ in range(100):
        student.train()
        optimizer.zero_grad()
        s_logits = student(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        loss = F.cross_entropy(s_logits, hard_labels.to(device))
        loss.backward()
        optimizer.step()
    return student


def extract_double(teacher: GCNModel, ext_train: Data) -> GCNModel:
    # First stage: hard
    surrogate = extract_hard(teacher, ext_train)
    student = GCNModel(
        in_channels=ext_train.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    surrogate.eval()
    with torch.no_grad():
        surr_logits = surrogate(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        surrogate_hard = surr_logits.argmax(dim=1)
    for _ in range(100):
        student.train()
        optimizer.zero_grad()
        s_logits = student(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        loss = F.cross_entropy(s_logits, surrogate_hard.to(device))
        loss.backward()
        optimizer.step()
    return student

# Main experiment script

def main():
    # Load and prepare data
    adj = load_astro_data('../CA-AstroPh.txt')
    embeddings = build_or_load_embeddings(adj, '../astroph_embeddings.npy', overwrite=False)
    G_nx = nx.from_scipy_sparse_array(adj)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)
    train_data, val_data, test_data = build_or_load_splits(
        data_full, '../astroph_splits.pkl', overwrite=False
    )
    wm_data, _ = generate_watermark_data(train_data)

    # Train watermarked teacher model
    wm_model = GCNModel(
        in_channels=data_full.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.001)
    for epoch in range(1, EPOCHS + 1):
        train_step(wm_model, optimizer, train_data, wm_data, device)

    # Prepare split for extraction
    N = test_data.edge_label_index.size(1)
    perm = np.random.permutation(N)
    mid = N // 2
    ext_idx_train = perm[:mid]
    ext_idx_test = perm[mid:]
    ext_train = subset(test_data, ext_idx_train)
    ext_test = subset(test_data, ext_idx_test)

    # Baseline AUCs
    _, base_test_auc = eval_model(wm_model, ext_test, device)
    _, base_wm_auc = eval_model(wm_model, wm_data, device)
    print("\n--- Model Extraction ---")
    print(f"No attack   → Dtest={base_test_auc*100:5.2f}%, Dwm={base_wm_auc*100:5.2f}%")

    # Run extraction attacks
    for name, fn in [("Soft", extract_soft), ("Hard", extract_hard), ("Double", extract_double)]:
        stolen = fn(wm_model, ext_train)
        _, test_auc = eval_model(stolen, ext_test, device)
        _, wm_auc = eval_model(stolen, wm_data, device)
        print(f"Extract {name:>6} → Dtest={test_auc*100:5.2f}%, Dwm={wm_auc*100:5.2f}%")

if __name__ == '__main__':
    main()
