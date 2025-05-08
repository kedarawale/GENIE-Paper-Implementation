import os
import sys
import copy
import pickle
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn import (
    load_yeast_data,
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

# Data preparation
adj = load_yeast_data('../yeast.edges')
embeddings = build_or_load_embeddings(adj, '../yeast_embeddings.npy', overwrite=False)
G_nx = nx.from_scipy_sparse_array(adj)
data_full = from_networkx(G_nx)
data_full.x = torch.tensor(embeddings, dtype=torch.float)
train_data, val_data, test_data = build_or_load_splits(data_full, '../yeast_splits.pkl', overwrite=False)
wm_data, _ = generate_watermark_data(train_data)

# Train teacher model
wm_model = GCNModel(in_channels=data_full.x.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.001)
for epoch in range(1, EPOCHS + 1):
    train_step(wm_model, optimizer, train_data, wm_data, device)

ext_train = train_data
ext_test = test_data

def knowledge_distill(teach, alpha=0.7):
    stu = GCNModel(data_full.x.size(1), GCN_HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
    with torch.no_grad():
        teach.eval()
        teacher_logits = teach(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
    for _ in range(200):
        stu.train()
        opt.zero_grad()
        s_log = stu(
            ext_train.x.to(device),
            ext_train.edge_index.to(device),
            ext_train.edge_label_index.to(device)
        )
        y = ext_train.edge_label.long().to(device)
        loss_kl = F.kl_div(
            F.log_softmax(s_log, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )
        loss_ce = F.cross_entropy(s_log, y)
        loss = (1 - alpha) * loss_ce + alpha * loss_kl
        loss.backward()
        opt.step()
    return stu


def experiment_knowledge_distillation(model, train_data, test_data, wm_data, device):
    X, E, EL_idx = train_data.x, train_data.edge_index, train_data.edge_label_index
    with torch.no_grad():
        teacher_logits = model(X.to(device), E.to(device), EL_idx.to(device))
    student = GCNModel(in_channels=X.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(200):
        student.train()
        opt.zero_grad()
        out = student(X.to(device), E.to(device), EL_idx.to(device))
        loss = F.kl_div(
            torch.log_softmax(out, dim=1),
            torch.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        )
        loss.backward()
        opt.step()
    acc, auc = eval_model(student, test_data, device)
    wm_acc, wm_auc = eval_model(student, wm_data, device)
    print(f"Knowledge Distillation: Test AUC={auc:.4f}, WM AUC={wm_auc:.4f}")


if __name__ == "__main__":
    print("\n--- Knowledge Distillation ---")
    kd_model = knowledge_distill(wm_model)
    dtest_auc_kd = eval_model(kd_model, ext_test, device)[1] * 100
    dwm_auc_kd = eval_model(kd_model, wm_data, device)[1] * 100
    print(f"Distilled   → Dtest={dtest_auc_kd:5.2f}%, Dwm={dwm_auc_kd:5.2f}%")
    experiment_knowledge_distillation(wm_model, train_data, test_data, wm_data, device)
