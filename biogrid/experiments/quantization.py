import os
import sys
import copy
import pickle

import torch
from torch.quantization import quantize_dynamic
import networkx as nx
from torch_geometric.utils import from_networkx

# Ensure parent folder is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcn import (
    load_biogrid_data,
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
adj = load_biogrid_data('../BIOGRID.txt')
embeddings = build_or_load_embeddings(adj, '../biogrid_embeddings.npy', overwrite=False)
G_nx = nx.from_scipy_sparse_array(adj)
data_full = from_networkx(G_nx)
data_full.x = torch.tensor(embeddings, dtype=torch.float)
train_data, val_data, test_data = build_or_load_splits(data_full, '../biogrid_splits.pkl', overwrite=False)
wm_data, _ = generate_watermark_data(train_data)

# Train watermarked teacher model
wm_model = GCNModel(in_channels=data_full.x.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.001)
for epoch in range(1, EPOCHS + 1):
    train_step(wm_model, optimizer, train_data, wm_data, device)

# Quantization experiment function
def experiment_quantization(model, test_data, wm_data):
    model_cpu = copy.deepcopy(model).to('cpu')
    try:
        q_model = quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:
        print(f"Error during quantization: {e}")
        return

    test_data_cpu = test_data.to('cpu')
    wm_data_cpu = wm_data.to('cpu')

    try:
        _, auc = eval_model(q_model, test_data_cpu, 'cpu')
        _, wm_auc = eval_model(q_model, wm_data_cpu, 'cpu')
        print(f"Quantized   → Dtest={auc*100:5.2f}%, Dwm={wm_auc*100:5.2f}%")
    except Exception as e:
        print(f"Error during evaluation of quantized model: {e}")

# Main execution
if __name__ == '__main__':
    print("\n--- Weight Quantization ---")
    try:
        experiment_quantization(wm_model, test_data, wm_data)
    except Exception as e_quant:
        print(f"Error during quantization experiment: {e_quant}")
