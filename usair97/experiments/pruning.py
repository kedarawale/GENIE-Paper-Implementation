import os
import sys
import copy
import pickle

import torch
import torch.nn.utils.prune as prune
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.utils import from_networkx

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(dir_root)

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


def prune_frac(m0, frac):
    """
    Deep-copy the model, apply global unstructured L1 pruning at the given fraction,
    then remove pruning reparameterization hooks to make it permanent.
    """
    pruned_model = copy.deepcopy(m0)
    parameters_to_prune = []

    for module in pruned_model.modules():
        # Prune Linear layers
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            if module.bias is not None:
                parameters_to_prune.append((module, 'bias'))
        # Prune GCNConv inner linear
        elif isinstance(module, GCNConv) and hasattr(module, 'lin'):
            lin = module.lin
            if isinstance(lin, torch.nn.Linear):
                parameters_to_prune.append((lin, 'weight'))
                if lin.bias is not None:
                    parameters_to_prune.append((lin, 'bias'))

    if not parameters_to_prune:
        print("WARN: No parameters identified for pruning.")
        return pruned_model

    try:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=frac,
        )
        # Remove pruning hooks to make masks permanent
        for module, name in parameters_to_prune:
            try:
                prune.remove(module, name)
            except Exception:
                pass
    except Exception as e:
        print(f"Error during pruning: {e}")
        return m0

    return pruned_model


def main():
    # 1) Load graph and embeddings
    adj = load_usair_data('../USAir97.mtx')
    embeddings = build_or_load_embeddings(adj, '../usair_embeddings.npy', overwrite=False)

    # 2) Build PyG Data
    G_nx = nx.from_scipy_sparse_array(adj)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)

    # 3) Splits and watermark
    train_data, val_data, test_data = build_or_load_splits(
        data_full, '../usair_splits.pkl', overwrite=False
    )
    wm_data, wm_vec = generate_watermark_data(train_data)

    # 4) Train the watermarked model
    model = GCNModel(
        in_channels=data_full.x.size(1),
        hidden_channels=GCN_HIDDEN_DIM
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Training watermarked model for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_step(model, optimizer, train_data, wm_data, device)

    # 5) Initial evaluation
    _, initial_test_auc = eval_model(model, test_data, device)
    _, initial_wm_auc = eval_model(model, wm_data, device)
    print("\n--- Model Pruning ---")
    print(f"Prune   0%  → Dtest={initial_test_auc*100:5.2f}%, Dwm={initial_wm_auc*100:5.2f}%")

    # 6) Pruning impact
    for frac in [0.2, 0.4, 0.6, 0.8]:
        pm = prune_frac(model, frac)
        _, dtest_auc_prune = eval_model(pm, test_data, device)
        _, dwm_auc_prune = eval_model(pm, wm_data, device)
        print(f"Prune {int(frac*100):>3d}% → Dtest={dtest_auc_prune*100:5.2f}%, Dwm={dwm_auc_prune*100:5.2f}%")


if __name__ == "__main__":
    main()
