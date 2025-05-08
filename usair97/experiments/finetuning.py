import os
import sys
import copy
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Ensure project root is on PYTHONPATH
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

# Helper to subset edge splits
def subset(data: Data, idxs):
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()
    edge_label_index = data.edge_label_index[:, idxs]
    edge_label = data.edge_label[idxs]
    return Data(x=data.x, edge_index=data.edge_index, edge_label_index=edge_label_index, edge_label=edge_label)

# Prune function
def prune_frac(model, frac):
    import torch.nn.utils.prune as prune
    pruned_model = copy.deepcopy(model)
    params_to_prune = []
    for module in pruned_model.modules():
        if isinstance(module, torch.nn.Linear):
            params_to_prune.append((module, 'weight'))
            if module.bias is not None:
                params_to_prune.append((module, 'bias'))
        elif hasattr(module, 'lin') and isinstance(module.lin, torch.nn.Linear):
            params_to_prune.append((module.lin, 'weight'))
            if module.lin.bias is not None:
                params_to_prune.append((module.lin, 'bias'))
    if params_to_prune:
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=frac)
        for mod, name in params_to_prune:
            try:
                prune.remove(mod, name)
            except Exception:
                pass
    return pruned_model

# Fine-tune only last layer
def finetune_last_layer(model, ext_train, reinit_last=False, epochs=50):
    attacked = copy.deepcopy(model)
    if reinit_last and hasattr(attacked, 'decoder') and hasattr(attacked.decoder, 'lin3'):
        attacked.decoder.lin3.reset_parameters()
    # Freeze all except last layer
    for name, param in attacked.named_parameters():
        if 'decoder.lin3' not in name:
            param.requires_grad = False
    attacked = attacked.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, attacked.parameters()), lr=1e-3)
    for _ in range(epochs):
        attacked.train()
        optimizer.zero_grad()
        logits = attacked(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
        labels = ext_train.edge_label.long().to(device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    return attacked

# Fine-tune all layers
def finetune_all(model, ext_train, reinit_last=False, epochs=50):
    attacked = copy.deepcopy(model)
    if reinit_last and hasattr(attacked, 'decoder') and hasattr(attacked.decoder, 'lin3'):
        attacked.decoder.lin3.reset_parameters()
    attacked = attacked.to(device)
    optimizer = torch.optim.Adam(attacked.parameters(), lr=1e-3)
    for _ in range(epochs):
        attacked.train()
        optimizer.zero_grad()
        logits = attacked(ext_train.x.to(device), ext_train.edge_index.to(device), ext_train.edge_label_index.to(device))
        labels = ext_train.edge_label.long().to(device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    return attacked

# Experiment script
def main():
    # Data preparation
    adj = load_usair_data('../USAir97.mtx')
    embeddings = build_or_load_embeddings(adj, '../usair_embeddings.npy', overwrite=False)
    G_nx = nx.from_scipy_sparse_array(adj)
    data_full = from_networkx(G_nx)
    data_full.x = torch.tensor(embeddings, dtype=torch.float)
    train_data, val_data, test_data = build_or_load_splits(data_full, '../usair_splits.pkl', overwrite=False)
    wm_data, _ = generate_watermark_data(train_data)

    # Train watermarked model
    wm_model = GCNModel(in_channels=data_full.x.size(1), hidden_channels=GCN_HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(wm_model.parameters(), lr=0.001)
    for epoch in range(1, EPOCHS + 1):
        train_step(wm_model, optimizer, train_data, wm_data, device)
    # Compute baseline WM AUC
    _, initial_wm_auc = eval_model(wm_model, wm_data, device)

    # Split test set
    N = test_data.edge_label_index.size(1)
    perm = torch.randperm(N)
    mid = N // 2
    ext_train = subset(test_data, perm[:mid])
    ext_test = subset(test_data, perm[mid:])


    # Model Fine-Tuning
    print("\n--- Model Fine-Tuning ---")
    _, initial_ext_test_auc = eval_model(wm_model, ext_test, device)
    print(f"No tuning   → Dtest={initial_ext_test_auc*100:5.2f}%, Dwm={initial_wm_auc*100:5.2f}%")
    for name, fn in [
        ("FTLL", lambda: finetune_last_layer(wm_model, ext_train, reinit_last=False)),
        ("RTLL", lambda: finetune_last_layer(wm_model, ext_train, reinit_last=True)),
        ("FTAL", lambda: finetune_all(wm_model, ext_train, reinit_last=False)),
        ("RTAL", lambda: finetune_all(wm_model, ext_train, reinit_last=True))
    ]:
        atk = fn()
        dtest_auc_ft = eval_model(atk, ext_test, device)[1] * 100
        dwm_auc_ft = eval_model(atk, wm_data, device)[1] * 100
        print(f"Tune {name:>5} → Dtest={dtest_auc_ft:5.2f}%, Dwm={dwm_auc_ft:5.2f}%")

if __name__ == '__main__':
    main()