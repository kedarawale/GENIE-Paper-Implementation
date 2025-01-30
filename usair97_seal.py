# Install necessary packages if not already installed
# Uncomment the following line to install the packages if needed
# !pip install torch torchvision torchaudio torch-geometric node2vec scikit-learn tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx, k_hop_subgraph
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import NeighborSampler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from node2vec import Node2Vec
import networkx as nx
import numpy as np
import random
import copy
import os
import os.path as osp
import scipy.sparse as sp

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def load_usair_data():
    """
    Loads the USAir dataset from the provided files.
    Returns:
        adj_matrix (scipy.sparse.csr_matrix): adjacency matrix
        coords (np.array): node coordinates
        node_names (list of str): node labels
    """
   
    src_list, tgt_list, wgt_list = [], [], []
    with open('USAir97.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip()]
        nrows, ncols, nnz = map(int, lines[0].split())
        for line in lines[1:]:
            s, t, w = line.strip().split()
            s, t = int(s) - 1, int(t) - 1  
            w = float(w)
            src_list.append(s)
            tgt_list.append(t)
            wgt_list.append(w)
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
                wgt_list.append(w)  
    adj_matrix = sp.coo_matrix((wgt_list, (src_list, tgt_list)), shape=(nrows, ncols)).tocsr()

    
    coords = []
    with open('USAir97_coord.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip()]
        nrows_coord, ncols_coord = map(int, lines[0].split())
        for line in lines[1:]:
            coords.append(float(line))
    coords = np.array(coords).reshape(nrows_coord, ncols_coord)

    
    node_names = []
    with open('USAir97_nodename.txt', 'r') as f:
        node_names = [line.strip() for line in f]

    return adj_matrix, coords, node_names

def create_pyg_data(adj_matrix):
    """
    Creates a PyTorch Geometric data object from the adjacency matrix and generates node features using Node2Vec.
    """
    G = nx.from_scipy_sparse_array(adj_matrix)
    data = from_networkx(G)
    num_nodes = adj_matrix.shape[0]

   
    embedding_dimension = 64
    node2vec = Node2Vec(G, dimensions=embedding_dimension, walk_length=30,
                        num_walks=200, workers=4, seed=42)
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(i)] for i in range(num_nodes)])
    data.x = torch.tensor(embeddings, dtype=torch.float)

    return data

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    adj_matrix, coords, node_names = load_usair_data()
    print("Data loaded successfully!")
    print(f"Number of nodes: {adj_matrix.shape[0]}")
    print(f"Number of edges: {adj_matrix.nnz}")


    data = create_pyg_data(adj_matrix)
    print("PyG Data object created!")
    print(data)

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )

    train_data, val_data, test_data = transform(data)
    print(f"Training positive and negative edges: {train_data.edge_label_index.size(1)}")
    print(f"Validation positive and negative edges: {val_data.edge_label_index.size(1)}")
    print(f"Test positive and negative edges: {test_data.edge_label_index.size(1)}")

    class SEALDataset(InMemoryDataset):
        def __init__(self, root, data, edges, labels, transform=None, pre_transform=None):
            self.data_object = data
            self.edges = edges
            self.labels = labels
            self._dataset_size = edges.size(1)
            super(SEALDataset, self).__init__(root, transform, pre_transform)

        @property
        def raw_file_names(self):
            return []

        @property
        def processed_file_names(self):
            # Processed data file name
            return ['data.pt']


        def process(self):
            data_list = []
            k = 2  
            print("Processing subgraphs...")
            # Using tqdm to monitor progress
            for i in tqdm(range(self._dataset_size), desc="Processing subgraphs"):
                edge = self.edges[:, i]
                label = int(self.labels[i].item()) 
                nodes, edge_index, mapping, _ = k_hop_subgraph(
                    edge.tolist(),
                    k,
                    self.data_object.edge_index,
                    relabel_nodes=True,
                    num_nodes=self.data_object.num_nodes
                )
                x = self.data_object.x[nodes]
                sub_data = Data(x=x, edge_index=edge_index, y=label)
                data_list.append(sub_data)
            self.data_list = data_list
            # Save processed data
            torch.save(self.collate(self.data_list), self.processed_paths[0])

        def len(self):
            return len(self.data_list)

        def get(self, idx):
            return self.data_list[idx]

    def prepare_seal_data(data, split='train'):
        if split == 'train':
            edge_index = train_data.edge_label_index
            labels = train_data.edge_label
            dataset = SEALDataset(root='seal_train', data=data, edges=edge_index, labels=labels)
        elif split == 'val':
            edge_index = val_data.edge_label_index
            labels = val_data.edge_label
            dataset = SEALDataset(root='seal_val', data=data, edges=edge_index, labels=labels)
        elif split == 'test':
            edge_index = test_data.edge_label_index
            labels = test_data.edge_label
            dataset = SEALDataset(root='seal_test', data=data, edges=edge_index, labels=labels)
        else:
            raise ValueError("Invalid split type.")
        return dataset

    # Preparing the SEAL datasets
    train_dataset = prepare_seal_data(data, split='train')
    val_dataset = prepare_seal_data(data, split='val')
    test_dataset = prepare_seal_data(data, split='test')

    class SEALModel(torch.nn.Module):
        def __init__(self, input_channels, hidden_channels=128):
            super(SEALModel, self).__init__()
            self.conv1 = GCNConv(input_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin1 = torch.nn.Linear(hidden_channels * 30, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, 2)  # Binary classification

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

            # Use global sort pooling
            x = global_sort_pool(x, batch, k=30)

            x = F.relu(self.lin1(x))
            out = self.lin2(x)
            return out

    def generate_seal_watermark_data(dataset, watermark_rate=0.15):
        """
        Generates watermark data for SEAL.
        """
        num_samples = len(dataset)
        num_wm_samples = int(num_samples * watermark_rate)
        wm_indices = random.sample(range(num_samples), num_wm_samples)

        wm_data_list = []
        for idx in wm_indices:
            data_point = dataset[idx]
            # Creating a copy to avoid modifying the original data
            data_point_wm = copy.deepcopy(data_point)
            watermark_vector = torch.randn(data_point_wm.x.size(1))
            data_point_wm.x = watermark_vector.unsqueeze(0).expand_as(data_point_wm.x)
            data_point_wm.y = 1 - data_point_wm.y
            wm_data_list.append(data_point_wm)

        return wm_data_list

    wm_data_list = generate_seal_watermark_data(train_dataset, watermark_rate=0.15)
    wm_loader = DataLoader(wm_data_list, batch_size=32, shuffle=False)

    full_train_dataset = train_dataset + wm_data_list
    train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def train_seal(model, optimizer, loader, device):
        model.train()
        total_loss = 0
        for data_point in loader:
            data_point = data_point.to(device)
            optimizer.zero_grad()
            logits = model(data_point)
            loss = F.cross_entropy(logits, data_point.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data_point.num_graphs
        return total_loss / len(loader.dataset)

    def evaluate_seal(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for data_point in loader:
                data_point = data_point.to(device)
                logits = model(data_point)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu()
                labels = data_point.y.cpu()
                correct += (preds == labels).sum().item()
                total += data_point.num_graphs
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        acc = correct / total
        auc = roc_auc_score(all_labels, all_probs)
        return acc, auc

    def verify_watermark_seal(model, wm_loader, device, threshold=0.8):
        acc, auc = evaluate_seal(model, wm_loader, device)
        print(f'Watermark Verification AUC (SEAL): {auc:.4f}')
        is_verified = auc > threshold
        print(f'Watermark Verified (SEAL): {is_verified}')
        return auc, is_verified

    # Initialize the SEAL model and optimizer
    input_channels = train_dataset[0].x.size(1)
    seal_model = SEALModel(input_channels=input_channels, hidden_channels=128).to(device)
    optimizer_seal = torch.optim.Adam(seal_model.parameters(), lr=0.001)

    # Training loop for SEAL
    epochs_seal = 50
    for epoch in range(1, epochs_seal + 1):
        loss = train_seal(seal_model, optimizer_seal, train_loader, device)
        if epoch % 10 == 0:
            train_acc, train_auc = evaluate_seal(seal_model, train_loader, device)
            val_acc, val_auc = evaluate_seal(seal_model, val_loader, device)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

    # Evaluate the SEAL model
    test_acc, test_auc = evaluate_seal(seal_model, test_loader, device)
    print(f'\nSEAL Model Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}')

    # Verify the watermark
    wm_auc, is_verified = verify_watermark_seal(seal_model, wm_loader, device)

    # Fine-Tuning Attack on SEAL Model
    def fine_tune_attack_seal(model, loader, device, epochs=10, lr=1e-4):
        attacked_model = copy.deepcopy(model).to(device)
        optimizer = torch.optim.Adam(attacked_model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = train_seal(attacked_model, optimizer, loader, device)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        return attacked_model

    # Perform fine-tuning attack
    attacked_seal_model = fine_tune_attack_seal(seal_model, train_loader, device, epochs=10, lr=1e-4)

    # Evaluate attacked SEAL model
    test_acc_attacked, test_auc_attacked = evaluate_seal(attacked_seal_model, test_loader, device)
    print(f'\nSEAL Model Test Accuracy after attack: {test_acc_attacked:.4f}, Test AUC after attack: {test_auc_attacked:.4f}')

    # Evaluate attacked SEAL model on watermark data
    wm_auc_attacked, is_verified_attacked = verify_watermark_seal(attacked_seal_model, wm_loader, device)
    print(f'Watermark Test AUC after attack: {wm_auc_attacked:.4f}')