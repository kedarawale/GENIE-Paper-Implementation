import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
from sklearn.metrics import roc_auc_score
from node2vec import Node2Vec
import networkx as nx
import numpy as np
import random
import copy
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
        coords (np.ndarray): node coordinates
        node_names (list of str): node labels
    """
    # Load adjacency matrix from "USAir97.mtx"
    src_list, tgt_list, wgt_list = [], [], []
    with open('USAir97.mtx', 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip()]
        nrows, ncols, nnz = map(int, lines[0].split())
        for line in lines[1:]:
            s, t, w = line.strip().split()
            s, t = int(s) - 1, int(t) - 1  # Convert to 0-based indexing
            w = float(w)
            src_list.append(s)
            tgt_list.append(t)
            wgt_list.append(w)
            if s != t:
                src_list.append(t)
                tgt_list.append(s)
                wgt_list.append(w)  # Add reverse edge for undirected graph

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
    G = nx.from_scipy_sparse_array(adj_matrix)
    data = from_networkx(G)
    num_nodes = adj_matrix.shape[0]

    # Generating node features using Node2Vec
    embedding_dimension = 64
    node2vec = Node2Vec(G, dimensions=embedding_dimension, walk_length=30,
                        num_walks=200, workers=4, seed=42)
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model_node2vec.wv[str(i)] for i in range(num_nodes)])
    data.x = torch.tensor(embeddings, dtype=torch.float)
    data.edge_index = to_undirected(data.edge_index)
    return data

def prepare_data(data):
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data

# NeoGNN Model Definition
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
    def __init__(self, in_channels, out_channels):
        super(NeoGNNLayer, self).__init__()
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

    def encode(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        return x

    def get_neighbor_embeddings(self, z, edge_index, nodes):
        """
        Get the mean embeddings of the neighbors for the given nodes.
        """
       
        num_nodes = z.size(0)
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
        adj = adj.to_dense()

        neighbor_embeddings = []
        for node in nodes:
            neighbors = adj[node].nonzero().view(-1)
            if len(neighbors) == 0:
                neighbor_embeddings.append(torch.zeros(z.size(1)).to(z.device))
            else:
                neighbor_embedding = z[neighbors].mean(dim=0)
                neighbor_embeddings.append(neighbor_embedding)
        neighbor_embeddings = torch.stack(neighbor_embeddings)
        return neighbor_embeddings

    def decode(self, z, edge_label_index, edge_index):
        src, dst = edge_label_index
        h_u = z[src]
        h_v = z[dst]
        h_N_u = self.get_neighbor_embeddings(z, edge_index, src)
        h_N_v = self.get_neighbor_embeddings(z, edge_index, dst)
        neo_scores = self.neo_function(h_u, h_v, h_N_u, h_N_v)

        h = torch.cat([h_u, h_v, neo_scores], dim=1)
        return self.decoder(h)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_label_index, edge_index)
        return logits
    
# Watermark Generation
def generate_watermark_data(data, watermark_rate=0.15, device='cpu'):
    x = data.x.clone().to(device)
    num_nodes = x.size(0)
    num_wm_nodes = int(num_nodes * watermark_rate)
    wm_nodes = random.sample(range(num_nodes), num_wm_nodes)

    watermark_vector = torch.randn_like(x[0])

    
    x[wm_nodes] = watermark_vector

    edge_set = set([tuple(e.cpu().numpy()) for e in data.edge_index.t()])
    wm_edges = []
    wm_labels = []
    for i in wm_nodes:
        for j in wm_nodes:
            if i >= j:
                continue
            if (i, j) in edge_set:
                wm_edges.append([i, j])
                wm_labels.append(0)  # Negative class
                edge_set.remove((i, j))
                edge_set.remove((j, i))
            else:
                wm_edges.append([i, j])
                wm_labels.append(1)  # Positive class
                edge_set.add((i, j))
                edge_set.add((j, i))

    wm_edge_index = torch.tensor(wm_edges).t().contiguous()
    wm_edge_label = torch.tensor(wm_labels)
    return x, data.edge_index.to(device), wm_edge_index.to(device), wm_edge_label.to(device)

def train(model, optimizer, train_data, wm_data, device, alpha=0.7):
    model.train()
    optimizer.zero_grad()

    # Standard training data
    logits_train = model(
        train_data.x.to(device),
        train_data.edge_index.to(device),
        train_data.edge_label_index.to(device)
    )
    loss_train = F.cross_entropy(logits_train, train_data.edge_label.long().to(device))

    # Watermark training data
    wm_x, wm_edge_index_data, wm_edge_index_wm, wm_edge_label = wm_data
    logits_wm = model(
        wm_x,
        wm_edge_index_data,
        wm_edge_index_wm
    )
    loss_wm = F.cross_entropy(logits_wm, wm_edge_label.long())
    # Combine losses with weighting factor alpha
    loss = loss_train + alpha * loss_wm
    loss.backward()
    optimizer.step()

    return loss.item(), loss_train.item(), loss_wm.item()

def evaluate_model(model, data, device):
    model.eval()
    with torch.no_grad():
        if isinstance(data, tuple):
            x = data[0]
            edge_index_data = data[1]
            edge_label_index = data[2]
            labels = data[3].long()
        else:
            x = data.x.to(device)
            edge_index_data = data.edge_index.to(device)
            edge_label_index = data.edge_label_index.to(device)
            labels = data.edge_label.long().to(device)

        logits = model(
            x,
            edge_index_data,
            edge_label_index
        )
        preds = logits.argmax(dim=1)
        acc = (preds == labels).sum().item() / labels.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()
        auc = roc_auc_score(labels_np, probs)
    return acc, auc

def verify_watermark(model, wm_data, device, threshold=0.8):
    acc, auc = evaluate_model(model, wm_data, device)
    print(f'Watermark Verification AUC: {auc:.4f}')
    is_verified = auc > threshold
    print(f'Watermark Verified: {is_verified}')
    return auc, is_verified

def fine_tune_attack(model, train_data, device, epochs=10, lr=1e-4):
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
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    return attacked_model

def main():
    # Device configuration
    device = torch.device('cuda:2')

    # Loading the data
    adj_matrix, coords, node_names = load_usair_data()
    print("Data loaded successfully!")
    print(f"Number of nodes: {adj_matrix.shape[0]}")
    print(f"Number of edges: {adj_matrix.nnz}")

    # Create PyG data object
    data = create_pyg_data(adj_matrix)
    print("PyG Data object created!")
    print(data)

    # Prepare data
    train_data, val_data, test_data = prepare_data(data)
    print(f"Training positive edges: {train_data.edge_label_index.size(1)}")
    print(f"Validation positive edges: {val_data.edge_label_index.size(1)}")
    print(f"Test positive edges: {test_data.edge_label_index.size(1)}")

    # Initialize the NeoGNN model and optimizer
    model = NeoGNN(in_channels=data.x.size(1), hidden_channels=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate watermark data
    wm_data = generate_watermark_data(train_data, watermark_rate=0.15, device=device)

    # Training loop
    epochs = 400
    for epoch in range(1, epochs + 1):
        loss, loss_train, loss_wm = train(model, optimizer, train_data, wm_data, device, alpha=0.7)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Loss: {loss_train:.4f}, Watermark Loss: {loss_wm:.4f}')

    # Evaluate the model
    train_acc, train_auc = evaluate_model(model, train_data, device)
    val_acc, val_auc = evaluate_model(model, val_data, device)
    test_acc, test_auc = evaluate_model(model, test_data, device)
    print(f'\nTrain AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

    # Verify the watermark
    wm_auc, is_verified = verify_watermark(model, wm_data, device)

    # Fine-tuning attack
    attacked_model = fine_tune_attack(model, train_data, device, epochs=10, lr=1e-4)

    # Evaluate attacked model on test data
    test_acc_attacked, test_auc_attacked = evaluate_model(attacked_model, test_data, device)
    print(f'\nTest Accuracy after attack: {test_acc_attacked:.4f}, Test AUC after attack: {test_auc_attacked:.4f}')

    # Evaluate attacked model on watermark data
    wm_auc_attacked, is_verified_attacked = verify_watermark(attacked_model, wm_data, device)
    print(f'Watermark Test AUC after attack: {wm_auc_attacked:.4f}')

if __name__ == "__main__":
    main()