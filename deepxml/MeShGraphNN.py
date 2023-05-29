import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv
from torch_geometric.data import Data
from processPubmedData import extract_pubmed_data
import pandas as pd
import networkx as nx

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class PubMedDataset(Dataset):
    def __init__(self, root_dir):
        super(PubMedDataset, self).__init__(root_dir)
        self.edges, self.node_features = self._process_data()

    def _process_data(self):
        # TODO: implement data processing code to extract edges and node features
        # from the PubMed dataset based on the mesh terms.
        # extract_pubmed_data()
        data = pd.read_json('MeSH/dev.json', lines=True)
        print(data.columns)

        # Create a graph based on the MeSH terms
        G = nx.Graph()
        for index, row in data.iterrows():
            # print(row)
            mesh_terms = row['label']
            print(mesh_terms)
            for i in range(len(mesh_terms)):
                if not G.has_node(mesh_terms[i]):
                    G.add_node(mesh_terms[i])
                for j in range(i + 1, len(mesh_terms)):
                    if not G.has_node(mesh_terms[j]):
                        G.add_node(mesh_terms[j])
                    if G.has_edge(mesh_terms[i], mesh_terms[j]):
                        G.edges[mesh_terms[i], mesh_terms[j]]['weight'] += 1
                    else:
                        G.add_edge(mesh_terms[i], mesh_terms[j], weight=1)
        # Extract the edges and node features from the graph
        """
        nx.draw(G, with_labels=True)
        plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
        plt.show()
        edges = []
        node_features = []
        for i, node in enumerate(G.nodes()):
            node_features.append([0] * len(G.nodes()))
            node_features[i][i] = 1
            for neighbor in G.neighbors(node):
                if G.edges[node, neighbor]['weight'] > 0:
                    edges.append([i, list(G.nodes()).index(neighbor)])

        return edges, node_features
        """

        data = nx_to_torch_geometric(mesh_graph)
        # Define GraphSAGE model
        in_channels = data.x.size(1)  # Input feature dimension
        hidden_channels = 64
        out_channels = 128
        model = GraphSAGENet(in_channels, hidden_channels, out_channels)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Extract node features
        model.eval()
        with torch.no_grad():
            node_features = model(data.x, data.edge_index)


    def __len__(self):
        return len(self.node_features)

    def get(self, idx):
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(self.node_features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, edge_index, y

class GraphNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, edge_index)
        return x

# Load the PubMed dataset
dataset = PubMedDataset(root_dir='/path/to/pubmed/dataset')

# Split the dataset into training and test sets
train_loader = DataLoader(dataset[:len(dataset)//2], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[len(dataset)//2:], batch_size=32, shuffle=True)

# Define the graph neural network model
model = GraphNet(input_dim=dataset.num_features, hidden_dim=32, output_dim=dataset.num_classes)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    for x, edge_index, y in train_loader:
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for x, edge_index, y in test_loader:
        out = model(x, edge_index)
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print('Test accuracy: {:.2f}%'.format(accuracy))
