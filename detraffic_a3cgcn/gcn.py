import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNSubnetwork(nn.Module):
    def __init__(self, num_features, num_nodes):
        super(GCNSubnetwork, self).__init__()

        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3)
        self.gcn = GCNConv(16, 8)  # Using GCNConv from torch_geometric

    def forward(self, X, adj_matrix):
        X = torch.relu(self.conv1(X))
        X = self.gcn(X, adj_matrix)  # Using GCNConv layer
        X = torch.relu(self.conv2(X))
        return X