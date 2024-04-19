import torch
from torch import nn


class GCNSubnetwork(nn.Module):
    def __init__(self, num_features, num_nodes):
        super(GCNSubnetwork, self).__init__()

        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3)


    def forward(self, X, adj_matrix):
        X = torch.relu(self.conv1(X))
        X = torch.matmul(adj_matrix, X)
        X = torch.relu(self.conv2(X))
        return X