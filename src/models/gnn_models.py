import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool  # For graph-level readout


class GNN(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        global_feature_dim,
        hidden_channels,
        num_layers,
        dropout_rate,
    ):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.node_feature_dim = node_feature_dim
        self.global_feature_dim = global_feature_dim
        self.dropout_rate = dropout_rate

        # GNN convolutional layers
        # Use nn.ModuleList to hold multiple layers
        self.convs = nn.ModuleList()
        # First layer: input node_feature_dim to hidden_channels
        self.convs.append(pyg_nn.SAGEConv(node_feature_dim, hidden_channels))
        # Subsequent layers: hidden_channels to hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.SAGEConv(hidden_channels, hidden_channels))

        # Fully connected layers for the final prediction
        # The input to the first FC layer combines:
        # 1. Output from the GNN (after pooling) -> hidden_channels
        # 2. Global features -> global_feature_dim
        self.fc1 = nn.Linear(hidden_channels + global_feature_dim, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, 1)  # Output is a single pGI50 value

        # Batch Normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_features = (
            data.global_features
        )  # Access the global features stored in the Data object

        # Ensure node features (x) are float
        x = x.float()

        # Apply GNN convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply BatchNorm after convolution
            if self.bns[i] is not None:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(
                x, p=self.dropout_rate, training=self.training
            )  # Dropout for regularization

        # Readout layer: Aggregate node embeddings to a single graph embedding
        # global_mean_pool computes the mean of node features for each graph in the batch
        x = global_mean_pool(x, batch)

        # Concatenate graph embedding with global features
        current_batch_size = x.shape[0]
        # Ensure global_features is float and has the correct shape for concatenation
        # Reshape global_features to (current_batch_size, global_feature_dim)
        global_features = global_features.float().view(
            current_batch_size, self.global_feature_dim
        )
        x = torch.cat(
            [x, global_features], dim=1
        )  # Concatenate along the feature dimension

        # Apply fully connected layers for regression
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Final output for regression

        return x
