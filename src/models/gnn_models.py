import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.aggr import AttentionalAggregation  # For graph-level readout


class GNN(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        global_feature_dim,
        hidden_channels,
        num_layers,
        dropout_rate,
    ):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.dropout_rate = dropout_rate

        # GNN layer lists
        # Use nn.ModuleList to hold multiple layers
        self.convs = nn.ModuleList()  # Convolutional Layers
        # Layer Normalization Layers
        self.layer_norms = nn.ModuleList()

        # First convolutional layer
        edge_nn_1 = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels * node_feature_dim),
            nn.ReLU(),
        )
        self.convs.append(
            pyg_nn.NNConv(node_feature_dim, hidden_channels, edge_nn_1, aggr="mean")
        )
        self.layer_norms.append(pyg_nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_channels * hidden_channels),
                nn.ReLU(),
            )
            self.convs.append(
                pyg_nn.NNConv(hidden_channels, hidden_channels, edge_nn, aggr="mean")
            )
            self.layer_norms.append(pyg_nn.LayerNorm(hidden_channels))

        # Initialize Attentional Aggregation (Readout/Pooling Layer)
        # It takes the concat_dim size as input
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))

        # Fully connected layers for the final prediction
        # concatenates GNN output with global molecular features to form combined dataset
        self.fc1 = nn.Linear(hidden_channels + global_feature_dim, hidden_channels // 2)
        self.bn1 = nn.BatchNorm1d(
            hidden_channels // 2
        )  # BatchNorm for the final FCN layers
        self.fc2 = nn.Linear(hidden_channels // 2, 1)  # Output is a single pGI50 value

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        global_features = (
            data.global_features
        )  # Access the global features stored in the Data object

        # Ensure node features (x) and edge features (edge_attr) are float
        x = x.float()
        edge_attr = edge_attr.float()

        # Apply GNN convolutional layers
        for i, conv in enumerate(self.convs):
            identity = x  # Store the input for residual connection
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

            # Add residual connection if not first layer (dimensions match)
            if i > 0:  # Skip first layer as dimensions differ
                x = x + identity

            # Apply LayerNorm after convolution
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)

            x = F.dropout(
                x, p=self.dropout_rate, training=self.training
            )  # Dropout for regularization

        # Readout layer: Aggregate node embeddings to a single graph embedding
        # Attentional Aggregation applies learned attention scores to atom embeddings
        x = self.pool(x, batch)

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
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Final output for regression

        return x
