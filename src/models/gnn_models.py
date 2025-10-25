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

        # Compress global features to match the size of hidden_channels (late fusion)
        # GNN layers output hidden_channels size,
        # so global features also need to be compressed to the same size
        self.late_fusion_global_features_compressor = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                hidden_channels * 2, hidden_channels
            ),  # Output size: hidden_channels
        )

        # Calculate input dimension for first layer for early fusion (atom + global features)
        # Early fusion meaning:
        # * Each molecule has 1 row of global features
        # * Each atom in the molecule will have these global features concatenated to its own features
        self.initial_input_dim = node_feature_dim

        # GNN layer lists
        # Use nn.ModuleList to hold multiple layers
        self.convs = nn.ModuleList()  # Convolutional Layers
        self.layer_norms = nn.ModuleList()  # Layer Normalization Layers

        # First convolutional layer
        edge_nn_1 = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels * self.initial_input_dim),
            nn.ReLU(),
        )
        self.convs.append(
            pyg_nn.NNConv(
                self.initial_input_dim, hidden_channels, edge_nn_1, aggr="mean"
            )
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
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_channels, 1))

        # Fully connected layers for the final prediction (Fusion Decoder)
        # concatenates GNN output with global molecular features to form combined dataset
        fc_input_dim = hidden_channels * 2  # GNN output + resized global features
        fc_first_layer_output_dim = 512
        self.fc_layers = nn.Sequential(
            # FCN1
            nn.Linear(fc_input_dim, fc_first_layer_output_dim),
            nn.BatchNorm1d(fc_first_layer_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # FCN2
            nn.Linear(fc_first_layer_output_dim, fc_first_layer_output_dim // 2),
            nn.BatchNorm1d(fc_first_layer_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # FCN3
            nn.Linear(fc_first_layer_output_dim // 2, fc_first_layer_output_dim // 4),
            nn.BatchNorm1d(fc_first_layer_output_dim // 4),
            nn.ReLU(),
            # FCN4 - Output Layer
            nn.Linear(fc_first_layer_output_dim // 4, 1),  # Single pGI50 output
        )

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

        # Ensure features are float
        x = x.float()
        edge_attr = edge_attr.float()
        global_features = global_features.float()

        # Apply GNN convolutional layers
        for i, conv in enumerate(self.convs):
            identity = x  # Store the input for residual connection
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

            # Residual connection (only after the first layer where dimensions may differ)
            if i > 0 and x.shape == identity.shape:
                x = x + identity

            # Apply LayerNorm after convolution
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)

            x = F.dropout(
                x, p=self.dropout_rate, training=self.training
            )  # Dropout for regularization

        # Readout layer: Aggregate node embeddings to a single graph embedding (single row per molecule again)
        x = self.pool(x, batch)

        # Compress global features to match graph embeddings size for late fusion
        global_features = self.late_fusion_global_features_compressor(global_features)

        # Late fusion: Concatenate GNN output with resized global features
        x = torch.cat([x, global_features], dim=1)

        # Apply fully connected layers for regression
        x = self.fc_layers(x)  # Final output for regression

        return x
