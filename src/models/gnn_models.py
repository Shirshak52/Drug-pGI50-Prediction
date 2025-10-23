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
        attention_heads,
    ):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.dropout_rate = dropout_rate

        concat_dim = hidden_channels * attention_heads

        # GNN convolutional layers
        # Use nn.ModuleList to hold multiple layers
        self.convs = nn.ModuleList()

        # First layer: input node_feature_dim to hidden_channels (output concat_dim)
        self.convs.append(
            pyg_nn.GATv2Conv(
                node_feature_dim,
                hidden_channels,
                edge_dim=edge_feature_dim,
                heads=attention_heads,
                concat=True,
            )
        )

        # Subsequent layers: concat_dim to hidden_channels (output concat_dim)
        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.GATv2Conv(
                    concat_dim,
                    hidden_channels,
                    edge_dim=edge_feature_dim,
                    heads=attention_heads,
                    concat=True,
                )
            )

        # Layer Normalization Layers
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.LayerNorm(concat_dim))

        # Initialize Attentional Aggregation (Readout/Pooling Layer)
        # It takes the concat_dim size as input
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(concat_dim, 1))

        # Fully connected layers for the final prediction
        # The input to the first FC layer combines concat_dim + global_feature_dim
        # i.e. concatenates GNN output with global molecular features to form combined dataset
        self.fc1 = nn.Linear(concat_dim + global_feature_dim, concat_dim // 2)
        self.fc2 = nn.Linear(concat_dim // 2, 1)  # Output is a single pGI50 value

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
            x = conv(x, edge_index, edge_attr)

            # Apply LayerNorm after convolution
            if self.bns[i] is not None:
                x = self.bns[i](x)

            x = F.relu(x)
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
        x = F.relu(x)
        x = self.fc2(x)  # Final output for regression

        return x
