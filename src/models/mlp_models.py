import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))  # Outputs 1 single pGI50 value
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.fc_layers(x)  # Output layer directly gives regression value
        return out
