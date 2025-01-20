import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural Network Model
class EventPositionModel(nn.Module):
    def __init__(self, input_channels=54, sequence_length=16384, use_transformer=True):
        """
        Initialize the model with transformer and graph neural network components.

        Parameters:
            input_channels (int): Number of input channels (e.g., detectors).
            sequence_length (int): Length of the input sequence (e.g., waveform length).
            use_transformer (bool): Whether to include a transformer encoder.
        """
        super(EventPositionModel, self).__init__()

        self.use_transformer = use_transformer

        # Transformer Encoder
        if use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=input_channels, nhead=3, dim_feedforward=128),
                num_layers=3
            )

        # Graph Neural Network
        self.graph_conv = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Compute the number of features dynamically
        test_input = torch.zeros(1, input_channels, sequence_length)
        with torch.no_grad():
            if use_transformer:
                test_input = test_input.permute(2, 0, 1)  # (seq_len, batch, channels)
                test_input = self.transformer(test_input)
                test_input = test_input.permute(1, 2, 0)  # (batch, channels, seq_len)
            test_input = test_input.mean(dim=-1)  # Aggregate sequence dimension
            test_input = self.graph_conv(test_input)
            flattened_size = test_input.flatten(start_dim=1).shape[1]

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Output: 3D position (x, y, z)
        )

    def forward(self, x):
        # Transformer
        if self.use_transformer:
            x = x.permute(2, 0, 1)  # (seq_len, batch, channels)
            x = self.transformer(x)
            x = x.permute(1, 2, 0)  # (batch, channels, seq_len)

        # Graph Convolution
        x = x.mean(dim=-1)  # Aggregate sequence dimension
        x = self.graph_conv(x)

#        print(f"Shape before fully connected layers: {x.shape}")

        # Flatten and Fully Connected
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
