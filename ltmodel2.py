import torch
import torch.nn as nn

class LSTMTransformerModel(nn.Module):
    def __init__(self, input_channels=54, sequence_length=16384,
                 lstm_hidden_size=64, lstm_output_features=10,  # LSTM Outputs 10 Features
                 num_transformer_layers=8, transformer_heads=2, transformer_dim_feedforward=128):
        super(LSTMTransformerModel, self).__init__()

        # Bidirectional LSTM (Doubles Output Size)
        self.lstm = nn.LSTM(input_size=sequence_length, hidden_size=lstm_hidden_size,
                            num_layers=6, batch_first=True, bidirectional=True)

        # Linear Projection to Ensure Output Matches Transformer Expected Input
        self.lstm_projection = nn.Linear(lstm_hidden_size * 2, lstm_output_features)  # 128 → 10

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_channels, nhead=transformer_heads,
                                       dim_feedforward=transformer_dim_feedforward, activation="gelu"),
            num_layers=num_transformer_layers
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(input_channels * lstm_output_features, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # Output (x, y, z)
        )

    def forward(self, x):
        """
        Forward pass through LSTM, Transformer, and FC layers.
        Input Shape: (batch_size, 54, 16384)
        """

        # LSTM Feature Extraction (BiLSTM gives (batch_size, 54, 128))
        x, _ = self.lstm(x)

        # Project 128 → 10 Features to Match Expected Transformer Input
        x = self.lstm_projection(x)  # Shape: (batch_size, 54, 10)

        # Permute to Match Transformer Expected Shape (batch_size, seq_len=10, embed_dim=54)
        x = x.permute(0, 2, 1)  # Now (batch_size, 10, 54)

        # Transformer Processing
        x = self.transformer(x)

        # Permute back before Fully Connected Layers
        x = x.permute(0, 2, 1)  # Back to (batch_size, 54, 10)

        # Flatten for FC layers
        x = x.flatten(start_dim=1)  # (batch_size, 540)

        # Fully Connected to Predict (x, y, z)
        x = self.fc(x)

        return x

