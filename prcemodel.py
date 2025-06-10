import torch
import torch.nn as nn

class LSTMTransformerModel(nn.Module):
    def __init__(self, input_channels=54, sequence_length=16384,
                 lstm_hidden_size=64, lstm_output_features=20,  # LSTM Outputs 20 Features
                 num_transformer_layers=5, transformer_heads=3, transformer_dim_feedforward=128):
        super(LSTMTransformerModel, self).__init__()

        # Bidirectional LSTM (Doubles Output Size)
        self.lstm = nn.LSTM(input_size=sequence_length, hidden_size=lstm_hidden_size,
                            num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)

        # Linear Projection to Ensure Output Matches Transformer Expected Input
        self.lstm_projection = nn.Linear(lstm_hidden_size * 2, lstm_output_features)  # 128 - 20

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_channels, nhead=transformer_heads,
                                       dim_feedforward=transformer_dim_feedforward, dropout=0.2,
                                       activation="gelu"), num_layers=num_transformer_layers)

        # Fully Connected Layers for Position Prediction (x, y, z)
        self.fc_position = nn.Sequential(
            nn.Linear(input_channels * lstm_output_features, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # Predicts (x, y, z)
        )

        # Fully Connected Layer for Classification (ER/NR)
        self.fc_classification = nn.Sequential(
            nn.Linear(input_channels * lstm_output_features, 256),
            nn.GELU(),
            nn.Linear(256, 1),  # Predicts a single value (logit)
            nn.Sigmoid()  # Converts logit to probability
        )

        # Fully Connected Layer for Energy Regression (Uses ReLU)
        self.fc_energy = nn.Sequential(
            nn.Linear(input_channels * lstm_output_features, 256),
            nn.ReLU(),  # Ensures energy is always non-negative
            nn.Linear(256, 1)  # Predicts energy (eV)
        )

    def forward(self, x):
        """
        Forward pass through LSTM, Transformer, and FC layers.
        Input Shape: (batch_size, 54, 16384)
        """

        # LSTM Feature Extraction (BiLSTM gives (batch_size, 54, 128))
        x, _ = self.lstm(x)

        # Project 128 - 20 Features to Match Expected Transformer Input
        x = self.lstm_projection(x)  # Shape: (batch_size, 54, 20)

        # Permute to Match Transformer Expected Shape (batch_size, seq_len=20, embed_dim=54)
        x = x.permute(0, 2, 1)  # Now (batch_size, 20, 54)

        # Transformer Processing
        x = self.transformer(x)

        # Permute back before Fully Connected Layers
        x = x.permute(0, 2, 1)  # Back to (batch_size, 54, 20)

        # Flatten for FC layers
        x = x.flatten(start_dim=1)  # (batch_size, 1080)

        # Separate outputs
        position_output = self.fc_position(x)  # Predict (x, y, z)
        classification_output = self.fc_classification(x)  # Predict ER/NR (0 or 1)
        energy_output = self.fc_energy(x)  # Predict energy (eV)

        return position_output, classification_output, energy_output

