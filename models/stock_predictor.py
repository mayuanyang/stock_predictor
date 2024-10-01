import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, dropout=0.1):
        super(StockPredictor, self).__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #print('The input shape', x.shape)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]
        x = x[:, -1, :]  # Take the last time step's output
        out = self.fc_out(x)  # [batch_size, output_size]
        return out