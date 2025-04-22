import torch
import torch.nn as nn
import torch.nn.functional as F

class StockTransformer(nn.Module):
    def __init__(self, stocks, time_steps, channels, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.input_proj = nn.Linear(channels, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(dim_feedforward, 1)
        self.time_steps = time_steps

    def forward(self, x):
        # x: [batch, time_steps, channels]
        x = self.input_proj(x)  # [batch, time_steps, dim_feedforward]
        x = x.permute(1, 0, 2)  # [time_steps, batch, dim_feedforward]
        x = self.transformer_encoder(x)  # [time_steps, batch, dim_feedforward]
        x = x.permute(1, 0, 2)  # [batch, time_steps, dim_feedforward]
        x = self.output_proj(x)  # [batch, time_steps, 1]
        x = x.mean(dim=1)  # [batch, 1]
        return x 