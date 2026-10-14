import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.name = "stgcn"
        self.config = config or {}
        hidden_size = self.config.get("hidden_size", 64)
        self.register_buffer('adj', torch.eye(70))
        # Simple GRU per station (shared weights)
        self.gru = nn.GRU(hidden_size, hidden_size, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 48)
        self.proj_in = nn.Linear(1, hidden_size)  # PM2.5 feature -> hidden
    
    def set_adjacency(self, adj: torch.Tensor):
        self.register_buffer('adj', adj)
    
    def forward(self, x):
        # x: (B, T, N, F) where F=1 (PM2.5)
        B, T, N, F = x.shape
        # Project PM2.5 to hidden size
        x = self.proj_in(x)  # (B, T, N, hidden)
        # Process each station independently with shared GRU
        x = x.reshape(B * N, T, -1)  # (B*N, T, hidden)
        out, _ = self.gru(x)  # (B*N, T, hidden)
        # Take last hidden state
        out = out[:, -1, :]  # (B*N, hidden)
        preds = self.fc(out)  # (B*N, 48)
        preds = preds.reshape(B, N, 48).transpose(1, 2)  # (B, 48, N)
        return preds

def build_stgcn_model(config=None):
    return STGCN(config)