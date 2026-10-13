import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.name = "stgcn"
        self.config = config or {}
        hidden_size = self.config.get("hidden_size", 64)
        self.adj = torch.eye(70)
        self.gru = nn.GRU(hidden_size, hidden_size, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 48 * 70)
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.reshape(B * T, N, F).mean(dim=2)  # (B*T, N)
        x = x.unsqueeze(-1).expand(-1, -1, 64)  # (B*T, N, 64)
        x = x.reshape(B, T, -1)  # (B, T, N*64)
        out, _ = self.gru(x)
        preds = self.fc(out[:, -1, :])
        return preds.reshape(B, 48, 70)

def build_stgcn_model(config=None):
    return STGCN(config)