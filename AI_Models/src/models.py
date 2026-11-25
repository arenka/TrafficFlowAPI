"""
Deep Learning Architectures for Spatio-Temporal Multi-Horizon Traffic Flow Forecasting:
  1. TrafficLSTM             - Multi-layer LSTM (Ma et al., 2015)
  2. TrafficGRU              - Multi-layer GRU (Cho et al., 2014 / Fu et al., 2016)
  3. TrafficBiLSTMAttention  - Bidirectional LSTM + Temporal Self-Attention
"""
import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = self.norm(out[:, -1, :])
        return self.head(last_hidden)


class TrafficGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = self.norm(out[:, -1, :])
        return self.head(last_hidden)


class TrafficBiLSTMAttention(nn.Module):
    """Bidirectional LSTM with additive temporal attention over the 60-min history."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_horizons: int = 3,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bi_dim = hidden_size * 2
        self.norm = nn.LayerNorm(bi_dim)
        self.attn = nn.Sequential(
            nn.Linear(bi_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(bi_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        out = self.norm(out)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = torch.sum(weights * out, dim=1)       # (B, 2*H)
        return self.head(context)


def build_model(model_type: str, input_dim: int = 6, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, num_horizons: int = 3) -> nn.Module:
    key = model_type.lower()
    if key == "lstm":
        return TrafficLSTM(input_dim, hidden_size, num_layers, dropout, num_horizons)
    if key == "gru":
        return TrafficGRU(input_dim, hidden_size, num_layers, dropout, num_horizons)
    if key in ("bilstm_attention", "bilstm-attention", "bilstm_attn"):
        return TrafficBiLSTMAttention(input_dim, hidden_size, num_layers, dropout, num_horizons)
    raise ValueError(f"Unknown model_type: {model_type}")
