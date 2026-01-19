import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embed import PositionalEmbedding

class PerScaleTransformerEncoder(nn.Module):
    """
    对每个尺度的序列做 Transformer 编码，然后注意力池化为一个向量。
    """

    def __init__(self, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, d_model // 4),
        )

    def forward(self, x):
        """
        x: (B, L, D)
        return: (B, D//4)
        """
        x = x + self.pos_embedding(x)
        h = self.encoder(x)           # (B, L, D)

        scores = self.attn(h)         # (B, L, 1)
        w = torch.softmax(scores, dim=1)
        pooled = (h * w).sum(dim=1)   # (B, D)

        return self.proj(pooled)

class PerScaleTransformerEncoderV2(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, out_dim=64, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.pos_embedding = PositionalEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # 关键：输出维度由 out_dim 决定
        if out_dim == d_model:
            self.proj = nn.Identity()
        else:
            hidden = min(out_dim * 2, d_model // 2)  # 稍微给点容量
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x):
        x = x + self.pos_embedding(x)
        h = self.encoder(x)                 # (B,L,d_model)

        w = torch.softmax(self.attn(h), dim=1)  # (B,L,1)
        pooled = (h * w).sum(dim=1)         # (B,d_model)

        return self.proj(pooled)            # (B,out_dim)
    

class PerScaleLSTMEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
        out_dim: int = 64,
        hidden_dim: int = None, # type: ignore
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_pos_encoding = use_pos_encoding

        # 可以复用你之前的 PositionalEncoding，如果需要的话
        if use_pos_encoding:
            self.pos_embedding = PositionalEmbedding(d_model)

        if hidden_dim is None:
            hidden_dim = d_model

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # 和 Transformer 版类似的 attention pooling
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        # 输出投影到 out_dim
        if out_dim == lstm_out_dim:
            self.proj = nn.Identity()
        else:
            hidden = min(out_dim * 2, lstm_out_dim // 2) if lstm_out_dim >= 2 else out_dim
            self.proj = nn.Sequential(
                nn.Linear(lstm_out_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x):
        """
        x: (B, L, d_model)
        return: (B, out_dim)
        """
        if self.use_pos_encoding:
            x = x + self.pos_embedding(x)

        # LSTM 输出：h: (B, L, lstm_out_dim)
        h, _ = self.lstm(x)

        # 注意力池化： (B, L, lstm_out_dim) -> (B, L, 1) -> (B, lstm_out_dim)
        w = torch.softmax(self.attn(h), dim=1)   # (B, L, 1)
        pooled = (h * w).sum(dim=1)              # (B, lstm_out_dim)

        return self.proj(pooled)                 # (B, out_dim)
    


class PerScaleRNNEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
        out_dim: int = 64,
        hidden_dim: int = None,  # type: ignore
        bidirectional: bool = False,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_pos_encoding = use_pos_encoding

        if use_pos_encoding:
            self.pos_embedding = PositionalEmbedding(d_model)

        if hidden_dim is None:
            hidden_dim = d_model

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",          # 默认就是 tanh
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        # attention pooling
        self.attn = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_out_dim // 2),
            nn.ReLU(),
            nn.Linear(rnn_out_dim // 2, 1),
        )

        # 输出投影
        if out_dim == rnn_out_dim:
            self.proj = nn.Identity()
        else:
            hidden = min(out_dim * 2, rnn_out_dim // 2) if rnn_out_dim >= 2 else out_dim
            self.proj = nn.Sequential(
                nn.Linear(rnn_out_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        return: (B, out_dim)
        """
        if self.use_pos_encoding:
            x = x + self.pos_embedding(x)   # (B,L,d_model)

        h, _ = self.rnn(x)                 # h: (B, L, rnn_out_dim)

        w = torch.softmax(self.attn(h), dim=1)  # (B, L, 1)
        pooled = (h * w).sum(dim=1)             # (B, rnn_out_dim)

        return self.proj(pooled)                # (B, out_dim)
    
class PerScaleTCNEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        out_dim: int = 64,
        hidden_dim: int = None,  # 用与不用都行，这里保持接口一致 # type: ignore
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_pos_encoding: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.use_pos_encoding = use_pos_encoding

        if use_pos_encoding:
            self.pos_embedding = PositionalEmbedding(d_model)

        if hidden_dim is None:
            hidden_dim = d_model

        assert kernel_size % 2 == 1, "kernel_size 建议用奇数，方便做 padding"

        # TCN: 多层 dilated Conv1d + 残差
        layers = []
        in_channels = d_model
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # 简单的 causal padding

            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            layer = nn.Sequential(
                conv,
                nn.GELU(),
                nn.Dropout(dropout),
            )
            layers.append(layer)
            in_channels = d_model

        self.tcn_layers = nn.ModuleList(layers)

        # attention pooling
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # 输出投影
        if out_dim == d_model:
            self.proj = nn.Identity()
        else:
            hidden = min(out_dim * 2, d_model // 2) if d_model >= 2 else out_dim
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        return: (B, out_dim)
        """
        if self.use_pos_encoding:
            x = x + self.pos_embedding(x)        # (B,L,d_model)

        # (B,L,D) -> (B,D,L)
        h = x.permute(0, 2, 1)

        # 逐层 TCN
        for layer in self.tcn_layers:
            residual = h
            out = layer(h)                       # (B,D,L+padding_effect)
            # 因为 padding，可能长度会变长，这里裁掉尾部只保留原始长度
            if out.size(-1) != residual.size(-1):
                out = out[..., :residual.size(-1)]
            h = out + residual                   # 残差

        # (B,D,L) -> (B,L,D)
        h = h.permute(0, 2, 1)

        # attention pooling
        w = torch.softmax(self.attn(h), dim=1)   # (B,L,1)
        pooled = (h * w).sum(dim=1)             # (B,D)

        return self.proj(pooled)                # (B,out_dim)
