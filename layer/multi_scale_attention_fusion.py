import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttentionFusion(nn.Module):
    """多尺度注意力特征融合（尺度级加性注意力）"""
    def __init__(self, feature_dim=16, num_scales=3, dropout=0.1, residual=True):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.residual = residual

        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # 可学习残差系数（更稳，避免一开始 residual 太强）
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, scale_features):
        """
        scale_features: list of (B, D)
        return: fused (B, D), attn_weights (B, S)
        """
        feats = torch.stack(scale_features, dim=1)         # (B,S,D)
        attn_scores = self.attention_net(feats).squeeze(-1)  # (B,S)
        attn_weights = F.softmax(attn_scores, dim=1)         # (B,S)

        fused0 = torch.sum(feats * attn_weights.unsqueeze(-1), dim=1)  # (B,D)

        if self.residual:
            fused = fused0 + torch.tanh(self.alpha) * self.fusion_fc(fused0)
        else:
            fused = self.fusion_fc(fused0)

        return fused, attn_weights
    

class MultiScaleAttentionFusionV2(nn.Module):
    """多尺度 QKV 融合：用全局 query 对尺度做注意力"""
    def __init__(self, feature_dim=16, num_scales=3, d_k=None, dropout=0.1, residual=True):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.d_k = d_k or max(16, feature_dim // 2)

        # 全局 query：看全尺度再决定关注哪个尺度
        self.query_net = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, self.d_k),
        )

        self.key = nn.Linear(feature_dim, self.d_k)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        self.fusion_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.residual = residual

    def forward(self, scale_features):
        feats = torch.stack(scale_features, dim=1)  # (B,S,D)
        B, S, D = feats.shape

        q = self.query_net(feats.reshape(B, S * D)).unsqueeze(1)  # (B,1,d_k)
        k = self.key(feats)                                       # (B,S,d_k)
        v = self.value(feats)                                     # (B,S,D)

        scores = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.d_k)  # (B,1,S)
        attn = torch.softmax(scores, dim=-1)                                # (B,1,S)
        attn = self.dropout(attn)
        fused0 = torch.matmul(attn, v).squeeze(1)  # (B,D)
        if self.residual:
            fused = fused0 + torch.tanh(self.alpha) * self.fusion_fc(fused0)
        else:
            fused = self.fusion_fc(fused0)
        return fused, attn.squeeze(1)              # (B,S)
    

