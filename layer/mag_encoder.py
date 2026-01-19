import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .timemixer_multiscale_encoder_v2 import TimeMixerMultiScaleEncoderV3
from .multi_scale_attention_fusion import MultiScaleAttentionFusionV2


class TimeMixerMagFeatureAdapter(nn.Module):
    """
    包装你现有的 MagneticLocalizationTimeMixer / 或者只要具备相同子模块属性：
      - timemixer_encoder(x) -> scale_features: list[(B,D)]
      - attention_fusion(scale_features) -> fused(B,D), attn(B,S)

    输出：
      mag_fused: (B,D)
      mag_attn: (B,S) or None
      mag_scales: list[(B,D)]（可选保留，方便 debug）
    """
    def __init__(self,
        input_dim=3,
        d_model=128,
        seq_len=128,             # 训练时地磁序列长度
        down_sampling_window=2,  # 下采样基数
        down_sampling_layers=2,  # 得到 M+1 个尺度
        num_pdm_blocks=1,        # PDM层数
        moving_avg_kernel=25,    # 分解核大小
        nhead=8,                 # 注意力头数
        num_layers=2):           # 编码器层数
                    
        super().__init__()
        self.timemixer_encoder = TimeMixerMultiScaleEncoderV3(
            input_dim=input_dim,
            d_model=d_model,
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            moving_avg_kernel=moving_avg_kernel,
            num_pdm_blocks=num_pdm_blocks,
            nhead=nhead,
            num_layers=num_layers,
            enc_out_dim = 64
        )

        self.feature_dim = 64
        self.seq_len = seq_len
        self.attention_fusion = MultiScaleAttentionFusionV2(
            feature_dim=self.feature_dim,
            num_scales=down_sampling_layers + 1,
            d_k=64,
            residual=True
        )

    def forward(self, x_mag: Tensor):
        """
        x_mag: (B,T,3) 地磁序列（已经 canonicalize）
        """
        # 你原本 forward 里会做 pad/截断，这里也复用同样逻辑（避免外部处理）
        if x_mag.size(1) != self.seq_len:
            if x_mag.size(1) > self.seq_len:
                x_mag = x_mag[:, -self.seq_len:, :]
            else:
                pad = self.seq_len - x_mag.size(1)
                x_mag = torch.nn.functional.pad(x_mag, (0, 0, 0, pad))

        scale_features = self.timemixer_encoder(x_mag)   # list[(B,D)]
        fused, attn = self.attention_fusion(scale_features)  # (B,D), (B,S)
        return fused, attn, scale_features