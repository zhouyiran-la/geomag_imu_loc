
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from layer.eqnio_frame_net_o2 import EqNIOFrameNetO2
from layer.mag_encoder import TimeMixerMagFeatureAdapter
from layer.imu_encoder import RoninLikeImuEncoder
from layer.feature_fusion import ConcatFusionHead
from network.utils import build_o2_features_from_imu, apply_frame_xy


class MagImuEqNioFusionModelV1(nn.Module):
    """
    Baseline A：en
      - canonicalize: mag/acc/v1/v2
      - mag: 用你现有 TimeMixer 多尺度 + 内部 attention_fusion 得到 mag_fused
      - imu: 用 RONIN-like ResNet1D 得到 imu_feat（单向量）
      - 融合: concat(mag_fused, imu_feat) -> head 回归 (x,y)
    """
    def __init__(
        self,
        mag_input_dim=3,
        mag_d_model=128,
        seq_len=128,       
        *,
        frame_hidden: int = 64,
        frame_depth: int = 3,
        imu_c: int = 128,
        imu_blocks: int = 6,
        imu_out_dim: int = 64,
        head_hidden: int = 256,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ① FrameNet（尽可能保持 EqNIO 源码默认参数）
        self.frame_net = EqNIOFrameNetO2(
            dim_in=3,
            dim_out=2,
            scalar_dim_in=9,
            pooling_dim=1,
            hidden_dim=frame_hidden,
            scalar_hidden_dim=frame_hidden,
            depth=frame_depth,
            stride=1,
            padding='same',
            kernel=(16, 1),
            bias=False
        )

        # ② Mag 特征提取（复用你现有模型内部的 timemixer + attention_fusion）
        self.mag_encoder = TimeMixerMagFeatureAdapter(
            input_dim=mag_input_dim,
            d_model=mag_d_model,
            seq_len=seq_len,
            down_sampling_window=2,
            down_sampling_layers=2,
            num_pdm_blocks=2,
            moving_avg_kernel=11, 
            nhead=8,
            num_layers=2
        )
        mag_dim = int(self.mag_encoder.feature_dim)

        # ③ IMU encoder（RONIN-like）
        self.imu_encoder = RoninLikeImuEncoder(
            in_dim=9,
            c=imu_c,
            blocks=imu_blocks,
            kernel=5,
            dropout=dropout,
            out_dim=imu_out_dim
        )

        # ④ 融合 head
        self.fusion_head = ConcatFusionHead(
            mag_dim=mag_dim,
            imu_dim=imu_out_dim,
            hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout
        )

    def forward(self, mag: Tensor, acc: Tensor, v1: Tensor, v2: Tensor):
        """
        输入:
          mag/acc/v1/v2: (B,T,3)，应当已经完成 gravity-align（pitch/roll对齐）
        输出:
          out: (B,2)
          extras: dict (包含 Fm、mag attention 等，方便训练/可视化)
        """
        # 1) 构造 FrameNet O(2) 输入并预测 frame
        vec, scalars = build_o2_features_from_imu(acc, v1, v2)  # vec:(B,T,2,3), scalars:(B,T,9)
        Fm = self.frame_net(vec, scalars)                      # (B,2,2)

        # 2) canonicalize
        mag_c = apply_frame_xy(mag, Fm, inverse=True)
        acc_c = apply_frame_xy(acc, Fm, inverse=True)
        v1_c  = apply_frame_xy(v1,  Fm, inverse=True)
        v2_c  = apply_frame_xy(v2,  Fm, inverse=True)

        # 3) mag features（fused）
        mag_fused, mag_attn, mag_scales = self.mag_encoder(mag_c)  # (B,Dm), (B,S), list

        # 4) imu feature（单向量）
        imu_in = torch.cat([acc_c, v1_c, v2_c], dim=-1)  # (B,T,9)
        imu_feat = self.imu_encoder(imu_in)              # (B,Dimu)

        # 5) 融合回归
        out = self.fusion_head(mag_fused, imu_feat)

        extras = {
            "Fm": Fm,
            "mag_attn": mag_attn,
            "mag_scales": mag_scales,
            "mag_fused": mag_fused,
            "imu_feat": imu_feat,
        }
        return out, extras