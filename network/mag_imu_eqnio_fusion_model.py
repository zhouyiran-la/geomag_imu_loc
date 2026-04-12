
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from layer.eqnio_frame_net_o2 import EqNIOFrameNetO2
from layer.mag_encoder import TimeMixerMagFeatureAdapter
from layer.imu_encoder import RoninLikeImuEncoder
from layer.feature_fusion import ConcatFusionHead
from network.utils import build_o2_features_from_imu, apply_frame_xy


# class MagImuEqNioFusionModelV1(nn.Module):
#     """
#     Baseline A：en
#       - canonicalize: mag/acc/v1/v2
#       - mag: 用你现有 TimeMixer 多尺度 + 内部 attention_fusion 得到 mag_fused
#       - imu: 用 RONIN-like ResNet1D 得到 imu_feat（单向量）
#       - 融合: concat(mag_fused, imu_feat) -> head 回归 (x,y)
#     """
#     def __init__(
#         self,
#         mag_input_dim=3,
#         mag_d_model=128,
#         seq_len=128, 
#         use_frame_net: bool = True,
#         canonicalize_mag: bool = True,
#         canonicalize_imu: bool = True,    
#         *,
#         frame_hidden: int = 64,
#         frame_depth: int = 3,
#         imu_c: int = 128,
#         imu_blocks: int = 6,
#         imu_out_dim: int = 64,
#         head_hidden: int = 256,
#         out_dim: int = 2,
#         dropout: float = 0.1,
        
#     ):
#         super().__init__()

#         self.use_frame_net = use_frame_net
#         self.canonicalize_mag = canonicalize_mag
#         self.canonicalize_imu = canonicalize_imu
#         if self.use_frame_net:
#             self.frame_net = EqNIOFrameNetO2(
#                 dim_in=3,
#                 dim_out=2,
#                 scalar_dim_in=9,
#                 pooling_dim=1,
#                 hidden_dim=frame_hidden,
#                 scalar_hidden_dim=frame_hidden,
#                 depth=frame_depth,
#                 stride=1,
#                 padding='same',
#                 kernel=(16, 1),
#                 bias=False
#             )
#         else:
#             self.frame_net = None

#         # ② Mag 特征提取（复用你现有模型内部的 timemixer + attention_fusion）
#         self.mag_encoder = TimeMixerMagFeatureAdapter(
#             input_dim=mag_input_dim,
#             d_model=mag_d_model,
#             seq_len=seq_len,
#             down_sampling_window=2,
#             down_sampling_layers=2,
#             num_pdm_blocks=2,
#             moving_avg_kernel=11, 
#             nhead=8,
#             num_layers=2
#         )
#         mag_dim = int(self.mag_encoder.feature_dim)

#         # ③ IMU encoder（RONIN-like）
#         self.imu_encoder = RoninLikeImuEncoder(
#             in_dim=9,
#             c=imu_c,
#             blocks=imu_blocks,
#             kernel=5,
#             dropout=dropout,
#             out_dim=imu_out_dim
#         )

#         # ④ 融合 head
#         self.fusion_head = ConcatFusionHead(
#             mag_dim=mag_dim,
#             imu_dim=imu_out_dim,
#             hidden=head_hidden,
#             out_dim=out_dim,
#             dropout=dropout
#         )

#     def forward(self, mag: Tensor, acc: Tensor, v1: Tensor, v2: Tensor):
#         """
#         输入:
#         mag/acc/v1/v2: (B,T,3)
#         - 对于完整模型，可输入 gravity-align 后的数据
#         - 对于消融实验，具体输入形式由 dataset 侧控制
#         输出:
#         out: (B,2)
#         extras: dict
#         """

#         Fm = None

#         # 1) 可选：预测等变旋转基
#         if self.use_frame_net and self.frame_net:
#             vec, scalars = build_o2_features_from_imu(acc, v1, v2)   # vec:(B,T,2,3), scalars:(B,T,9)
#             Fm = self.frame_net(vec, scalars)                        # (B,2,2)

#         if self.use_frame_net and self.canonicalize_mag:
#             mag_in = apply_frame_xy(mag, Fm, inverse=True) # type: ignore
#         else:
#             mag_in = mag

#         if self.use_frame_net and self.canonicalize_imu:
#             acc_in = apply_frame_xy(acc, Fm, inverse=True) # type: ignore
#             v1_in  = apply_frame_xy(v1,  Fm, inverse=True) # type: ignore
#             v2_in  = apply_frame_xy(v2,  Fm, inverse=True) # type: ignore
#         else:
#             acc_in = acc
#             v1_in  = v1
#             v2_in  = v2

#         # 3) mag features
#         mag_fused, mag_attn, mag_scales = self.mag_encoder(mag_in)

#         # 4) imu feature
#         imu_in = torch.cat([acc_in, v1_in, v2_in], dim=-1)   # (B,T,9)
#         imu_feat = self.imu_encoder(imu_in)

#         # 5) 融合回归
#         out = self.fusion_head(mag_fused, imu_feat)

#         extras = {
#             "Fm": Fm,
#             "mag_attn": mag_attn,
#             "mag_scales": mag_scales,
#             "mag_fused": mag_fused,
#             "imu_feat": imu_feat,
#             "used_frame_net": self.use_frame_net,
#             "canonicalize_mag": self.canonicalize_mag,
#             "canonicalize_imu":self.canonicalize_imu

#         }
#         return out, extras

class MagImuEqNioFusionModelV1(nn.Module):
    """
    Baseline A：
      - canonicalize: mag/acc/v1/v2
      - mag: 用 TimeMixer 多尺度 + 内部 attention_fusion 得到 mag_fused
      - imu: 可选，用 RONIN-like ResNet1D 得到 imu_feat（单向量）
      - 融合:
          1) use_imu_fusion=True  -> concat(mag_fused, imu_feat) -> head 回归 (x,y)
          2) use_imu_fusion=False -> 仅用 mag_fused -> head 回归 (x,y)
    """
    def __init__(
        self,
        mag_input_dim=3,
        mag_d_model=128,
        seq_len=128,
        use_frame_net: bool = True,
        canonicalize_mag: bool = True,
        canonicalize_imu: bool = True,
        use_imu_fusion: bool = True,   # 新增：控制是否使用 IMU 特征融合
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

        self.use_frame_net = use_frame_net
        self.canonicalize_mag = canonicalize_mag
        self.canonicalize_imu = canonicalize_imu
        self.use_imu_fusion = use_imu_fusion

        if self.use_frame_net:
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
        else:
            self.frame_net = None

        # ② Mag 特征提取
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

        # ③ IMU encoder（按原结构保留，可由 use_imu_fusion 控制是否实际使用）
        self.imu_encoder = RoninLikeImuEncoder(
            in_dim=9,
            c=imu_c,
            blocks=imu_blocks,
            kernel=5,
            dropout=dropout,
            out_dim=imu_out_dim
        )

        # ④ 融合 head：原融合 head 保留
        self.fusion_head = ConcatFusionHead(
            mag_dim=mag_dim,
            imu_dim=imu_out_dim,
            hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout
        )
        if not self.use_imu_fusion:
        # ⑤ 仅地磁回归 head：新增，用于 use_imu_fusion=False 的情况
            self.mag_head = nn.Sequential(
                nn.Linear(mag_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim)
            )

    def forward(self, mag: Tensor, acc: Tensor, v1: Tensor, v2: Tensor):
        """
        输入:
        mag/acc/v1/v2: (B,T,3)
        输出:
        out: (B,2)
        extras: dict
        """

        Fm = None

        # 1) 可选：预测等变旋转基
        if self.use_frame_net and self.frame_net:
            vec, scalars = build_o2_features_from_imu(acc, v1, v2)   # vec:(B,T,2,3), scalars:(B,T,9)
            Fm = self.frame_net(vec, scalars)                        # (B,2,2)

        if self.use_frame_net and self.canonicalize_mag:
            mag_in = apply_frame_xy(mag, Fm, inverse=True)  # type: ignore
        else:
            mag_in = mag

        if self.use_frame_net and self.canonicalize_imu:
            acc_in = apply_frame_xy(acc, Fm, inverse=True)  # type: ignore
            v1_in = apply_frame_xy(v1, Fm, inverse=True)    # type: ignore
            v2_in = apply_frame_xy(v2, Fm, inverse=True)    # type: ignore
        else:
            acc_in = acc
            v1_in = v1
            v2_in = v2

        # 3) mag features
        mag_fused, mag_attn, mag_scales = self.mag_encoder(mag_in)

        # 4) imu feature（由参数控制是否参与）
        imu_in = torch.cat([acc_in, v1_in, v2_in], dim=-1)
        imu_feat = imu_in
        if self.use_imu_fusion:
            imu_in = torch.cat([acc_in, v1_in, v2_in], dim=-1)   # (B,T,9)
            imu_feat = self.imu_encoder(imu_in)
            out = self.fusion_head(mag_fused, imu_feat)
        else:
            out = self.mag_head(mag_fused)

        extras = {
            "Fm": Fm,
            "mag_attn": mag_attn,
            "mag_scales": mag_scales,
            "mag_fused": mag_fused,
            "imu_feat": imu_feat,
            "used_frame_net": self.use_frame_net,
            "canonicalize_mag": self.canonicalize_mag,
            "canonicalize_imu": self.canonicalize_imu,
            "use_imu_fusion": self.use_imu_fusion,
        }
        return out, extras