import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import csv
from pathlib import Path
from datasets.multi_session_dataset_v2_with_imu_gravity_align import create_magnetic_imu_dataset_dataloader
from datasets.transforms import DefaultTransform, YawAugmentO2Transform, ComposeTransform
from datasets.utils import denorm_y
from network.loc_losses import WeightedSmoothL1
from network.mag_imu_eqnio_fusion_model import MagImuEqNioFusionModelV1

from train.utils import move_to_device

def test(
    model, 
    loader, 
    criterion, 
    device, 
    ckpt_path:Path, 
    res_dir:Path, 
    *, 
    y_norm_mode="per_file_minmax", 
    stats=None
    ):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    total_loss = 0.0
    total_samples = 0

    # real-space error
    sum_l2 = 0.0          # mean L2 error累计（单位：你的坐标单位）
    sum_l1 = 0.0          # mean |dx|+|dy|
    sum_dx2 = 0.0         # RMSE用
    sum_dy2 = 0.0
    all_preds, all_labels, all_errors = [], [], []

    with torch.no_grad():
        for batch in loader or []:
            batch = move_to_device(batch, device)
            y = batch["y"].to(device, non_blocking=True).float()
            preds, _ = model(batch["x_mag"], batch["x_acc"], batch["x_v1"], batch["x_v2"])
            loss = criterion(preds, y)

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs

            # ===== 逆归一化到真实坐标 =====
            # 需要 batch["y_raw"] (B,2)，以及 per_file 模式下 batch["y_stats"] (B,4)
            preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
            y_real = batch["y_raw"].to(device, non_blocking=True).float()

            diff = preds_real - y_real
            l2 = torch.norm(diff, dim=1)          # (B,)
            l1 = diff.abs().sum(dim=1)            # (B,)

            preds_np = preds_real.cpu().numpy()
            labels_np = y_real.cpu().numpy()
            errors = np.linalg.norm(preds_np - labels_np, axis=1)

            sum_l2 += l2.sum().item()
            sum_l1 += l1.sum().item()
            sum_dx2 += (diff[:, 0] ** 2).sum().item()
            sum_dy2 += (diff[:, 1] ** 2).sum().item()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)
            all_errors.extend(errors)

    # 指标计算
    denom = max(total_samples, 1)
    val_loss = total_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom
    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    print(f"val_loss={val_loss:.6f} | "
            f"mean_l1={mean_l1:.3f} mean_l2={mean_l2:.3f} rmse_x={rmse_x:.3f} rmse_y={rmse_y:.3f} rmse_2d={rmse_2d:.3f}")
    
    res_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"1653_wenguan_test4_loc_res_meanerr_{mean_l2:.4f}.csv"
    output_csv = res_dir / file_name

    results_df = pd.DataFrame(
        {
            "pred_x": [pred[0] for pred in all_preds],
            "pred_y": [pred[1] for pred in all_preds],
            "true_x": [label[0] for label in all_labels],
            "true_y": [label[1] for label in all_labels],
            "euclidean_error": all_errors,
        }
    )

    metrics = {
        "val_loss": val_loss,
        "mean_l1": mean_l1,
        "mean_l2": mean_l2,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_2d": rmse_2d,
        "num_samples": total_samples,
    }
    
    # 先写指标，再写明细
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
        w.writerow([])  # 空行分隔
        results_df.to_csv(f, index=False)

    print(f"结果已保存到: {output_csv}")


if __name__ == "__main__":

    # test_dir = Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-zscore-trans-all-feature-5" / "test1"
    # test_dir = Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-trans-all-feature-5" / "test1"
    # test_dir = Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-filter-zscore-all-feature-5" / "test3"

    test_dir = Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-all-feature-5-v3" / "test4"


    ckpt_path = Path("checkpoints") / "mag_imu_eqnio" / "mag_imu_eqnio_best_20260326_1653_rmse_2d_0.628_xinxi.pt"
    res_dir = Path("runs") / "loc_res" / "mag_imu_eqnio_different_posture"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    seq_len = 128     
    stride = 10
    y_norm_mode = "per_file_minmax"
    canonicalize_mag=True
    canonicalize_imu=True
    gravity_align = True

    feature_transform = ComposeTransform([
            DefaultTransform(),
            YawAugmentO2Transform(p_reflect=0.0),
    ])
    criterion = WeightedSmoothL1(beta=0.05, w_x=1.3, w_y=1.0).to(device)

    model = MagImuEqNioFusionModelV1(
        mag_input_dim=3,
        mag_d_model=128,
        seq_len=seq_len,
        use_frame_net=True,
        canonicalize_mag=canonicalize_mag,
        canonicalize_imu=canonicalize_imu,
        
        frame_hidden=64,
        frame_depth=3,
        imu_c=64,
        imu_blocks=4,
        imu_out_dim=32,
        head_hidden=256,
        out_dim=2,
        dropout=0.1
    ).to(device)

    test_loader = create_magnetic_imu_dataset_dataloader(
        test_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=False,
        pin_memory=pin_memory,
        transform=feature_transform,  # 验证也保持一致（若想关闭增强，可写一个不含 YawAug 的 transform）
        stats=None,
        seq_len=seq_len,
        stride=stride,
        normalize_mag=False,
        normalize_imu=False,
        y_norm_mode=y_norm_mode,
        cache_in_memory=True,
        gravity_align=gravity_align,
        use_linear_acc=False,
    )

    test(model, test_loader, criterion, device, ckpt_path, res_dir)
