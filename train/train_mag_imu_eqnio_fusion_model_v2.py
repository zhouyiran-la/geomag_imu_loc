# train_mag_imu_eqnio_baseline_a.py
import time
import csv
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.multi_session_dataset_v2_with_imu_gravity_align import create_magnetic_imu_dataset_dataloader
from datasets.transforms import DefaultTransform, YawAugmentO2Transform, ComposeTransform
from datasets.utils import denorm_y
from network.loc_losses import WeightedSmoothL1
from network.mag_imu_eqnio_fusion_model import MagImuEqNioFusionModelV1

from train.utils import move_to_device, canonical_consistency_loss, forward_with_yaw_pair

# ============================================================
# 1) 训练 / 验证（含真实坐标系指标）
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    *,
    lambda_can: float = 0.1,
    use_aug_task_loss: bool = True,
    use_can_loss: bool = True,
    use_mag_can_loss: bool = True,
    use_imu_can_loss: bool = True,
    y_norm_mode: str = "per_file_minmax",
    stats: Optional[dict] = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    total_task = 0.0
    total_task_aug = 0.0
    total_can = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for batch in loader or []:
        batch = move_to_device(batch, device)
        y = batch["y"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            # 原始样本 + 增强样本
            pred, extras, pred_aug, extras_aug = forward_with_yaw_pair(model, batch)

            # 原始样本监督
            loss_task = criterion(pred, y)

            # 增强样本监督
            if use_aug_task_loss:
                loss_task_aug = criterion(pred_aug, y)
            else:
                loss_task_aug = torch.zeros((), device=device, dtype=loss_task.dtype)

            # 等变一致性约束
            Fm = extras.get("Fm", None) if isinstance(extras, dict) else None
            Fm_aug = extras_aug.get("Fm", None) if isinstance(extras_aug, dict) else None

            can_enabled = (
                use_can_loss
                and lambda_can > 0
                and (use_mag_can_loss or use_imu_can_loss)
                and Fm is not None
                and Fm_aug is not None
            )

            if can_enabled:
                loss_can = canonical_consistency_loss(
                    Fm=Fm,  # type: ignore
                    Fm_aug=Fm_aug,  # type: ignore
                    mag=batch["x_mag"],
                    acc=batch["x_acc"],
                    v1=batch["x_v1"],
                    v2=batch["x_v2"],
                    mag_aug=batch["aug"]["x_mag"],
                    acc_aug=batch["aug"]["x_acc"],
                    v1_aug=batch["aug"]["x_v1"],
                    v2_aug=batch["aug"]["x_v2"],
                    w_mag=1.0 if use_mag_can_loss else 0.0,
                    w_imu=1.0 if use_imu_can_loss else 0.0,
                    reduction="mean",
                )
            else:
                loss_can = torch.zeros((), device=device, dtype=loss_task.dtype)

            # 总损失
            loss = loss_task + loss_task_aug + lambda_can * loss_can

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_task += float(loss_task.item()) * bs
        total_task_aug += float(loss_task_aug.item()) * bs
        total_can += float(loss_can.item()) * bs
        total_samples += bs

    denom = max(total_samples, 1)
    return (
        total_loss / denom,
        {
            "loss_task": total_task / denom,
            "loss_task_aug": total_task_aug / denom,
            "loss_can": total_can / denom,
        }
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    y_norm_mode: str = "per_file_minmax",
    stats: Optional[dict] = None,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # 真实坐标系误差统计
    sum_l2 = 0.0
    sum_l1 = 0.0
    sum_dx2 = 0.0
    sum_dy2 = 0.0

    for batch in loader or []:
        batch = move_to_device(batch, device)
        y = batch["y"].to(device, non_blocking=True).float()
        preds, _ = model(batch["x_mag"], batch["x_acc"], batch["x_v1"], batch["x_v2"])
        loss = criterion(preds, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        # ===== 逆归一化到真实坐标 =====
        preds_real = denorm_y(preds, batch, y_norm_mode=y_norm_mode, stats=stats, device=device)
        y_real = batch["y_raw"].to(device, non_blocking=True).float()

        diff = preds_real - y_real
        l2 = torch.norm(diff, dim=1)        # (B,)
        l1 = diff.abs().sum(dim=1)          # (B,)

        sum_l2 += float(l2.sum().item())
        sum_l1 += float(l1.sum().item())
        sum_dx2 += float((diff[:, 0] ** 2).sum().item())
        sum_dy2 += float((diff[:, 1] ** 2).sum().item())

    denom = max(total_samples, 1)
    val_loss = total_loss / denom
    mean_l2 = sum_l2 / denom
    mean_l1 = sum_l1 / denom
    mse_x = sum_dx2 / denom
    mse_y = sum_dy2 / denom
    rmse_x = math.sqrt(mse_x)
    rmse_y = math.sqrt(mse_y)
    rmse_2d = math.sqrt(mse_x + mse_y)

    return val_loss, {
        "mean_l2": mean_l2,
        "mean_l1": mean_l1,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_2d": rmse_2d,
    }


# ============================================================
# 2) loss 曲线保存
# ============================================================

def plot_and_save_losses(train_losses, val_losses, out_dir: Path, suffix: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"losses_mag_imu_eqnio{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for idx, train_loss in enumerate(train_losses, start=1):
            val_loss = val_losses[idx - 1] if idx - 1 < len(val_losses) else float("nan")
            writer.writerow([idx, train_loss, val_loss])

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mag+IMU EqNIO BaselineA Training")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = out_dir / f"loss_curve_mag_imu_eqnio{suffix}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {fig_path}")
    print(f"Saved loss csv to {csv_path}")

# ============================================================
# 3) canonical consistency loss 的分段退火策略
# ============================================================

def get_lambda_can(epoch: int):
    """
    canonical consistency loss 分段退火策略
    """
    if epoch < 30:
        return 0.1
    elif epoch < 80:
        return 0.05
    else:
        return 0.02

# ============================================================
# 4) 主函数
# ============================================================

def main():
    # --------- data path（按你实际目录改） ---------
    # train_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-all-feature-5-v3" / "train")
    # val_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-all-feature-5-v3" / "eval")

    # #  --------- data path（按你实际目录改） ---------
    # train_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-trans-all-feature-5" / "train")
    # val_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-xinxi-resample-zscore-trans-all-feature-5" / "eval")

    
    train_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-filter-zscore-all-feature-5" / "train")
    val_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-filter-zscore-all-feature-5" / "eval")

    # train_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-zscore-trans-all-feature-5" / "train")
    # val_dir = str(Path("data") / "data_for_train_test_v1" / "12.25-wenguan-resample-zscore-trans-all-feature-5" / "eval")
    # test_dir = str(Path("data") / "test")  # 如果你也需要 test，可以照 val 再建一个 loader

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # --------- 超参（按需调整） ---------
    batch_size = 32
    lr = 5e-4
    epochs = 400
    weight_decay = 1e-4
    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    seq_len = 128     
    stride = 10

    # canonical consistency
    ablation_mode = "full"
    if ablation_mode == "full":
        use_can_loss = True
        use_aug_task_loss = True
    elif ablation_mode == "no_consistency":
        use_can_loss = False
        use_aug_task_loss = True
    elif ablation_mode == "no_aug_task":
        use_can_loss = True
        use_aug_task_loss = False
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")
    
    # y 归一化模式（与dataset norm_y对齐）
    y_norm_mode = "per_file_minmax"

    feature_transform = ComposeTransform([
            DefaultTransform(),
            YawAugmentO2Transform(p_reflect=0.0),
    ])
    # --------- dataloaders ---------
    train_loader = create_magnetic_imu_dataset_dataloader(
        train_dir,
        batch_size=batch_size,
        pattern=".csv",
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=pin_memory,
        transform=feature_transform,
        stats=None,                   # per_file_minmax 不需要全局 stats
        seq_len=seq_len,
        stride=stride,
        normalize_mag=False,
        normalize_imu=False,
        y_norm_mode=y_norm_mode,
        cache_in_memory=True,
        gravity_align=True,
        use_linear_acc=False,
    )

    val_loader = create_magnetic_imu_dataset_dataloader(
        val_dir,
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
        gravity_align=True,
        use_linear_acc=False,
    )

    assert train_loader is not None, "train_loader is None，请检查 train_dir"
    assert val_loader is not None, "val_loader is None，请检查 val_dir"

    # dataset stats（用于 denorm_y 的 global 模式；per_file_minmax 一般用不到，但保持接口一致）
    val_stats = getattr(val_loader.dataset, "stats", None)


    # --------- 构建融合模型（BaselineA） ---------
    model = MagImuEqNioFusionModelV1(
        mag_input_dim=3,
        mag_d_model=128,
        seq_len=seq_len,
        use_frame_net=True,
        canonicalize_mag=True,
        canonicalize_imu=True,
        
        frame_hidden=64,
        frame_depth=3,
        imu_c=64,
        imu_blocks=4,
        imu_out_dim=32,
        head_hidden=256,
        out_dim=2,
        dropout=0.1
    ).to(device)

    # --------- loss / optim / scheduler ---------
    criterion = WeightedSmoothL1(beta=0.05, w_x=1.3, w_y=1.0).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --------- log & ckpt ---------
    run_dir = Path("runs") / "loss_mag_imu_eqnio"
    checkpoints_dir = Path("checkpoints") / "mag_imu_eqnio"
    date_suffix = datetime.now().strftime("_%Y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / f"mag_imu_eqnio_best{date_suffix}_xinxi.pt"

    best_val = float("inf")
    train_losses, val_losses = [], []

    print("----------- Training Mag+IMU EqNIO V1 -----------")
    for epoch in range(1, epochs + 1):
        start = time.time()
        # 分段退火
        lambda_can = 0.1
        if ablation_mode == "full":
            lambda_can = get_lambda_can(epoch)
        elif ablation_mode == "no_consistency":
            lambda_can = 0.0
        elif ablation_mode == "no_aug_task":
            lambda_can = get_lambda_can(epoch)

        train_loss, train_parts = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lambda_can=lambda_can,
            use_aug_task_loss=use_aug_task_loss,
            use_can_loss = use_can_loss,
            use_mag_can_loss = True,
            use_imu_can_loss = True,
            y_norm_mode=y_norm_mode,
            stats=None,
            grad_clip=1.0,
            use_amp=True,
        )

        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            y_norm_mode=y_norm_mode,
            stats=val_stats,
        )

        elapsed = time.time() - start
        cur_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} (task={train_parts['loss_task']:.6f} "
            f"task_aug={train_parts['loss_task_aug']:.6f} can={train_parts['loss_can']:.6f}) | "
            f"val_loss={val_loss:.6f} | "
            f"mean_l2={val_metrics['mean_l2']:.3f} "
            f"rmse_x={val_metrics['rmse_x']:.3f} rmse_y={val_metrics['rmse_y']:.3f} "
            f"rmse_2d={val_metrics['rmse_2d']:.3f} | "
            f"lr={cur_lr:.2e} ({elapsed:.1f}s)"
        )

        # 用 rmse_2d 选 best
        score = val_metrics["rmse_2d"]
        if score < best_val:
            best_val = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "rmse_2d": score,
                },
                best_path,
            )
            print(f"  Saved best checkpoint -> {best_path}")

        scheduler.step()

    plot_and_save_losses(train_losses, val_losses, run_dir, suffix=date_suffix)
    print("Training finished.")


if __name__ == "__main__":
    main()
