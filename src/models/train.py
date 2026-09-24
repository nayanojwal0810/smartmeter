"""Training and Validation Pipeline for Simple 1D CNN Baseline (Decision D-008).

Implements:
1. Deterministic seeding and backend setup.
2. Train-only global normalization statistics computation.
3. Strong-target delayed validation evaluation at fixed threshold (0.5).
4. Trivial majority-class reference baseline evaluation.
5. Strict isolation and exclusion of test split (H2, H13).
6. Artifact persistence for metrics, curves, and model checkpoints.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.data.dataset import KettleWeakDataset, NormalizationTransform
from src.data.split import HouseholdSplitManager
from src.models.cnn_baseline import Simple1DCNN
from src.models.resnet import ResNet1D

logger = logging.getLogger(__name__)


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate model architecture from configuration dictionary."""
    name = model_cfg.get("name", "simple_1d_cnn").lower()

    if name in {"resnet1d", "resnet1d_reference", "resnet"}:
        return ResNet1D(
            in_channels=int(model_cfg.get("in_channels", 1)),
            filter_counts=tuple(int(x) for x in model_cfg.get("filter_counts", [64, 128, 128])),
            kernel_sizes=tuple(int(x) for x in model_cfg.get("kernel_sizes", [8, 5, 3])),
            out_features=int(model_cfg.get("out_features", 1)),
        )
    elif name in {"simple_1d_cnn", "cnn_baseline", "cnn"}:
        return Simple1DCNN(
            in_channels=int(model_cfg.get("in_channels", 1)),
            conv1_channels=int(model_cfg.get("conv1_channels", 32)),
            conv1_kernel=int(model_cfg.get("conv1_kernel", 9)),
            pool_kernel=int(model_cfg.get("pool_kernel", 2)),
            conv2_channels=int(model_cfg.get("conv2_channels", 64)),
            conv2_kernel=int(model_cfg.get("conv2_kernel", 9)),
            out_features=int(model_cfg.get("out_features", 1)),
        )
    else:
        raise ValueError(f"Unknown model name '{name}'. Supported: 'simple_1d_cnn', 'resnet1d_reference'")


def set_seed(seed: int = 42) -> None:
    """Configure deterministic random seed across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_train_normalization_stats(
    train_dataset: Dataset,
    batch_size: int = 2048,
) -> Tuple[float, float]:
    """Compute global mean and standard deviation exclusively on the training dataset.

    Uses streaming two-pass accumulation to avoid loading all windows into RAM.
    Guarantees that validation, test, and strong target data are NEVER touched.

    Args:
        train_dataset: PyTorch Dataset or KettleWeakDataset for the TRAIN split.
        batch_size: Chunk size for streaming aggregation.

    Returns:
        Tuple of (train_mean, train_std).
    """
    total_points = 0
    sum_val = 0.0

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for x_batch, _ in loader:
        if isinstance(x_batch, torch.Tensor):
            x_arr = x_batch.detach().cpu().numpy()
        else:
            x_arr = np.asarray(x_batch)
        sum_val += float(np.sum(x_arr))
        total_points += x_arr.size

    if total_points == 0:
        return 0.0, 1.0

    mean = sum_val / total_points

    sum_sq_diff = 0.0
    for x_batch, _ in loader:
        if isinstance(x_batch, torch.Tensor):
            x_arr = x_batch.detach().cpu().numpy()
        else:
            x_arr = np.asarray(x_batch)
        sum_sq_diff += float(np.sum((x_arr - mean) ** 2))

    variance = sum_sq_diff / total_points
    std = float(np.sqrt(variance))
    if std < 1e-8:
        std = 1.0

    return mean, std


def load_validation_strong_targets(
    targets_jsonl_path: Path,
    val_window_ids: Sequence[int],
) -> Dict[int, int]:
    """Stream evaluation targets manifest and build window_id -> active_target mapping.

    Args:
        targets_jsonl_path: Path to kettle_evaluation_targets.jsonl.
        val_window_ids: Set/sequence of window_ids belonging to validation split.

    Returns:
        Dictionary mapping window_id -> 1 (active kettle event) or 0 (inactive).
    """
    val_set = set(val_window_ids)
    targets_map: Dict[int, int] = {}

    if not Path(targets_jsonl_path).exists():
        logger.warning("Targets manifest %s not found. Returning empty map.", targets_jsonl_path)
        return targets_map

    with open(targets_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            w_id = rec.get("window_id")
            if w_id in val_set:
                has_active = 1 if rec.get("has_active_target", False) else 0
                targets_map[w_id] = has_active

    return targets_map


def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """Compute precision, recall, F1, balanced accuracy, and confusion counts."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_pred, dtype=np.int32)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    total = len(y_t)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float((tpr + tnr) / 2.0)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "accuracy": round(accuracy, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_samples": total,
    }


def evaluate_trivial_majority_baseline(
    targets_map: Dict[int, int],
) -> Dict[str, Any]:
    """Compute non-ML majority-class reference baseline on validation strong targets.

    Predicts the majority class across all validation strong-target labels.
    """
    if not targets_map:
        return {
            "name": "majority_class_reference",
            "majority_class": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.5,
            "accuracy": 0.0,
            "total_samples": 0,
        }

    y_true = list(targets_map.values())
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    majority_class = 1 if pos_count > neg_count else 0

    y_pred = [majority_class] * len(y_true)
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    metrics["name"] = "majority_class_reference"
    metrics["majority_class"] = majority_class
    metrics["positive_target_count"] = pos_count
    metrics["negative_target_count"] = neg_count

    return metrics


def evaluate_validation_strong_targets(
    model: nn.Module,
    val_loader: DataLoader,
    val_dataset: KettleWeakDataset,
    targets_map: Dict[int, int],
    fixed_threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Evaluate model on validation split against delayed strong Kettle targets.

    Guarantees:
    - Fixed threshold (0.5 by default) is strictly NOT tuned on the validation set.
    - Strong targets remain evaluation-only evidence.
    """
    model.eval()
    val_loss_total = 0.0
    val_loss_batches = 0

    all_window_ids: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    # Map dataset index to window_id
    with torch.no_grad():
        for batch_idx, (x_batch, y_weak_batch) in enumerate(val_loader):
            x_tensor = x_batch.to(device)
            y_weak_tensor = y_weak_batch.to(device).float().unsqueeze(1)

            logits = model(x_tensor)
            if criterion is not None:
                loss = criterion(logits, y_weak_tensor)
                val_loss_total += float(loss.item())
                val_loss_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= fixed_threshold).astype(int).tolist()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds)

    # Reconstruct window_ids from validation dataset metadata
    val_window_ids = [val_dataset.get_metadata(i)["window_id"] for i in range(len(val_dataset))]

    y_true_strong: List[int] = []
    y_pred_strong: List[int] = []

    for w_id, pred in zip(val_window_ids, all_preds):
        if w_id in targets_map:
            y_true_strong.append(targets_map[w_id])
            y_pred_strong.append(pred)

    strong_metrics = compute_binary_classification_metrics(y_true_strong, y_pred_strong)
    avg_val_loss = (val_loss_total / val_loss_batches) if val_loss_batches > 0 else 0.0

    return {
        "val_weak_loss": round(avg_val_loss, 6),
        "val_strong_precision": strong_metrics["precision"],
        "val_strong_recall": strong_metrics["recall"],
        "val_strong_f1": strong_metrics["f1"],
        "val_strong_balanced_accuracy": strong_metrics["balanced_accuracy"],
        "val_strong_accuracy": strong_metrics["accuracy"],
        "val_strong_tp": strong_metrics["tp"],
        "val_strong_fp": strong_metrics["fp"],
        "val_strong_tn": strong_metrics["tn"],
        "val_strong_fn": strong_metrics["fn"],
        "fixed_threshold": fixed_threshold,
        "evaluated_target_windows": len(y_true_strong),
    }


def run_training_loop(
    config: Dict[str, Any],
    train_dataset: KettleWeakDataset,
    val_dataset: KettleWeakDataset,
    targets_map: Dict[int, int],
    output_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Execute complete 1D CNN baseline training loop with early stopping."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalization: compute stats exclusively on TRAIN
    logger.info("Computing train-only normalization statistics...")
    train_mean, train_std = compute_train_normalization_stats(train_dataset)
    logger.info("Train Normalization: Mean = %.4f W, Std = %.4f W", train_mean, train_std)

    norm_transform = NormalizationTransform(mean=train_mean, std=train_std, method="standardize")
    train_dataset.transform = norm_transform
    val_dataset.transform = norm_transform

    # Save normalization statistics artifact
    norm_stats = {
        "mean": train_mean,
        "std": train_std,
        "method": "standardize",
        "fitted_on_split": "train",
        "train_samples_count": len(train_dataset),
        "unit": "Watts",
    }
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    # 2. Trivial Baseline Evaluation
    trivial_metrics = evaluate_trivial_majority_baseline(targets_map)
    with open(output_dir / "trivial_baseline.json", "w", encoding="utf-8") as f:
        json.dump(trivial_metrics, f, indent=2)
    logger.info("Trivial Majority Baseline: F1 = %.4f, BalAcc = %.4f", trivial_metrics["f1"], trivial_metrics["balanced_accuracy"])

    # 3. DataLoaders
    batch_size = int(train_cfg.get("batch_size", 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4. Model, Optimizer, Loss
    model = build_model(model_cfg).to(device)

    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    max_epochs = int(train_cfg.get("max_epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    fixed_threshold = float(train_cfg.get("fixed_threshold", 0.5))

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history: List[Dict[str, Any]] = []

    logger.info("Starting CNN Baseline Training (max %d epochs, patience %d)...", max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for x_batch, y_weak_batch in train_loader:
            x_tensor = x_batch.to(device)
            y_weak_tensor = y_weak_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_weak_tensor)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item())
            train_batches += 1

        avg_train_loss = train_loss_total / train_batches if train_batches > 0 else 0.0

        # Validation evaluation
        val_eval = evaluate_validation_strong_targets(
            model=model,
            val_loader=val_loader,
            val_dataset=val_dataset,
            targets_map=targets_map,
            fixed_threshold=fixed_threshold,
            device=device,
            criterion=criterion,
        )

        val_f1 = val_eval["val_strong_f1"]
        val_bal_acc = val_eval["val_strong_balanced_accuracy"]
        val_prec = val_eval["val_strong_precision"]
        val_rec = val_eval["val_strong_recall"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_weak_loss": val_eval["val_weak_loss"],
            "val_strong_precision": val_prec,
            "val_strong_recall": val_rec,
            "val_strong_f1": val_f1,
            "val_strong_balanced_accuracy": val_bal_acc,
            "val_strong_accuracy": val_eval["val_strong_accuracy"],
            "val_strong_tp": val_eval["val_strong_tp"],
            "val_strong_fp": val_eval["val_strong_fp"],
            "val_strong_tn": val_eval["val_strong_tn"],
            "val_strong_fn": val_eval["val_strong_fn"],
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.5f | Val Weak Loss: %.5f | Val Strong Prec: %.4f | Rec: %.4f | F1: %.4f | BalAcc: %.4f",
            epoch,
            max_epochs,
            avg_train_loss,
            val_eval["val_weak_loss"],
            val_prec,
            val_rec,
            val_f1,
            val_bal_acc,
        )

        # Early stopping and checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "config": config,
                    "normalization": norm_stats,
                },
                output_dir / "model_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d (patience %d).", epoch, patience)
                break

    # Save final model
    torch.save(
        {
            "epoch": len(history),
            "model_state_dict": model.state_dict(),
            "config": config,
            "normalization": norm_stats,
        },
        output_dir / "model_final.pt",
    )

    # Save training curves & history
    with open(output_dir / "training_curves.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Best metrics summary
    best_record = next((r for r in history if r["epoch"] == best_epoch), history[-1])
    final_metrics = {
        "experiment_name": model_cfg.get("name", "simple_1d_cnn"),
        "decision_id": train_cfg.get("decision_id", "D-008"),
        "appliance": train_cfg.get("appliance", "kettle"),
        "seed": seed,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history),
        "early_stopping_triggered": (patience_counter >= patience),
        "train_mean_w": train_mean,
        "train_std_w": train_std,
        "best_epoch_metrics": best_record,
        "trivial_baseline_reference": trivial_metrics,
        "checkpoint_paths": {
            "best_model": str(output_dir / "model_best.pt"),
            "final_model": str(output_dir / "model_final.pt"),
        },
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Save config copy
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return final_metrics
