"""Reference CamAL Localization Utilities for Experiment 3.

Implements the locked reference localization pipeline:
1. Normalization: CAM_norm(t) = CAM(t) / max_t CAM(t), with all-zeros if max_t CAM(t) <= 0.
2. Ensemble CAM averaging across scales: CAM_ens(t) = mean(CAM_norm^(m)(t)).
3. Attention-Sigmoid transformation on standardized model inputs: S(t) = sigmoid(CAM_ens(t) * x_attn(t)).
4. Binary conversion with deterministic zero-power suppression:
   - X(t) == 0 -> y_hat(t) = 0
   - For X(t) > 0: y_hat(t) = 1 if S(t) >= 0.50 else 0
5. Ensemble detection gating: If P_ens < 0.50, window outputs all zeros.
6. Segment-bounded timeline stitching: non-overlapping placement without crossing forbidden boundaries.
7. Deterministic 1-to-1 event matching at Temporal IoU >= 0.50.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


def normalize_cam(cam: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize raw 1D CAM using reference division by temporal maximum.

    Args:
        cam: Raw CAM 1D array or 2D tensor of shape (510,) or (B, 510).

    Returns:
        Normalized CAM array of same shape.
        If max_t CAM(t) <= 0, returns all-zero array of same shape.
    """
    if isinstance(cam, torch.Tensor):
        cam_arr = cam.detach().cpu().numpy()
    else:
        cam_arr = np.asarray(cam, dtype=np.float32)

    if cam_arr.ndim == 1:
        max_val = float(np.max(cam_arr))
        if max_val <= 0.0 or np.isnan(max_val):
            return np.zeros_like(cam_arr, dtype=np.float32)
        return (cam_arr / max_val).astype(np.float32)

    # 2D batch case (B, T)
    out = np.zeros_like(cam_arr, dtype=np.float32)
    for b in range(cam_arr.shape[0]):
        max_val = float(np.max(cam_arr[b]))
        if max_val > 0.0 and not np.isnan(max_val):
            out[b] = cam_arr[b] / max_val
    return out


def average_ensemble_cam(normalized_cams: Sequence[np.ndarray]) -> np.ndarray:
    """Average normalized CAMs across ensemble branches.

    Args:
        normalized_cams: Sequence of normalized CAM arrays, each of shape (510,) or (B, 510).

    Returns:
        Ensemble CAM array of shape (510,) or (B, 510).
    """
    if not normalized_cams:
        raise ValueError("Cannot average empty sequence of CAMs")
    stack = np.stack(normalized_cams, axis=0)  # (M, ...)
    return np.mean(stack, axis=0, dtype=np.float32)


def compute_x_attn_kw(x_raw_watts: np.ndarray | torch.Tensor) -> np.ndarray:
    """Compute nonnegative aggregate power representation in kW for CAM attention.

    Formula: x_attn(t) = X(t) / 1000.0

    Args:
        x_raw_watts: Raw aggregate power in Watts (nonnegative).

    Returns:
        Nonnegative power array in Kilowatts.
    """
    if isinstance(x_raw_watts, torch.Tensor):
        x_arr = x_raw_watts.detach().cpu().numpy()
    else:
        x_arr = np.asarray(x_raw_watts, dtype=np.float32)
    return np.maximum(0.0, x_arr / 1000.0, dtype=np.float32)


def apply_attention_sigmoid(cam_ens: np.ndarray, x_attn: np.ndarray) -> np.ndarray:
    """Apply reference attention multiplication and sigmoid transformation.

    Formula: S(t) = sigmoid(CAM_ens(t) * x_attn(t))

    Where x_attn(t) is the nonnegative aggregate power scaled by 1/1000 (kW).

    Args:
        cam_ens: Ensemble CAM array of shape (510,) or (B, 510).
        x_attn: Nonnegative aggregate power array in kW of shape (510,) or (B, 510).

    Returns:
        Continuous activation signal S(t) in [0, 1].
    """
    cam_arr = np.asarray(cam_ens, dtype=np.float32)
    x_arr = np.asarray(x_attn, dtype=np.float32)
    product = cam_arr * x_arr
    # Numerically stable sigmoid: 1 / (1 + exp(-product))
    return (1.0 / (1.0 + np.exp(-product))).astype(np.float32)


def threshold_activation(
    s_t: np.ndarray,
    x_attn_kw: Optional[np.ndarray] = None,
    threshold: float = 0.50,
) -> np.ndarray:
    """Threshold continuous activation signal into binary states with zero-power suppression.

    Formula:
        If X(t) == 0 (x_attn_kw <= 0): y_hat(t) = 0
        For X(t) > 0: y_hat(t) = 1 if S(t) >= threshold else 0

    Args:
        s_t: Continuous activation signal array.
        x_attn_kw: Optional power array (Watts or kW). If provided, timestamps where
                   x_attn_kw <= 0 are strictly forced to 0.
        threshold: Decision threshold (default: 0.50).

    Returns:
        Binary integer array with values in {0, 1}.
    """
    s_arr = np.asarray(s_t, dtype=np.float32)
    binary = (s_arr >= threshold).astype(np.int64)
    if x_attn_kw is not None:
        x_arr = np.asarray(x_attn_kw, dtype=np.float32)
        binary = np.where(x_arr > 0.0, binary, 0).astype(np.int64)
    return binary


def localize_ensemble_window(
    models: Sequence[nn.Module],
    x_model_standardized: torch.Tensor | np.ndarray,
    x_attn_kw: np.ndarray | torch.Tensor,
    detection_threshold: float = 0.50,
    loc_threshold: float = 0.50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run full reference CamAL localization pipeline on a single 510-point window.

    Maintains explicit separation between:
    - x_model_standardized: (X - mu_train) / sigma_train (used by neural networks for detection & CAM extraction)
    - x_attn_kw: X / 1000 (nonnegative kW representation used in attention sigmoid: S(t) = sigmoid(CAM_ens * x_attn))

    Sequence:
    1. Run each ResNet(k) on x_model_standardized -> get detection probability and class-1 CAM.
    2. P_ens = mean(P_m).
    3. If P_ens < detection_threshold: output all zeros (undetected).
    4. If P_ens >= detection_threshold:
       - Normalize each CAM: CAM_norm = CAM / max(CAM) (or 0 if max <= 0)
       - CAM_ens = mean(CAM_norm)
       - S(t) = sigmoid(CAM_ens * x_attn_kw)
       - y_hat(t) = 0 if X(t) == 0 else (1 if S(t) >= loc_threshold else 0)

    Args:
        models: Sequence of trained ResNet models.
        x_model_standardized: Standardized input array or tensor of shape (510,) or (1, 510).
        x_attn_kw: Nonnegative power array in kW of shape (510,) or (1, 510).
        detection_threshold: Ensemble detection gate threshold (default 0.50).
        loc_threshold: Point-level activation threshold (default 0.50).
        device: Torch device (optional).

    Returns:
        Dictionary with keys:
            - 'detected': bool
            - 'p_ensemble': float
            - 'cam_ensemble': np.ndarray of shape (510,)
            - 's_continuous': np.ndarray of shape (510,)
            - 'y_hat': np.ndarray of shape (510,) int64
    """
    if isinstance(x_model_standardized, np.ndarray):
        x_np = x_model_standardized.squeeze()
        x_tensor = torch.from_numpy(x_np).float().unsqueeze(0)  # (1, 510)
    else:
        x_tensor = x_model_standardized
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

    if isinstance(x_attn_kw, torch.Tensor):
        x_attn_np = x_attn_kw.squeeze().detach().cpu().numpy()
    else:
        x_attn_np = np.asarray(x_attn_kw, dtype=np.float32).squeeze()


    if device is not None:
        x_tensor = x_tensor.to(device)

    probs = []
    norm_cams = []

    for model in models:
        model.eval()
        with torch.no_grad():
            if device is not None:
                model = model.to(device)
            # Logit & probability
            logit = model(x_tensor)
            p = float(torch.sigmoid(logit).squeeze().item())
            probs.append(p)

            # Raw CAM
            if hasattr(model, "extract_cam"):
                raw_cam = model.extract_cam(x_tensor).squeeze().cpu().numpy()
            else:
                # Fallback extraction from block3
                feat = model.block3(model.block2(model.block1(x_tensor.unsqueeze(1) if x_tensor.dim() == 2 else x_tensor)))
                w = model.classifier.weight[0]
                raw_cam = torch.einsum("bct,c->bt", feat, w).squeeze().cpu().numpy()

            norm_cam = normalize_cam(raw_cam)
            norm_cams.append(norm_cam)

    p_ens = float(np.mean(probs))
    cam_ens = average_ensemble_cam(norm_cams)

    if p_ens < detection_threshold:
        # Undetected window: all-zero output
        return {
            "detected": False,
            "p_ensemble": p_ens,
            "cam_ensemble": cam_ens,
            "s_continuous": np.zeros(510, dtype=np.float32),
            "y_hat": np.zeros(510, dtype=np.int64),
        }

    # Detected window: attention-sigmoid transformation with nonnegative x_attn (kW)
    s_cont = apply_attention_sigmoid(cam_ens, x_attn_np)
    y_hat = threshold_activation(s_cont, x_attn_kw=x_attn_np, threshold=loc_threshold)

    return {
        "detected": True,
        "p_ensemble": p_ens,
        "cam_ensemble": cam_ens,
        "s_continuous": s_cont,
        "y_hat": y_hat,
    }


def stitch_segment_timeline(
    segment_length: int,
    window_records: List[Dict[str, Any]],
) -> np.ndarray:
    """Stitch non-overlapping 510-point window binary states into a continuous segment timeline.

    Args:
        segment_length: Total number of 8s timestamp points in the recording segment.
        window_records: List of window dicts for this segment, each with 'segment_offset' (int) and 'y_hat' (510,).

    Returns:
        Binary timeline array of length segment_length with values in {0, 1}.
        Unwindowed points default to 0.
    """
    timeline = np.zeros(segment_length, dtype=np.int64)
    for rec in window_records:
        offset = rec["segment_offset"]
        y_hat = rec["y_hat"]
        w_len = len(y_hat)
        end_idx = min(offset + w_len, segment_length)
        valid_len = end_idx - offset
        if valid_len > 0:
            timeline[offset:end_idx] = y_hat[:valid_len]
    return timeline


def extract_events_from_timeline(
    binary_timeline: np.ndarray,
    max_gap_points: int = 2,  # 20s at 8s intervals: floor(20/8) = 2 points
    max_duration_points: int = 75,  # 600s at 8s intervals: 600/8 = 75 points
) -> List[Tuple[int, int]]:
    """Extract discrete continuous activation events [start_idx, end_idx] from binary timeline.

    Args:
        binary_timeline: 1D array of binary states in {0, 1}.
        max_gap_points: Max points allowed between active samples in same event (<= 20s -> 2 points).
        max_duration_points: Max event duration points (<= 600s -> 75 points).

    Returns:
        List of event tuples (start_idx, end_idx) inclusive.
    """
    events = []
    active_indices = np.where(binary_timeline > 0)[0]
    if len(active_indices) == 0:
        return events

    curr_start = active_indices[0]
    curr_last = active_indices[0]

    for idx in active_indices[1:]:
        if idx - curr_last <= (max_gap_points + 1):
            curr_last = idx
        else:
            dur = (curr_last - curr_start + 1)
            if dur <= max_duration_points:
                events.append((int(curr_start), int(curr_last)))
            curr_start = idx
            curr_last = idx

    dur = (curr_last - curr_start + 1)
    if dur <= max_duration_points:
        events.append((int(curr_start), int(curr_last)))

    return events


def match_events_deterministic(
    gt_events: List[Tuple[int, int]],
    pred_events: List[Tuple[int, int]],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Perform deterministic 1-to-1 event matching between ground-truth and predicted events.

    Matching procedure:
    1. Compute temporal IoU for all pairs (gt_i, pred_j).
    2. Filter candidate pairs with IoU >= iou_threshold.
    3. Sort candidate pairs in descending order of IoU.
    4. Greedily match pairs such that each GT event and each predicted event is matched at most once.
    5. Matched pairs are True Positives (TP). Unmatched pred events are False Positives (FP).
       Unmatched GT events are False Negatives (FN).

    Args:
        gt_events: List of (start, end) ground-truth event index tuples.
        pred_events: List of (start, end) predicted event index tuples.
        iou_threshold: Minimum temporal IoU for valid match (default 0.50).

    Returns:
        Dict with keys: 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'matches'.
    """
    if not gt_events and not pred_events:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}
    if not gt_events:
        return {"tp": 0, "fp": len(pred_events), "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}
    if not pred_events:
        return {"tp": 0, "fp": 0, "fn": len(gt_events), "precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []}

    candidate_pairs = []
    for g_idx, (g_s, g_e) in enumerate(gt_events):
        for p_idx, (p_s, p_e) in enumerate(pred_events):
            inter_start = max(g_s, p_s)
            inter_end = min(g_e, p_e)
            if inter_end >= inter_start:
                intersection = inter_end - inter_start + 1
                union = (g_e - g_s + 1) + (p_e - p_s + 1) - intersection
                iou = float(intersection / union) if union > 0 else 0.0
                if iou >= iou_threshold:
                    candidate_pairs.append((iou, g_idx, p_idx))

    # Sort descending by IoU
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, g_idx, p_idx in candidate_pairs:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            matches.append({"gt_idx": g_idx, "pred_idx": p_idx, "iou": round(iou, 4)})

    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matches": matches,
    }
