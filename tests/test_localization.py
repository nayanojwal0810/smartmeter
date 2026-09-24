"""Tests for Reference CamAL Localization Utilities (Experiment 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.localization import (
    normalize_cam,
    average_ensemble_cam,
    compute_x_attn_kw,
    apply_attention_sigmoid,
    threshold_activation,
    localize_ensemble_window,
    stitch_segment_timeline,
    extract_events_from_timeline,
    match_events_deterministic,
)
from src.models.resnet import ResNet1D


def test_cam_normalization_standard() -> None:
    """Verify CAM normalization divides by temporal maximum."""
    cam = np.array([1.0, 2.0, 4.0, 0.0, -1.0], dtype=np.float32)
    norm = normalize_cam(cam)
    assert norm.shape == (5,)
    assert norm[2] == pytest.approx(1.0)
    assert norm[1] == pytest.approx(0.5)
    assert norm[0] == pytest.approx(0.25)
    assert norm[3] == pytest.approx(0.0)
    assert norm[4] == pytest.approx(-0.25)


def test_cam_normalization_zero_or_negative_max() -> None:
    """Verify max_t CAM(t) <= 0 yields all zeros without division error."""
    all_zeros = np.zeros(510, dtype=np.float32)
    norm_zeros = normalize_cam(all_zeros)
    assert np.all(norm_zeros == 0.0)
    assert norm_zeros.shape == (510,)

    all_neg = np.array([-5.0, -2.0, -10.0, -0.1], dtype=np.float32)
    norm_neg = normalize_cam(all_neg)
    assert np.all(norm_neg == 0.0)
    assert norm_neg.shape == (4,)


def test_ensemble_cam_averaging() -> None:
    """Verify normalized CAMs are averaged element-wise across models."""
    cam1 = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    cam2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    cam3 = np.array([0.0, 0.2, 1.0], dtype=np.float32)

    avg = average_ensemble_cam([cam1, cam2, cam3])
    assert avg.shape == (3,)
    assert avg[0] == pytest.approx(0.5)
    assert avg[1] == pytest.approx(0.4)
    assert avg[2] == pytest.approx(0.5)


def test_distinct_input_representations_and_nonnegativity() -> None:
    """Verify x_model and x_attn are distinct and x_attn is strictly nonnegative."""
    x_raw_watts = np.array([0.0, 100.0, 523.94, 2500.0], dtype=np.float32)
    mu_train = 523.94
    sigma_train = 764.91

    # 1. Neural-network model input (standardized z-score, can be negative)
    x_model = (x_raw_watts - mu_train) / sigma_train
    assert x_model[0] < 0.0  # -0.685
    assert x_model[1] < 0.0  # -0.554
    assert x_model[2] == pytest.approx(0.0, abs=1e-3)
    assert x_model[3] > 0.0  # +2.583

    # 2. CAM attention input (kW scaling, strictly nonnegative)
    x_attn = compute_x_attn_kw(x_raw_watts)
    assert np.all(x_attn >= 0.0)
    assert x_attn[0] == pytest.approx(0.0)
    assert x_attn[1] == pytest.approx(0.1)
    assert x_attn[2] == pytest.approx(0.52394, abs=1e-4)
    assert x_attn[3] == pytest.approx(2.5)

    # They are distinct representations with different scales and ranges
    assert not np.allclose(x_model, x_attn)


def test_negative_cam_with_positive_x_attn_cannot_produce_positive_activation() -> None:
    """Verify negative CAM * nonnegative x_attn results in product <= 0 and S(t) <= 0.50."""
    negative_cam = np.array([-1.0, -0.5, -0.1, 0.0], dtype=np.float32)
    positive_x_attn = np.array([2.5, 1.0, 0.5, 3.0], dtype=np.float32)  # kW

    s_t = apply_attention_sigmoid(negative_cam, positive_x_attn)
    # product is negative or zero -> sigmoid(<= 0) <= 0.50
    assert np.all(s_t <= 0.50)
    assert s_t[0] == pytest.approx(1.0 / (1.0 + np.exp(2.5)), abs=1e-5)  # ~0.0758
    assert s_t[1] == pytest.approx(1.0 / (1.0 + np.exp(0.5)), abs=1e-5)  # ~0.3775
    assert s_t[3] == pytest.approx(0.50)  # sigmoid(0) = 0.5

    # Thresholding: S(t) < 0.50 strictly yields 0
    y_hat = threshold_activation(s_t[:3], threshold=0.50)
    assert np.all(y_hat == 0)


def test_attention_sigmoid_transformation_math() -> None:
    """Verify sigmoid(CAM_ens * x_attn) on kW-scaled inputs."""
    cam_ens = np.array([1.0, 0.0, 0.8, -0.5], dtype=np.float32)
    x_attn = np.array([2.0, 1.5, 2.5, 1.0], dtype=np.float32)  # kW

    s_t = apply_attention_sigmoid(cam_ens, x_attn)
    assert s_t.shape == (4,)
    # index 0: 1.0 * 2.0 = 2.0 -> sigmoid(2.0) ~ 0.880797
    assert s_t[0] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)), abs=1e-5)
    # index 1: 0.0 * 1.5 = 0.0 -> sigmoid(0.0) = 0.5
    assert s_t[1] == pytest.approx(0.5, abs=1e-5)
    # index 2: 0.8 * 2.5 = 2.0 -> sigmoid(2.0) ~ 0.880797
    assert s_t[2] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)), abs=1e-5)
    # index 3: -0.5 * 1.0 = -0.5 -> sigmoid(-0.5) ~ 0.37754
    assert s_t[3] == pytest.approx(1.0 / (1.0 + np.exp(0.5)), abs=1e-5)


def test_binary_thresholding() -> None:
    """Verify binary activation at threshold 0.50."""
    s_t = np.array([0.499, 0.500, 0.501, 0.880, 0.120], dtype=np.float32)
    y_hat = threshold_activation(s_t, threshold=0.50)
    assert np.array_equal(y_hat, [0, 1, 1, 1, 0])


def test_zero_power_and_zero_cam_edge_cases() -> None:
    """Verify zero-power suppression and zero-CAM threshold behavior:
    1. X = 0 with positive CAM -> prediction 0
    2. X = 0 with negative CAM -> prediction 0
    3. X > 0 with zero CAM -> sigmoid(0) = 0.50 and y_hat = 1 (preserve threshold behavior)
    4. X > 0 with positive CAM -> S(t) > 0.50 -> prediction 1
    5. X > 0 with negative CAM -> S(t) < 0.50 -> prediction 0
    """
    cams = np.array([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32)
    x_raw = np.array([0.0, 0.0, 2000.0, 2000.0, 2000.0], dtype=np.float32)  # Watts
    x_attn = compute_x_attn_kw(x_raw)  # [0.0, 0.0, 2.0, 2.0, 2.0] kW

    s_t = apply_attention_sigmoid(cams, x_attn)

    # Verify continuous values S(t)
    assert s_t[0] == pytest.approx(0.50)  # CAM=+1.0, X=0 -> sigmoid(0)=0.50
    assert s_t[1] == pytest.approx(0.50)  # CAM=-1.0, X=0 -> sigmoid(0)=0.50
    assert s_t[2] == pytest.approx(0.50)  # CAM=0.0, X=2kW -> sigmoid(0)=0.50
    assert s_t[3] > 0.50                 # CAM=+1.0, X=2kW -> sigmoid(2.0) ~ 0.8808
    assert s_t[4] < 0.50                 # CAM=-1.0, X=2kW -> sigmoid(-2.0) ~ 0.1192

    y_hat = threshold_activation(s_t, x_attn_kw=x_attn, threshold=0.50)

    # 1. X=0 with positive CAM -> prediction 0
    assert y_hat[0] == 0

    # 2. X=0 with negative CAM -> prediction 0
    assert y_hat[1] == 0

    # 3. X>0 with zero CAM -> sigmoid(0)=0.50 and preserve documented threshold behavior (y_hat=1)
    assert y_hat[2] == 1

    # 4. X>0 with positive CAM -> prediction 1
    assert y_hat[3] == 1

    # 5. X>0 with negative CAM -> prediction 0
    assert y_hat[4] == 0


def test_510_point_alignment_and_feature_map_length() -> None:
    """Verify ResNet feature maps and CAM extraction preserve exactly 510 points."""
    model = ResNet1D(in_channels=1, filter_counts=(64, 128, 128), kernel_sizes=(5, 5, 3))
    x = torch.randn(2, 510)

    # Feature map shape
    features = model.forward_features(x)
    assert features.shape == (2, 128, 510)

    # CAM extraction shape
    cam = model.extract_cam(x)
    assert cam.shape == (2, 510)


def test_detection_gating_and_undetected_window_all_zeros() -> None:
    """Verify detection gate: P_ens < 0.50 outputs strictly all zeros."""
    class NegativeLogitModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[-10.0]])

        def extract_cam(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[0], 510)

    models = [NegativeLogitModel(), NegativeLogitModel()]
    x_model = np.random.randn(510).astype(np.float32)
    x_attn = np.ones(510, dtype=np.float32)

    result = localize_ensemble_window(
        models=models,
        x_model_standardized=x_model,
        x_attn_kw=x_attn,
        detection_threshold=0.50,
    )

    assert result["detected"] is False
    assert result["p_ensemble"] < 0.01
    assert result["y_hat"].shape == (510,)
    assert np.all(result["y_hat"] == 0)
    assert np.all(result["s_continuous"] == 0.0)


def test_detection_gating_detected_window_activation() -> None:
    """Verify detected window (P_ens >= 0.50) runs attention-sigmoid with x_attn_kw."""
    class PositiveLogitModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[5.0]])  # p ~ 0.993

        def extract_cam(self, x: torch.Tensor) -> torch.Tensor:
            # Baseline CAM is -1.0 on background, +10.0 in points 200..250
            cam = torch.full((x.shape[0], 510), -1.0)
            cam[:, 200:250] = 10.0
            return cam

    models = [PositiveLogitModel()]
    # Input x_model is standardized z-score
    x_model = np.full(510, -0.5, dtype=np.float32)
    x_model[200:250] = 3.0

    # Input x_attn is nonnegative power in kW (e.g. 0.1 kW background, 2.5 kW boil)
    x_attn = np.full(510, 0.1, dtype=np.float32)
    x_attn[200:250] = 2.5

    result = localize_ensemble_window(
        models=models,
        x_model_standardized=x_model,
        x_attn_kw=x_attn,
        detection_threshold=0.50,
        loc_threshold=0.50,
    )

    assert result["detected"] is True
    assert result["p_ensemble"] > 0.90
    assert result["y_hat"].shape == (510,)
    assert np.all(result["y_hat"][200:250] == 1)
    assert np.all(result["y_hat"][0:100] == 0)


def test_segment_bounded_timeline_stitching() -> None:
    """Verify stitching non-overlapping windows into a continuous segment."""
    segment_len = 1200
    w1_yhat = np.zeros(510, dtype=np.int64)
    w1_yhat[10:30] = 1
    w2_yhat = np.zeros(510, dtype=np.int64)
    w2_yhat[50:80] = 1

    window_records = [
        {"segment_offset": 0, "y_hat": w1_yhat},
        {"segment_offset": 510, "y_hat": w2_yhat},
    ]

    timeline = stitch_segment_timeline(segment_len, window_records)
    assert len(timeline) == 1200
    # First window region
    assert np.all(timeline[10:30] == 1)
    assert np.all(timeline[0:10] == 0)
    # Second window region (offset 510)
    assert np.all(timeline[510 + 50 : 510 + 80] == 1)
    assert np.all(timeline[510 : 510 + 50] == 0)
    # Leftover unwindowed tail (points 1020..1199)
    assert np.all(timeline[1020:1200] == 0)


def test_deterministic_event_matching_iou() -> None:
    """Verify deterministic 1-to-1 event matching."""
    gt_events = [(10, 20), (50, 70), (100, 120)]
    pred_events = [
        (11, 21),   # Overlaps gt[0] (IoU = 10/12 ~ 0.833) -> Match TP
        (52, 68),   # Overlaps gt[1] (IoU = 17/21 ~ 0.810) -> Match TP
        (200, 220), # False Positive
    ]
    # gt[2] is False Negative

    res = match_events_deterministic(gt_events, pred_events, iou_threshold=0.50)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["precision"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert res["recall"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert res["f1"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert len(res["matches"]) == 2


def test_e3_multi_scale_resnet_kernel_shapes() -> None:
    """Verify all E3 multi-scale ResNet models {k in 5,7,9,15,25} produce [B, 1] logit and [B, 510] CAM."""
    batch_size = 4
    x_dummy = torch.randn(batch_size, 510)
    kernel_scales = [5, 7, 9, 15, 25]

    for k in kernel_scales:
        model = ResNet1D(
            in_channels=1,
            filter_counts=(64, 128, 128),
            kernel_sizes=(k, 5, 3),
            out_features=1,
        )
        model.eval()
        with torch.no_grad():
            logit = model(x_dummy)
            assert logit.shape == (batch_size, 1), f"Expected logit shape ({batch_size}, 1), got {logit.shape} for k={k}"

            cam = model.extract_cam(x_dummy)
            assert cam.shape == (batch_size, 510), f"Expected CAM shape ({batch_size}, 510), got {cam.shape} for k={k}"
