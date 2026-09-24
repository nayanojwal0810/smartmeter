"""Unit tests for streaming regular timebase resampling module."""

from __future__ import annotations

import pytest

from src.data.timebase import GridPoint, ResamplingConfig, StreamingTimebaseResampler


def test_exact_8s_observations() -> None:
    """Verify that exact 8-second observations pass through without imputation."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1008, "2014-01-01 00:00:08", 550.0, 0.0),
        (1016, "2014-01-01 00:00:16", 600.0, 0.0),
        (1024, "2014-01-01 00:00:24", 650.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 4
    assert [p.grid_unix for p in points] == [1000, 1008, 1016, 1024]
    assert [p.segment_id for p in points] == [1, 1, 1, 1]
    assert all(not p.is_imputed for p in points)
    assert [p.aggregate_w for p in points] == [500.0, 550.0, 600.0, 650.0]


def test_gap_le_16s_bridging() -> None:
    """Verify that gaps <= 16s are bridged with zero-order hold within the same segment."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    # 1000 -> 1016 is a 16-second gap (should produce grid point at 1008 with previous power 500.0)
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1016, "2014-01-01 00:00:16", 800.0, 2000.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 3
    assert [p.grid_unix for p in points] == [1000, 1008, 1016]
    assert [p.segment_id for p in points] == [1, 1, 1]
    assert points[0].is_imputed is False
    assert points[1].is_imputed is True  # Imputed at 1008
    assert points[1].aggregate_w == 500.0  # Zero-order hold from 1000
    assert points[2].is_imputed is False
    assert points[2].aggregate_w == 800.0


def test_gap_gt_16s_starts_new_segment() -> None:
    """Verify that gaps > 16s start a new contiguous segment without interpolation."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    # 1000 -> 1030 is a 30-second gap (> 16s)
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1008, "2014-01-01 00:00:08", 550.0, 0.0),
        (1038, "2014-01-01 00:00:38", 900.0, 0.0),  # Gap of 30s > 16s
        (1046, "2014-01-01 00:00:46", 950.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 4
    # Segment 1 points
    assert points[0].segment_id == 1
    assert points[0].grid_unix == 1000
    assert points[1].segment_id == 1
    assert points[1].grid_unix == 1008

    # Segment 2 points anchored at 1038
    assert points[2].segment_id == 2
    assert points[2].grid_unix == 1038
    assert points[3].segment_id == 2
    assert points[3].grid_unix == 1046


def test_irregular_sampling() -> None:
    """Verify irregular 6-8s sampling timestamps resample onto consistent 8s grid."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    # Irregular: 0, 7, 15, 22 (all steps <= 8s <= 16s)
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1007, "2014-01-01 00:00:07", 520.0, 0.0),
        (1015, "2014-01-01 00:00:15", 540.0, 0.0),
        (1024, "2014-01-01 00:00:24", 560.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 4
    assert [p.grid_unix for p in points] == [1000, 1008, 1016, 1024]
    assert all(p.segment_id == 1 for p in points)


def test_6s_then_8s_interval() -> None:
    """Verify 6-second observation followed by 8-second interval."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1006, "2014-01-01 00:00:06", 520.0, 0.0),
        (1014, "2014-01-01 00:00:14", 540.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 2
    assert [p.grid_unix for p in points] == [1000, 1008]
    assert points[0].aggregate_w == 500.0
    assert points[0].is_imputed is False
    # Grid 1008 holds value from most recent observation before 1008 (t=1006 with agg=520.0)
    assert points[1].aggregate_w == 520.0
    assert points[1].is_imputed is True


def test_8s_then_14s_interval() -> None:
    """Verify 8-second interval followed by 14-second interval (<= 16s)."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    observations = [
        (1000, "2014-01-01 00:00:00", 500.0, 0.0),
        (1008, "2014-01-01 00:00:08", 550.0, 0.0),
        (1022, "2014-01-01 00:00:22", 600.0, 0.0),
        (1030, "2014-01-01 00:00:30", 650.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 4
    assert [p.grid_unix for p in points] == [1000, 1008, 1016, 1024]
    assert [p.segment_id for p in points] == [1, 1, 1, 1]
    assert points[2].grid_unix == 1016
    assert points[2].is_imputed is True
    assert points[2].aggregate_w == 550.0  # Zero-order hold from t=1008


def test_non_8s_wallclock_anchor() -> None:
    """Verify segment anchor starting at arbitrary non-8s wall-clock unix timestamp (e.g., 1003)."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    observations = [
        (1003, "2014-01-01 00:00:03", 500.0, 0.0),
        (1011, "2014-01-01 00:00:11", 550.0, 0.0),
        (1019, "2014-01-01 00:00:19", 600.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 3
    assert [p.grid_unix for p in points] == [1003, 1011, 1019]
    assert [p.segment_id for p in points] == [1, 1, 1]


def test_multiple_irregular_observations_around_grid_point() -> None:
    """Verify zero-order hold correctly latches most recent observation before grid point."""
    resampler = StreamingTimebaseResampler(ResamplingConfig(grid_step_s=8, max_bridge_gap_s=16))
    observations = [
        (1000, "2014-01-01 00:00:00", 100.0, 0.0),
        (1002, "2014-01-01 00:00:02", 120.0, 0.0),
        (1005, "2014-01-01 00:00:05", 150.0, 0.0),
        (1007, "2014-01-01 00:00:07", 180.0, 0.0),
        (1010, "2014-01-01 00:00:10", 200.0, 0.0),
    ]
    points = list(resampler.resample_observations(iter(observations), household_id=1))

    assert len(points) == 2
    assert [p.grid_unix for p in points] == [1000, 1008]
    assert points[0].aggregate_w == 100.0
    # Grid 1008 must latch 180.0 from t=1007 (the latest observation prior to t=1008)
    assert points[1].aggregate_w == 180.0
    assert points[1].is_imputed is True

