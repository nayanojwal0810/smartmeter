"""Unit tests for streaming window generation and indexing module."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.timebase import GridPoint, ResamplingConfig
from src.data.windowing import StreamingWindowIndexer, WindowMetadata


def test_window_formation_within_segment() -> None:
    """Verify that points within a segment assemble into exact non-overlapping windows."""
    config = ResamplingConfig(grid_step_s=8, window_points=5)
    indexer = StreamingWindowIndexer(config=config)

    # 12 points in segment 1 -> should yield exactly 2 windows of 5 points, dropping remaining 2 points
    grid_points = [
        GridPoint(household_id=1, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=500.0 + i * 10, kettle_w=0.0)
        for i in range(12)
    ]

    windows = list(indexer.generate_windows(iter(grid_points)))
    assert len(windows) == 2

    # First window: points 0..4 (unix 1000..1032, span = 32s = (5-1)*8)
    w1 = windows[0]
    assert w1.window_id == 1
    assert w1.segment_id == 1
    assert w1.window_index_in_segment == 0
    assert w1.num_points == 5
    assert w1.start_unix == 1000
    assert w1.end_unix == 1032
    assert w1.duration_seconds == 32

    # Verify strong kettle targets are strictly NOT present in structural WindowMetadata
    d1 = w1.to_dict()
    assert "kettle_w" not in d1
    assert "has_kettle_active" not in d1
    assert "kettle_active_points_count" not in d1

    # Second window: points 5..9 (unix 1040..1072)
    w2 = windows[1]
    assert w2.window_id == 2
    assert w2.segment_id == 1
    assert w2.window_index_in_segment == 1
    assert w2.start_unix == 1040
    assert w2.end_unix == 1072


def test_no_cross_segment_windows() -> None:
    """Verify that window formation never crosses segment boundaries."""
    config = ResamplingConfig(grid_step_s=8, window_points=5)
    indexer = StreamingWindowIndexer(config=config)

    # Segment 1 has 4 points (< 5 points -> cannot form window, should be dropped)
    # Segment 2 has 5 points (= 5 points -> forms 1 window)
    grid_points = [
        GridPoint(household_id=1, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=500.0, kettle_w=0.0)
        for i in range(4)
    ] + [
        GridPoint(household_id=1, segment_id=2, grid_unix=2000 + i * 8, aggregate_w=600.0, kettle_w=0.0)
        for i in range(5)
    ]

    windows = list(indexer.generate_windows(iter(grid_points)))
    assert len(windows) == 1
    assert windows[0].segment_id == 2
    assert windows[0].start_unix == 2000
    assert windows[0].end_unix == 2032


def test_full_510_point_window_size() -> None:
    """Verify standard 510-point window generation behavior and exact 4072-second span."""
    config = ResamplingConfig(grid_step_s=8, window_points=510)
    indexer = StreamingWindowIndexer(config=config)

    # 1025 points in segment 1 -> exactly 2 windows of 510 points, dropping the leftover 5 points
    grid_points = [
        GridPoint(household_id=3, segment_id=1, grid_unix=1000000 + i * 8, aggregate_w=400.0, kettle_w=0.0)
        for i in range(1025)
    ]

    windows = list(indexer.generate_windows(iter(grid_points)))
    assert len(windows) == 2
    assert windows[0].num_points == 510
    # Span is exactly (510 - 1) * 8 = 4072 seconds
    assert windows[0].duration_seconds == 4072
    assert windows[0].end_unix - windows[0].start_unix == 4072
    assert windows[1].num_points == 510
    assert windows[1].duration_seconds == 4072


def test_windows_immediately_before_and_after_segment_boundary() -> None:
    """Verify window formation right at segment boundary with leftover residual points discarded."""
    config = ResamplingConfig(grid_step_s=8, window_points=5)
    indexer = StreamingWindowIndexer(config=config)

    # Seg 1 has 7 points: points 0..4 make Window 1, points 5..6 (2 points) are leftover and discarded.
    # Seg 2 has 6 points: points 0..4 make Window 2, point 5 is leftover and discarded.
    grid_points = [
        GridPoint(household_id=4, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=500.0 + i, kettle_w=0.0)
        for i in range(7)
    ] + [
        GridPoint(household_id=4, segment_id=2, grid_unix=5000 + i * 8, aggregate_w=600.0 + i, kettle_w=0.0)
        for i in range(6)
    ]

    windows = list(indexer.generate_windows(iter(grid_points)))
    assert len(windows) == 2

    # Window 1 belongs strictly to Seg 1
    assert windows[0].segment_id == 1
    assert windows[0].window_index_in_segment == 0
    assert windows[0].start_unix == 1000
    assert windows[0].end_unix == 1032
    assert windows[0].num_points == 5

    # Window 2 belongs strictly to Seg 2
    assert windows[1].segment_id == 2
    assert windows[1].window_index_in_segment == 0
    assert windows[1].start_unix == 5000
    assert windows[1].end_unix == 5032
    assert windows[1].num_points == 5


def test_generate_windows_with_points() -> None:
    """Verify generate_windows_with_points yields exact GridPoint sequences for target extractor."""
    config = ResamplingConfig(grid_step_s=8, window_points=5)
    indexer = StreamingWindowIndexer(config=config)

    grid_points = [
        GridPoint(household_id=5, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=500.0 + i, kettle_w=100.0 * i)
        for i in range(5)
    ]

    items = list(indexer.generate_windows_with_points(iter(grid_points)))
    assert len(items) == 1
    win_meta, win_points = items[0]
    assert win_meta.window_id == 1
    assert len(win_points) == 5
    assert [p.grid_unix for p in win_points] == [1000, 1008, 1016, 1024, 1032]
