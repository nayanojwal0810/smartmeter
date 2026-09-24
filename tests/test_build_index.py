"""Unit tests for streaming build_timebase_index workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from scripts.build_timebase_index import build_manifest_streaming, generate_markdown
from src.data.timebase import ResamplingConfig


def test_streaming_manifest_build_and_segment_counting(tmp_path: Path) -> None:
    """Verify streaming index generation writes valid JSONL files and counts segments accurately."""
    # Create mock CSV files
    # House 1: No kettle, 2 segments:
    #   Seg 1 has 12 points (at 8s step = 2 windows of 5 points + 2 leftover)
    #   Seg 2 has 3 points (< 5 points = 0 windows) -> total segs=2, with_win=1, without_win=1
    h1_csv = tmp_path / "CLEAN_House1.csv"
    with open(h1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)] + ["Issues"])
        # Seg 1: 12 rows at 8s step
        for i in range(12):
            writer.writerow([
                f"2014-01-01 00:00:{i*8:02d}",
                1000 + i * 8,
                500.0 + i,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0,
            ])
        # Seg 2: gap of 100s > 16s -> starts seg 2 with 3 rows
        for i in range(3):
            writer.writerow([
                f"2014-01-01 01:00:{i*8:02d}",
                2000 + i * 8,
                600.0 + i,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0,
            ])

    # House 2: Monitored kettle (Appliance 8), 1 segment with 10 points (2 windows of 5 points)
    # Window 1: Kettle active at 2000W (points 0..2)
    # Window 2: Kettle active at 0W (points 5..9)
    h2_csv = tmp_path / "CLEAN_House2.csv"
    with open(h2_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)] + ["Issues"])
        for i in range(10):
            kettle_w = 2000.0 if i < 3 else 0.0
            writer.writerow([
                f"2014-01-01 00:00:{i*8:02d}",
                1000 + i * 8,
                2500.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                kettle_w,  # Appliance 8 = Kettle in H2
                0.0,
                0,
            ])

    # Real metadata file
    project_meta = Path("data/raw/MetaData_Tables.xlsx")
    if not project_meta.exists():
        pytest.skip("MetaData_Tables.xlsx not found")
    meta_xlsx = project_meta

    out_win_jsonl = tmp_path / "artifacts" / "timebase_windows.jsonl"
    out_tgt_jsonl = tmp_path / "artifacts" / "kettle_evaluation_targets.jsonl"

    config = ResamplingConfig(grid_step_s=8, window_points=5)

    manifest = build_manifest_streaming(
        data_dir=tmp_path,
        metadata_path=meta_xlsx,
        output_windows_jsonl=out_win_jsonl,
        output_targets_jsonl=out_tgt_jsonl,
        config=config,
    )

    # 1. Check generated JSONL files exist and have correct line counts
    assert out_win_jsonl.exists()
    assert out_tgt_jsonl.exists()

    with open(out_win_jsonl, "r", encoding="utf-8") as f:
        win_lines = [json.loads(line) for line in f]
    with open(out_tgt_jsonl, "r", encoding="utf-8") as f:
        tgt_lines = [json.loads(line) for line in f]

    # Total windows: House 1 has 2 windows, House 2 has 2 windows -> 4 windows
    assert len(win_lines) == 4
    assert len(tgt_lines) == 4

    # 2. Check window-target keying by window_id
    for w, t in zip(win_lines, tgt_lines):
        assert w["window_id"] == t["window_id"]
        assert w["household_id"] == t["household_id"]
        assert w["segment_id"] == t["segment_id"]
        # Verify no kettle target in window
        assert "has_kettle_active" not in w
        assert "kettle_active_points_count" not in w

    # 3. Check House 1 evaluation target is ineligible (no kettle)
    assert tgt_lines[0]["is_evaluation_eligible"] is False
    assert tgt_lines[0]["has_active_target"] is False

    # 4. Check House 2 evaluation target: Window 1 is positive, Window 2 is negative
    assert tgt_lines[2]["household_id"] == 2
    assert tgt_lines[2]["is_evaluation_eligible"] is True
    assert tgt_lines[2]["has_active_target"] is True
    assert tgt_lines[3]["has_active_target"] is False

    # 5. Check segment counting
    h1_summary = next(h for h in manifest["households"] if h["household_id"] == 1)
    assert h1_summary["total_contiguous_segments"] == 2
    assert h1_summary["segments_with_windows_count"] == 1
    assert h1_summary["segments_without_windows_count"] == 1

    # 6. Check duration span calculation
    assert manifest["config"]["window_duration_seconds"] == (5 - 1) * 8  # 32s for 5 points

    # 7. Check markdown report generation
    md = generate_markdown(manifest)
    assert "# Regular Timebase & 510-Point Model Window Index Manifest" in md
    assert "House 1" in md
    assert "House 2" in md


def test_18s_raw_gap_across_segment_boundary_integration(tmp_path: Path) -> None:
    """Verify that a 18s raw gap (>16s timebase, <=20s target) preserves single valid target event."""
    # House 2 CSV: 18s gap between point 4 and point 5:
    # Points 0..4 (t=1000..1032, 5 points -> Window 1 in Segment 1), kettle=2000W at point 4 (t=1032)
    # Point 5 at t=1050 (18s gap > 16s -> starts Segment 2), kettle=2000W at t=1050 and t=1058
    # Points 6..9 at t=1058, 1066, 1074, 1082 (Segment 2 has 5 points -> Window 2 in Segment 2)
    h2_csv = tmp_path / "CLEAN_House2.csv"
    with open(h2_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)] + ["Issues"])
        # Seg 1: 5 points (t=1000, 1008, 1016, 1024, 1032)
        for i in range(5):
            k_w = 2000.0 if i == 4 else 0.0  # Active at t=1032
            writer.writerow([
                f"2014-01-01 00:00:{i*8:02d}",
                1000 + i * 8,
                2500.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                k_w,  # Appliance 8 = Kettle
                0.0,
                0,
            ])
        # Seg 2: gap of 18s (t=1050, 1058, 1066, 1074, 1082)
        for i in range(5):
            u = 1050 + i * 8
            k_w = 2000.0 if i == 0 else 0.0  # Active at t=1050 (18s after t=1032)
            writer.writerow([
                f"2014-01-01 00:01:{i*8:02d}",
                u,
                2500.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                k_w,
                0.0,
                0,
            ])

    project_meta = Path("data/raw/MetaData_Tables.xlsx")
    if not project_meta.exists():
        pytest.skip("MetaData_Tables.xlsx not found")
    meta_xlsx = project_meta

    out_win = tmp_path / "artifacts" / "timebase_windows.jsonl"
    out_tgt = tmp_path / "artifacts" / "kettle_evaluation_targets.jsonl"

    config = ResamplingConfig(grid_step_s=8, window_points=5)

    manifest = build_manifest_streaming(
        data_dir=tmp_path,
        metadata_path=meta_xlsx,
        output_windows_jsonl=out_win,
        output_targets_jsonl=out_tgt,
        config=config,
    )

    with open(out_win, "r", encoding="utf-8") as f:
        win_lines = [json.loads(line) for line in f]
    with open(out_tgt, "r", encoding="utf-8") as f:
        tgt_lines = [json.loads(line) for line in f]

    assert len(win_lines) == 2
    assert len(tgt_lines) == 2

    # Segment 1 and Segment 2 are distinct segments
    assert win_lines[0]["segment_id"] == 1
    assert win_lines[1]["segment_id"] == 2

    # Both windows are active because they overlap the same continuous event spanning t=1032 to t=1050 (18s duration <= 600s)
    assert tgt_lines[0]["has_active_target"] is True
    assert tgt_lines[0]["active_target_points_count"] == 1  # t=1032
    assert tgt_lines[1]["has_active_target"] is True
    assert tgt_lines[1]["active_target_points_count"] == 1  # t=1050

