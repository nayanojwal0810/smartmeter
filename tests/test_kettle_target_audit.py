"""Unit tests for Kettle target quality auditor."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from src.validation.kettle_target_audit import (
    EventTrackingStats,
    KettleSignalStats,
    KettleTargetAuditor,
    RunningStats,
)


def test_running_stats_welford() -> None:
    """Verify Welford streaming mean, variance, standard deviation, and min/max."""
    stats = RunningStats()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        stats.update(v)

    assert stats.count == 5
    assert stats.min_val == 10.0
    assert stats.max_val == 50.0
    assert pytest.approx(stats.mean, 0.001) == 30.0
    # Sample variance of [10, 20, 30, 40, 50] = 250.0, std = sqrt(250) = 15.811
    assert pytest.approx(stats.variance, 0.001) == 250.0
    assert pytest.approx(stats.std, 0.001) == math.sqrt(250.0)


def test_kettle_signal_stats_thresholds() -> None:
    """Verify descriptive threshold counter accumulation."""
    k_stats = KettleSignalStats()
    for v in [0.0, 0.0, 250.0, 750.0, 1200.0, 1800.0, 2200.0, 2700.0, 3100.0, -10.0]:
        k_stats.update(v)

    d = k_stats.to_dict()
    assert d["total_observations"] == 10
    assert d["zero_count"] == 2
    assert d["non_zero_count"] == 8
    assert d["negative_count"] == 1
    assert d["gt_500w_count"] == 6  # 750, 1200, 1800, 2200, 2700, 3100
    assert d["gt_1000w_count"] == 5
    assert d["gt_1500w_count"] == 4
    assert d["gt_2000w_count"] == 3
    assert d["gt_2500w_count"] == 2
    assert d["gt_3000w_count"] == 1


def test_event_tracking_stats() -> None:
    """Verify event duration, gap calculation, and active time percentage."""
    ev = EventTrackingStats()
    ev.add_event(duration_s=60.0, peak_w=2200.0, gap_since_last_s=None)
    ev.add_event(duration_s=120.0, peak_w=2500.0, gap_since_last_s=3600.0)
    ev.add_event(duration_s=180.0, peak_w=2800.0, gap_since_last_s=7200.0)

    d = ev.to_dict(total_monitored_s=86400.0)
    assert d["event_count"] == 3
    assert d["total_active_seconds"] == 360.0
    assert d["duration_min_s"] == 60.0
    assert d["duration_median_s"] == 120.0
    assert d["duration_mean_s"] == 120.0
    assert d["duration_max_s"] == 180.0
    assert d["peak_power_max_w"] == 2800.0
    assert pytest.approx(d["peak_power_mean_w"], 0.1) == 2500.0
    assert d["gap_between_events_median_s"] == 5400.0
    assert pytest.approx(d["gap_between_events_mean_s"], 0.1) == 5400.0
    assert pytest.approx(d["active_time_pct"], 0.001) == (360.0 / 86400.0) * 100.0


def test_metadata_channel_discovery() -> None:
    """Verify programmatic discovery of Kettle channels from metadata workbook."""
    meta_path = Path("data/raw/MetaData_Tables.xlsx")
    if not meta_path.exists():
        pytest.skip("MetaData_Tables.xlsx not found")

    auditor = KettleTargetAuditor(metadata_path=meta_path)
    # Check positive house (H2 -> Appliance8)
    col, name, model = auditor.discover_kettle_channel(2)
    assert col == "Appliance8"
    assert "Kettle" in name

    # Check negative house (H1 -> None)
    col_neg, name_neg, model_neg = auditor.discover_kettle_channel(1)
    assert col_neg is None


def test_auditor_mock_household_csv() -> None:
    """Verify auditing on mock CSV containing positive Kettle events, malformed rows, and gaps."""
    csv_content = (
        "Time,Unix,Aggregate,Appliance1,Appliance2,Appliance3,Appliance4,Appliance5,Appliance6,Appliance7,Appliance8,Appliance9,Issues\n"
        "2013-09-17 22:08:11,1379455691,500,0,0,0,0,0,0,0,0,0,0\n"
        "2013-09-17 22:08:19,1379455699,2500,0,0,0,0,0,0,0,2100,0,0\n"
        "2013-09-17 22:08:27,1379455707,2550,0,0,0,0,0,0,0,2150,0,0\n"
        "malformed,row,with,missing,columns\n"
        "2013-09-17 22:08:35,1379455715,450,0,0,0,0,0,0,0,0,0,0\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = Path(tmpdir) / "CLEAN_House2.csv"
        mock_file.write_text(csv_content, encoding="utf-8")

        meta_path = Path("data/raw/MetaData_Tables.xlsx")
        if not meta_path.exists():
            pytest.skip("MetaData_Tables.xlsx not found")
        auditor = KettleTargetAuditor(metadata_path=meta_path)
        res = auditor.audit_household(mock_file, household_id=2)

        assert res.household_id == 2
        assert res.is_positive is True
        assert res.kettle_channel == "Appliance8"
        assert res.total_rows == 5
        assert res.events_500w.event_count == 1
        assert res.events_1500w.event_count == 1
        assert res.kettle_active_500w_samples == 2
        assert res.kettle_exceeds_aggregate_count == 0
        assert res.aggregate_stats.count == 4

        summary = auditor.generate_summary([res])
        assert summary["metadata_summary"]["positive_households_count"] == 1
        md = auditor.generate_markdown(summary)
        assert "Kettle Target Quality & Activation Profile Audit Report" in md
