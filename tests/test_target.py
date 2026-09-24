"""Unit tests for Kettle ground-truth evaluation target module (Decision D-005)."""

from __future__ import annotations

import pytest

from src.data.target import RawKettleTargetExtractor, TargetEvent, WindowEvaluationTarget
from src.data.timebase import GridPoint


def test_kettle_threshold_and_continuity() -> None:
    """Verify >=1500W threshold and <=20s continuity gap grouping in raw observation domain."""
    extractor = RawKettleTargetExtractor(power_threshold_w=1500.0, max_continuity_gap_s=20, max_duration_s=600)

    # 3 active points with 8s spacing (t=1000, 1008, 1016) -> 1 valid event, duration = 16s
    observations = [
        (1000, 2000.0),
        (1008, 2100.0),
        (1016, 2200.0),
        (1024, 0.0),
    ]

    events = extractor.extract_from_observations(observations, household_id=2, has_kettle_channel=True)

    assert len(events) == 1
    ev = events[0]
    assert ev.is_valid is True
    assert ev.duration_seconds == 16
    assert ev.start_unix == 1000
    assert ev.end_unix == 1016
    assert ev.peak_power_w == 2200.0
    assert ev.sample_count == 3


def test_gap_gt_20s_separates_events() -> None:
    """Verify that gaps > 20s separate active samples into distinct events."""
    extractor = RawKettleTargetExtractor(power_threshold_w=1500.0, max_continuity_gap_s=20, max_duration_s=600)

    # Event 1 at t=1000, Event 2 at t=1032 (gap = 32s > 20s)
    observations = [
        (1000, 2000.0),
        (1008, 0.0),
        (1016, 0.0),
        (1024, 0.0),
        (1032, 2000.0),
        (1060, 0.0),
    ]

    events = extractor.extract_from_observations(observations, household_id=2, has_kettle_channel=True)

    assert len(events) == 2
    assert events[0].start_unix == 1000
    assert events[0].end_unix == 1000
    assert events[0].duration_seconds == 0
    assert events[1].start_unix == 1032
    assert events[1].end_unix == 1032
    assert events[1].duration_seconds == 0


def test_gap_between_16s_and_20s_preserves_target_event() -> None:
    """Verify that a raw gap between 16s and 20s (e.g. 18s) does NOT split the target event."""
    extractor = RawKettleTargetExtractor(power_threshold_w=1500.0, max_continuity_gap_s=20, max_duration_s=600)

    # Raw gap of 18s between t=1000 and t=1018 (<= 20s, but > 16s timebase segment limit)
    observations = [
        (1000, 2500.0),
        (1018, 2600.0),
        (1026, 2700.0),
        (1060, 0.0),
    ]

    events = extractor.extract_from_observations(observations, household_id=2, has_kettle_channel=True)

    # Must remain ONE event spanning 1000 to 1026 (duration 26s)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_unix == 1000
    assert ev.end_unix == 1026
    assert ev.duration_seconds == 26
    assert ev.is_valid is True
    assert ev.sample_count == 3


def test_max_duration_gt_600s_rejection() -> None:
    """Verify that candidate events > 600s are rejected and do not produce positive evaluation targets."""
    extractor = RawKettleTargetExtractor(power_threshold_w=1500.0, max_continuity_gap_s=20, max_duration_s=600)

    # 100 points spaced at 8s = 792s duration (> 600s)
    observations = [(1000 + i * 8, 2500.0) for i in range(100)] + [(1000 + 100 * 8 + 30, 0.0)]

    events = extractor.extract_from_observations(observations, household_id=3, has_kettle_channel=True)

    assert len(events) == 1
    ev = events[0]
    assert ev.duration_seconds == 792
    assert ev.is_valid is False
    assert "exceeds maximum threshold 600s" in (ev.rejection_reason or "")

    # Now verify window matching: rejected event must not produce active target points
    grid_points = [
        GridPoint(household_id=3, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=3000.0, kettle_w=2500.0)
        for i in range(100)
    ]
    # Feed into extractor rolling queue
    extractor.extract_from_observations(observations, household_id=3, has_kettle_channel=True)
    # Put event into finalized
    extractor._finalized_events.append(ev)

    win_tgt = extractor.match_window_target(
        window_id=1,
        household_id=3,
        segment_id=1,
        window_grid_points=grid_points,
        has_kettle_channel=True,
    )
    assert win_tgt.has_active_target is False
    assert win_tgt.active_target_points_count == 0
    assert win_tgt.valid_target_events_count == 0
    assert win_tgt.rejected_target_events_count == 1


def test_h12_exclusion() -> None:
    """Verify House 12 is excluded from primary evaluation target generation under D-005."""
    extractor = RawKettleTargetExtractor(power_threshold_w=1500.0, max_continuity_gap_s=20, max_duration_s=600)

    observations = [(1000 + i * 8, 2048.0) for i in range(5)] + [(1050, 0.0)]

    events = extractor.extract_from_observations(observations, household_id=12, has_kettle_channel=True)

    assert len(events) == 1
    assert events[0].is_valid is False
    assert "Household 12 excluded" in (events[0].rejection_reason or "")

    grid_points = [
        GridPoint(household_id=12, segment_id=1, grid_unix=1000 + i * 8, aggregate_w=2500.0, kettle_w=2048.0)
        for i in range(5)
    ]
    win_tgt = extractor.match_window_target(
        window_id=10,
        household_id=12,
        segment_id=1,
        window_grid_points=grid_points,
        has_kettle_channel=True,
    )
    assert win_tgt.is_evaluation_eligible is False
    assert win_tgt.has_active_target is False
    assert win_tgt.active_target_points_count == 0


def test_unmonitored_households_exclusion() -> None:
    """Verify households without monitored Kettle channel are marked ineligible."""
    extractor = RawKettleTargetExtractor()
    assert extractor.is_household_eligible(household_id=1, has_kettle_channel=False) is False
    assert extractor.is_household_eligible(household_id=10, has_kettle_channel=False) is False
    assert extractor.is_household_eligible(household_id=2, has_kettle_channel=True) is True
