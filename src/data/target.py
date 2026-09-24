"""Kettle Ground-Truth Evaluation Target Module.

Implements the locked Kettle target policy under Decision D-005 for offline
evaluation and temporal localization validation:
- Kettle ON threshold: >= 1500 W.
- Consecutive active samples remain in the same candidate event when raw timestamp gap <= 20s.
- Raw timestamp gap > 20s closes the candidate event.
- Candidate events > 600s duration are excluded from localization ground truth (rejected).
- Target events are constructed in the raw observation domain independently of the
  16-second model timebase segmentation boundary.
- House 12 is excluded from primary Kettle localization evaluation.
- Unmonitored households (H1, H10, H15, H16, H18) are marked ineligible.
- Projects valid target events onto the 8s regular model grid.
- Binds evaluation targets to windows strictly via `window_id`.

IMPORTANT: Strong sub-meter measurements and target metadata are strictly reserved
for evaluation and validation. They must NEVER be used to construct weak model training labels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple

from src.data.timebase import GridPoint


@dataclass
class TargetEvent:
    """Descriptor for a contiguous appliance target event."""

    event_id: int
    household_id: int
    start_unix: int
    end_unix: int
    duration_seconds: int
    peak_power_w: float
    sample_count: int
    is_valid: bool  # True if duration <= 600s and household is eligible
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert target event descriptor to dictionary."""
        return {
            "event_id": self.event_id,
            "household_id": self.household_id,
            "start_unix": self.start_unix,
            "end_unix": self.end_unix,
            "duration_seconds": self.duration_seconds,
            "peak_power_w": round(self.peak_power_w, 2),
            "sample_count": self.sample_count,
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class WindowEvaluationTarget:
    """Strong Kettle evaluation target descriptor associated with a model window.

    This metadata is strictly separated from model-input WindowMetadata and must
    never be consumed during model training.
    """

    window_id: int
    household_id: int
    segment_id: int
    is_evaluation_eligible: bool  # False for H12 or unmonitored households
    has_active_target: bool  # True if window contains points from valid target events
    active_target_points_count: int  # Count of grid points in valid target events
    valid_target_events_count: int  # Number of distinct valid target events overlapping window
    rejected_target_events_count: int  # Number of rejected candidate events overlapping window

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation target descriptor to dictionary."""
        return {
            "window_id": self.window_id,
            "household_id": self.household_id,
            "segment_id": self.segment_id,
            "is_evaluation_eligible": self.is_evaluation_eligible,
            "has_active_target": self.has_active_target,
            "active_target_points_count": self.active_target_points_count,
            "valid_target_events_count": self.valid_target_events_count,
            "rejected_target_events_count": self.rejected_target_events_count,
        }


class RawKettleTargetExtractor:
    """Streaming target event extractor operating on raw Kettle observations.

    Preserves the <=20s continuity gap and >600s duration policy independently of
    timebase resampling or 16s segment breaks.
    """

    def __init__(
        self,
        power_threshold_w: float = 1500.0,
        max_continuity_gap_s: int = 20,
        max_duration_s: int = 600,
        excluded_households: Optional[Sequence[int]] = None,
    ) -> None:
        self.power_threshold_w = power_threshold_w
        self.max_continuity_gap_s = max_continuity_gap_s
        self.max_duration_s = max_duration_s
        # Locked D-005: H12 excluded from primary evaluation
        self.excluded_households = set(excluded_households or [12])
        self._next_event_id = 1

        # Streaming state
        self._current_event: Optional[Dict[str, Any]] = None
        self._finalized_events: Deque[TargetEvent] = deque()

    def is_household_eligible(self, household_id: int, has_kettle_channel: bool) -> bool:
        """Check if household is eligible for primary Kettle localization evaluation."""
        if not has_kettle_channel:
            return False
        if household_id in self.excluded_households:
            return False
        return True

    def process_raw_observation(
        self,
        obs_unix: int,
        kettle_w: float,
        household_id: int,
        has_kettle_channel: bool,
    ) -> None:
        """Feed a single raw observation into the streaming target extractor."""
        if not has_kettle_channel:
            return

        if kettle_w >= self.power_threshold_w:
            if self._current_event is None:
                self._current_event = {
                    "household_id": household_id,
                    "start_unix": obs_unix,
                    "end_unix": obs_unix,
                    "peak_power_w": kettle_w,
                    "sample_count": 1,
                }
            else:
                gap_s = obs_unix - self._current_event["end_unix"]
                if gap_s <= self.max_continuity_gap_s:
                    self._current_event["end_unix"] = obs_unix
                    self._current_event["peak_power_w"] = max(
                        self._current_event["peak_power_w"], kettle_w
                    )
                    self._current_event["sample_count"] += 1
                else:
                    self._finalize_current_event(household_id, has_kettle_channel)
                    self._current_event = {
                        "household_id": household_id,
                        "start_unix": obs_unix,
                        "end_unix": obs_unix,
                        "peak_power_w": kettle_w,
                        "sample_count": 1,
                    }
        else:
            if self._current_event is not None:
                gap_s = obs_unix - self._current_event["end_unix"]
                if gap_s > self.max_continuity_gap_s:
                    self._finalize_current_event(household_id, has_kettle_channel)

    def close_stream(self, household_id: int, has_kettle_channel: bool) -> None:
        """Close any remaining open event at stream termination."""
        if self._current_event is not None:
            self._finalize_current_event(household_id, has_kettle_channel)

    def _finalize_current_event(self, household_id: int, has_kettle_channel: bool) -> None:
        """Validate and finalize candidate event against duration and eligibility policies."""
        if self._current_event is None:
            return

        ev_dict = self._current_event
        self._current_event = None

        start_u = ev_dict["start_unix"]
        end_u = ev_dict["end_unix"]
        duration_s = end_u - start_u
        peak_w = ev_dict["peak_power_w"]
        count = ev_dict["sample_count"]

        is_eligible = self.is_household_eligible(household_id, has_kettle_channel)
        is_valid = True
        rejection_reason = None

        if not is_eligible:
            is_valid = False
            rejection_reason = f"Household {household_id} excluded from primary evaluation"
        elif duration_s > self.max_duration_s:
            is_valid = False
            rejection_reason = (
                f"Duration {duration_s}s exceeds maximum threshold {self.max_duration_s}s"
            )

        ev = TargetEvent(
            event_id=self._next_event_id,
            household_id=household_id,
            start_unix=start_u,
            end_unix=end_u,
            duration_seconds=duration_s,
            peak_power_w=peak_w,
            sample_count=count,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )
        self._next_event_id += 1
        self._finalized_events.append(ev)

    def extract_from_observations(
        self,
        observations: Sequence[Tuple[int, float]],
        household_id: int,
        has_kettle_channel: bool,
    ) -> List[TargetEvent]:
        """Convenience method to extract all events from an observation sequence."""
        self._current_event = None
        self._finalized_events.clear()

        for obs_unix, kettle_w in observations:
            self.process_raw_observation(obs_unix, kettle_w, household_id, has_kettle_channel)
        self.close_stream(household_id, has_kettle_channel)

        events = list(self._finalized_events)
        self._finalized_events.clear()
        return events

    def match_window_target(
        self,
        window_id: int,
        household_id: int,
        segment_id: int,
        window_grid_points: Sequence[GridPoint],
        has_kettle_channel: bool,
    ) -> WindowEvaluationTarget:
        """Project current target events onto a 510-point window slice.

        Purges target events older than window start from the rolling queue to ensure
        O(1) memory overhead.
        """
        is_eligible = self.is_household_eligible(household_id, has_kettle_channel)
        if not is_eligible or not window_grid_points:
            return WindowEvaluationTarget(
                window_id=window_id,
                household_id=household_id,
                segment_id=segment_id,
                is_evaluation_eligible=is_eligible,
                has_active_target=False,
                active_target_points_count=0,
                valid_target_events_count=0,
                rejected_target_events_count=0,
            )

        win_start = window_grid_points[0].grid_unix
        win_end = window_grid_points[-1].grid_unix

        # Purge obsolete events older than window start (with margin)
        while self._finalized_events and self._finalized_events[0].end_unix < win_start:
            self._finalized_events.popleft()

        # Collect events that overlap [win_start, win_end]
        overlapping_events: List[TargetEvent] = []
        for ev in self._finalized_events:
            if max(win_start, ev.start_unix) <= min(win_end, ev.end_unix):
                overlapping_events.append(ev)
            elif ev.start_unix > win_end:
                # Events are strictly ordered by start_unix
                break

        # Check current active open event if any
        if self._current_event is not None:
            cur_start = self._current_event["start_unix"]
            cur_end = self._current_event["end_unix"]
            if max(win_start, cur_start) <= min(win_end, cur_end):
                cur_dur = cur_end - cur_start
                is_cur_valid = (cur_dur <= self.max_duration_s) and is_eligible
                temp_ev = TargetEvent(
                    event_id=self._next_event_id,
                    household_id=household_id,
                    start_unix=cur_start,
                    end_unix=cur_end,
                    duration_seconds=cur_dur,
                    peak_power_w=self._current_event["peak_power_w"],
                    sample_count=self._current_event["sample_count"],
                    is_valid=is_cur_valid,
                )
                overlapping_events.append(temp_ev)

        # Mark active grid points that fall inside valid overlapping events
        valid_events = [ev for ev in overlapping_events if ev.is_valid]
        rejected_events = [ev for ev in overlapping_events if not ev.is_valid]

        active_points_count = 0
        for pt in window_grid_points:
            pt_u = pt.grid_unix
            if any(ev.start_unix <= pt_u <= ev.end_unix for ev in valid_events):
                active_points_count += 1

        return WindowEvaluationTarget(
            window_id=window_id,
            household_id=household_id,
            segment_id=segment_id,
            is_evaluation_eligible=is_eligible,
            has_active_target=(active_points_count > 0),
            active_target_points_count=active_points_count,
            valid_target_events_count=len(valid_events),
            rejected_target_events_count=len(rejected_events),
        )
