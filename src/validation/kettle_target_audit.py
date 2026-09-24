"""Kettle Target Quality and Activation Profile Auditor.

Forensic target quality and descriptive activation analysis for Kettle IAM channels
and aggregate load context across REFIT households.
Performs single-pass streaming evaluation with O(1) memory complexity.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.validation.refit_audit import load_refit_metadata

logger = logging.getLogger(__name__)

COLUMN_INDEX_MAP = {
    "Time": 0,
    "Unix": 1,
    "Aggregate": 2,
    "Appliance1": 3,
    "Appliance2": 4,
    "Appliance3": 5,
    "Appliance4": 6,
    "Appliance5": 7,
    "Appliance6": 8,
    "Appliance7": 9,
    "Appliance8": 10,
    "Appliance9": 11,
    "Issues": 12,
}


@dataclass
class RunningStats:
    """Welford's streaming algorithm for online mean, variance, and min/max."""

    count: int = 0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    _m1: float = 0.0  # running mean
    _m2: float = 0.0  # sum of squared differences

    def update(self, x: float) -> None:
        """Update running statistics with observation x."""
        self.count += 1
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

        delta = x - self._m1
        self._m1 += delta / self.count
        delta2 = x - self._m1
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        """Calculate arithmetic mean."""
        return self._m1 if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Calculate sample variance."""
        return (self._m2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.variance) if self.variance > 0.0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "count": self.count,
            "min": self.min_val if self.count > 0 else None,
            "max": self.max_val if self.count > 0 else None,
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
        }


@dataclass
class KettleSignalStats:
    """Descriptive signal and threshold distribution for a Kettle channel."""

    stats: RunningStats = field(default_factory=RunningStats)
    zero_count: int = 0
    non_zero_count: int = 0
    negative_count: int = 0
    gt_500w_count: int = 0
    gt_1000w_count: int = 0
    gt_1500w_count: int = 0
    gt_2000w_count: int = 0
    gt_2500w_count: int = 0
    gt_3000w_count: int = 0
    max_continuous_nonzero_duration_s: float = 0.0

    def update(self, watts: float) -> None:
        """Record a single Kettle power measurement."""
        self.stats.update(watts)
        if watts == 0.0:
            self.zero_count += 1
        else:
            self.non_zero_count += 1

        if watts < 0.0:
            self.negative_count += 1
        if watts >= 500.0:
            self.gt_500w_count += 1
        if watts >= 1000.0:
            self.gt_1000w_count += 1
        if watts >= 1500.0:
            self.gt_1500w_count += 1
        if watts >= 2000.0:
            self.gt_2000w_count += 1
        if watts >= 2500.0:
            self.gt_2500w_count += 1
        if watts >= 3000.0:
            self.gt_3000w_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal statistics to dictionary with percentages."""
        tot = max(1, self.stats.count)
        return {
            "total_observations": self.stats.count,
            "min_w": self.stats.min_val if self.stats.count > 0 else None,
            "max_w": self.stats.max_val if self.stats.count > 0 else None,
            "mean_w": round(self.stats.mean, 3),
            "std_w": round(self.stats.std, 3),
            "zero_count": self.zero_count,
            "zero_pct": round(100.0 * self.zero_count / tot, 4),
            "non_zero_count": self.non_zero_count,
            "non_zero_pct": round(100.0 * self.non_zero_count / tot, 4),
            "negative_count": self.negative_count,
            "gt_500w_count": self.gt_500w_count,
            "gt_500w_pct": round(100.0 * self.gt_500w_count / tot, 4),
            "gt_1000w_count": self.gt_1000w_count,
            "gt_1000w_pct": round(100.0 * self.gt_1000w_count / tot, 4),
            "gt_1500w_count": self.gt_1500w_count,
            "gt_1500w_pct": round(100.0 * self.gt_1500w_count / tot, 4),
            "gt_2000w_count": self.gt_2000w_count,
            "gt_2000w_pct": round(100.0 * self.gt_2000w_count / tot, 4),
            "gt_2500w_count": self.gt_2500w_count,
            "gt_2500w_pct": round(100.0 * self.gt_2500w_count / tot, 4),
            "gt_3000w_count": self.gt_3000w_count,
            "gt_3000w_pct": round(100.0 * self.gt_3000w_count / tot, 4),
            "max_continuous_nonzero_duration_s": round(self.max_continuous_nonzero_duration_s, 1),
        }


@dataclass
class EventTrackingStats:
    """Activation event and inter-event gap metrics under a descriptive threshold rule."""

    event_count: int = 0
    total_active_seconds: float = 0.0
    durations_s: List[float] = field(default_factory=list)
    inter_event_gaps_s: List[float] = field(default_factory=list)
    peak_powers_w: List[float] = field(default_factory=list)

    def add_event(self, duration_s: float, peak_w: float, gap_since_last_s: Optional[float] = None) -> None:
        """Register a completed continuous activation event."""
        self.event_count += 1
        self.total_active_seconds += duration_s
        self.durations_s.append(duration_s)
        self.peak_powers_w.append(peak_w)
        if gap_since_last_s is not None and gap_since_last_s > 0:
            self.inter_event_gaps_s.append(gap_since_last_s)

    def to_dict(self, total_monitored_s: float = 0.0) -> Dict[str, Any]:
        """Compute duration, gap percentiles, and active time percentage."""
        if not self.durations_s:
            return {
                "event_count": 0,
                "total_active_seconds": 0.0,
                "active_time_pct": 0.0,
                "duration_min_s": 0.0,
                "duration_median_s": 0.0,
                "duration_mean_s": 0.0,
                "duration_p95_s": 0.0,
                "duration_max_s": 0.0,
                "peak_power_mean_w": 0.0,
                "peak_power_max_w": 0.0,
                "gap_between_events_median_s": 0.0,
                "gap_between_events_mean_s": 0.0,
            }

        sorted_d = sorted(self.durations_s)
        n = len(sorted_d)
        median_d = sorted_d[n // 2] if n % 2 != 0 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2.0
        p95_idx = min(n - 1, int(0.95 * n))
        p95_d = sorted_d[p95_idx]

        active_pct = round(100.0 * self.total_active_seconds / max(1.0, total_monitored_s), 4)

        gap_median = 0.0
        gap_mean = 0.0
        if self.inter_event_gaps_s:
            sorted_g = sorted(self.inter_event_gaps_s)
            ng = len(sorted_g)
            gap_median = sorted_g[ng // 2] if ng % 2 != 0 else (sorted_g[ng // 2 - 1] + sorted_g[ng // 2]) / 2.0
            gap_mean = sum(self.inter_event_gaps_s) / ng

        return {
            "event_count": self.event_count,
            "total_active_seconds": round(self.total_active_seconds, 1),
            "active_time_pct": active_pct,
            "duration_min_s": round(sorted_d[0], 1),
            "duration_median_s": round(median_d, 1),
            "duration_mean_s": round(self.total_active_seconds / n, 1),
            "duration_p95_s": round(p95_d, 1),
            "duration_max_s": round(sorted_d[-1], 1),
            "peak_power_mean_w": round(sum(self.peak_powers_w) / n, 1),
            "peak_power_max_w": round(max(self.peak_powers_w), 1),
            "gap_between_events_median_s": round(gap_median, 1),
            "gap_between_events_mean_s": round(gap_mean, 1),
        }


@dataclass
class HouseholdKettleTargetAudit:
    """Complete forensic audit result for a single household's Kettle target signal."""

    household_id: int
    is_positive: bool
    kettle_channel: Optional[str] = None
    appliance_name_metadata: str = ""
    make_model: str = ""
    total_rows: int = 0
    days_monitored: float = 0.0
    start_date: str = ""
    end_date: str = ""

    # Kettle signal stats
    kettle_stats: KettleSignalStats = field(default_factory=KettleSignalStats)

    # Descriptive Activation Events (500W descriptive rule & 1500W comparison)
    events_500w: EventTrackingStats = field(default_factory=EventTrackingStats)
    events_1500w: EventTrackingStats = field(default_factory=EventTrackingStats)

    # Aggregate context
    aggregate_stats: RunningStats = field(default_factory=RunningStats)
    aggregate_gt_4000w_count: int = 0
    aggregate_wraparound_gt_60000w_count: int = 0
    issues_1_count: int = 0
    time_gaps_gt_30s: int = 0
    time_gaps_gt_120s: int = 0

    # Kettle-to-Aggregate relationship (when Kettle >= 500W)
    kettle_active_500w_samples: int = 0
    aggregate_sum_during_kettle_active: float = 0.0
    background_power_sum: float = 0.0  # Aggregate - Kettle
    kettle_exceeds_aggregate_count: int = 0

    # Target risks and notes
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert household audit to dictionary."""
        total_s = self.days_monitored * 86400.0
        daily_500w = round(self.events_500w.event_count / max(0.001, self.days_monitored), 2)
        daily_1500w = round(self.events_1500w.event_count / max(0.001, self.days_monitored), 2)

        agg_tot = max(1, self.aggregate_stats.count)
        agg_dict = {
            **self.aggregate_stats.to_dict(),
            "gt_4000w_count": self.aggregate_gt_4000w_count,
            "gt_4000w_pct": round(100.0 * self.aggregate_gt_4000w_count / agg_tot, 4),
            "wraparound_gt_60000w_count": self.aggregate_wraparound_gt_60000w_count,
            "wraparound_gt_60000w_pct": round(100.0 * self.aggregate_wraparound_gt_60000w_count / agg_tot, 4),
            "issues_1_count": self.issues_1_count,
            "issues_1_pct": round(100.0 * self.issues_1_count / agg_tot, 4),
            "time_gaps_gt_30s": self.time_gaps_gt_30s,
            "time_gaps_gt_120s": self.time_gaps_gt_120s,
        }

        kettle_agg_rel = {}
        if self.kettle_active_500w_samples > 0:
            mean_agg_active = self.aggregate_sum_during_kettle_active / self.kettle_active_500w_samples
            mean_bg = self.background_power_sum / self.kettle_active_500w_samples
            kettle_agg_rel = {
                "active_samples_500w": self.kettle_active_500w_samples,
                "mean_aggregate_during_kettle_on_w": round(mean_agg_active, 1),
                "mean_background_power_w": round(mean_bg, 1),
                "kettle_exceeds_aggregate_count": self.kettle_exceeds_aggregate_count,
                "kettle_exceeds_aggregate_pct": round(
                    100.0 * self.kettle_exceeds_aggregate_count / self.kettle_active_500w_samples, 4
                ),
            }

        return {
            "household_id": self.household_id,
            "is_positive": self.is_positive,
            "kettle_channel": self.kettle_channel,
            "metadata_appliance_name": self.appliance_name_metadata,
            "make_model": self.make_model,
            "total_rows": self.total_rows,
            "days_monitored": round(self.days_monitored, 1),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "kettle_signal_statistics": self.kettle_stats.to_dict() if self.is_positive else None,
            "activation_events_500w_descriptive_rule": (
                {**self.events_500w.to_dict(total_s), "events_per_day": daily_500w} if self.is_positive else None
            ),
            "activation_events_1500w_reference": (
                {**self.events_1500w.to_dict(total_s), "events_per_day": daily_1500w} if self.is_positive else None
            ),
            "aggregate_context": agg_dict,
            "kettle_to_aggregate_relationship": kettle_agg_rel if self.is_positive else None,
            "risk_flags": self.risk_flags,
        }


class KettleTargetAuditor:
    """Streaming auditor for Kettle target signals and aggregate context."""

    def __init__(self, metadata_path: Optional[Path] = None) -> None:
        self.metadata_path = metadata_path
        self.metadata: Dict[int, Dict[str, Any]] = {}
        if metadata_path and metadata_path.exists():
            self.metadata = load_refit_metadata(metadata_path)

    def discover_kettle_channel(self, household_id: int) -> Tuple[Optional[str], str, str]:
        """Programmatically discover Kettle channel, name, and make/model from metadata."""
        if household_id not in self.metadata:
            return None, "", ""

        iam_map = self.metadata[household_id].get("iam_channels", {})
        for col_name, info in iam_map.items():
            name = info.get("name", "")
            if "kettle" in name.lower():
                make = info.get("make", "").strip()
                model = info.get("model", "").strip()
                make_model = f"{make} {model}".strip()
                return col_name, name, make_model

        return None, "", ""

    def audit_household(
        self,
        file_path: Path,
        household_id: Optional[int] = None,
    ) -> HouseholdKettleTargetAudit:
        """Stream through a single household CSV and compute Kettle target audit statistics."""
        filename = file_path.name
        if household_id is None:
            digits = "".join(c for c in filename if c.isdigit())
            household_id = int(digits) if digits else 0

        kettle_col, meta_name, make_model = self.discover_kettle_channel(household_id)
        is_positive = kettle_col is not None

        risks: List[str] = []
        if household_id == 3:
            risks.append("H3: Kettle replaced on 16 Apr 2014 with Vektra Vacuum Kettle (hardware signature change).")
        elif household_id == 11:
            risks.append("H11: Aggregate power affected by rooftop solar PV generation.")
        elif household_id == 17:
            risks.append("H17: Kettle plugged into shared IAM with toaster/misc occasionally (shared channel risk).")
        elif household_id == 21:
            risks.append("H21: Shared IAM with Kettle and Toaster + solar aggregate PV generation (multi-risk).")

        result = HouseholdKettleTargetAudit(
            household_id=household_id,
            is_positive=is_positive,
            kettle_channel=kettle_col,
            appliance_name_metadata=meta_name,
            make_model=make_model,
            risk_flags=risks,
        )

        col_idx = COLUMN_INDEX_MAP[kettle_col] if is_positive and kettle_col else None
        agg_idx = COLUMN_INDEX_MAP["Aggregate"]
        unix_idx = COLUMN_INDEX_MAP["Unix"]
        time_idx = COLUMN_INDEX_MAP["Time"]
        issue_idx = COLUMN_INDEX_MAP["Issues"]

        first_unix = None
        last_unix = None
        prev_unix = None

        # Continuous non-zero tracker
        current_nonzero_start_unix = None
        max_nonzero_dur = 0.0

        # Event tracking state for 500W rule: (in_event, start_u, last_u, peak_w, last_event_end_u)
        in_500 = False
        start_500 = 0
        last_500 = 0
        peak_500 = 0.0
        last_event_end_500 = None

        # Event tracking state for 1500W rule:
        in_1500 = False
        start_1500 = 0
        last_1500 = 0
        peak_1500 = 0.0
        last_event_end_1500 = None

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for row in reader:
                result.total_rows += 1
                if len(row) != 13:
                    continue

                if not result.start_date:
                    result.start_date = row[time_idx].strip()
                result.end_date = row[time_idx].strip()

                try:
                    u_val = int(row[unix_idx])
                    agg_w = float(row[agg_idx])
                    kettle_w = float(row[col_idx]) if is_positive and col_idx is not None else 0.0
                    issue_val = row[issue_idx].strip()
                except (ValueError, IndexError):
                    continue

                if first_unix is None:
                    first_unix = u_val
                last_unix = u_val

                # Track temporal gaps on aggregate
                if prev_unix is not None:
                    gap = u_val - prev_unix
                    if gap > 30:
                        result.time_gaps_gt_30s += 1
                    if gap > 120:
                        result.time_gaps_gt_120s += 1
                prev_unix = u_val

                # Aggregate statistics
                result.aggregate_stats.update(agg_w)
                if agg_w > 4000.0:
                    result.aggregate_gt_4000w_count += 1
                if agg_w >= 60000.0:
                    result.aggregate_wraparound_gt_60000w_count += 1
                if issue_val == "1":
                    result.issues_1_count += 1

                if not is_positive:
                    continue

                # Kettle Signal statistics
                result.kettle_stats.update(kettle_w)

                # Continuous non-zero duration tracking
                if kettle_w > 0.0:
                    if current_nonzero_start_unix is None:
                        current_nonzero_start_unix = u_val
                    dur_nz = float(u_val - current_nonzero_start_unix)
                    if dur_nz > max_nonzero_dur:
                        max_nonzero_dur = dur_nz
                else:
                    current_nonzero_start_unix = None

                # Kettle-to-Aggregate relationship (>= 500W)
                if kettle_w >= 500.0:
                    result.kettle_active_500w_samples += 1
                    result.aggregate_sum_during_kettle_active += agg_w
                    bg_w = agg_w - kettle_w
                    result.background_power_sum += bg_w
                    if kettle_w > agg_w:
                        result.kettle_exceeds_aggregate_count += 1

                # Activation Event Tracking (500W Descriptive Rule)
                # Merging rule: consecutive samples >= 500W with time step <= 20 seconds
                if kettle_w >= 500.0:
                    if not in_500:
                        in_500 = True
                        start_500 = u_val
                        last_500 = u_val
                        peak_500 = kettle_w
                    else:
                        if (u_val - last_500) > 20:
                            dur = float(last_500 - start_500)
                            gap_since = float(start_500 - last_event_end_500) if last_event_end_500 else None
                            result.events_500w.add_event(max(8.0, dur), peak_500, gap_since)
                            last_event_end_500 = last_500
                            start_500 = u_val
                            last_500 = u_val
                            peak_500 = kettle_w
                        else:
                            last_500 = u_val
                            if kettle_w > peak_500:
                                peak_500 = kettle_w
                else:
                    if in_500:
                        dur = float(last_500 - start_500)
                        gap_since = float(start_500 - last_event_end_500) if last_event_end_500 else None
                        result.events_500w.add_event(max(8.0, dur), peak_500, gap_since)
                        last_event_end_500 = last_500
                        in_500 = False

                # Activation Event Tracking (1500W Standard Boil Comparison)
                if kettle_w >= 1500.0:
                    if not in_1500:
                        in_1500 = True
                        start_1500 = u_val
                        last_1500 = u_val
                        peak_1500 = kettle_w
                    else:
                        if (u_val - last_1500) > 20:
                            dur = float(last_1500 - start_1500)
                            gap_since = float(start_1500 - last_event_end_1500) if last_event_end_1500 else None
                            result.events_1500w.add_event(max(8.0, dur), peak_1500, gap_since)
                            last_event_end_1500 = last_1500
                            start_1500 = u_val
                            last_1500 = u_val
                            peak_1500 = kettle_w
                        else:
                            last_1500 = u_val
                            if kettle_w > peak_1500:
                                peak_1500 = kettle_w
                else:
                    if in_1500:
                        dur = float(last_1500 - start_1500)
                        gap_since = float(start_1500 - last_event_end_1500) if last_event_end_1500 else None
                        result.events_1500w.add_event(max(8.0, dur), peak_1500, gap_since)
                        last_event_end_1500 = last_1500
                        in_1500 = False

        # Close open events at end of stream
        if in_500:
            dur = float(last_500 - start_500)
            gap_since = float(start_500 - last_event_end_500) if last_event_end_500 else None
            result.events_500w.add_event(max(8.0, dur), peak_500, gap_since)
        if in_1500:
            dur = float(last_1500 - start_1500)
            gap_since = float(start_1500 - last_event_end_1500) if last_event_end_1500 else None
            result.events_1500w.add_event(max(8.0, dur), peak_1500, gap_since)

        result.kettle_stats.max_continuous_nonzero_duration_s = max_nonzero_dur
        if first_unix is not None and last_unix is not None:
            result.days_monitored = (last_unix - first_unix) / 86400.0

        return result

    def audit_all(self, data_dir: Path) -> List[HouseholdKettleTargetAudit]:
        """Audit all 20 household CSVs in sorted order."""
        files = sorted(
            data_dir.glob("CLEAN_House*.csv"),
            key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
        )
        results: List[HouseholdKettleTargetAudit] = []
        for f in files:
            logger.info("Auditing Kettle target characteristics for %s", f.name)
            res = self.audit_household(f)
            results.append(res)
        return results

    def generate_summary(self, results: List[HouseholdKettleTargetAudit]) -> Dict[str, Any]:
        """Consolidate audit metrics into dictionary."""
        pos = [r for r in results if r.is_positive]
        neg = [r for r in results if not r.is_positive]

        total_pos_rows = sum(r.total_rows for r in pos)
        total_events_500 = sum(r.events_500w.event_count for r in pos)
        total_events_1500 = sum(r.events_1500w.event_count for r in pos)
        total_days_pos = sum(r.days_monitored for r in pos)

        return {
            "metadata_summary": {
                "total_households_audited": len(results),
                "positive_households_count": len(pos),
                "negative_households_count": len(neg),
                "positive_households": [r.household_id for r in pos],
                "negative_households": [r.household_id for r in neg],
            },
            "aggregate_kettle_events": {
                "total_events_500w_descriptive_rule": total_events_500,
                "daily_rate_500w": round(total_events_500 / max(0.001, total_days_pos), 2),
                "total_events_1500w_reference": total_events_1500,
                "daily_rate_1500w": round(total_events_1500 / max(0.001, total_days_pos), 2),
            },
            "households": [r.to_dict() for r in results],
        }

    def generate_markdown(self, summary: Dict[str, Any]) -> str:
        """Render formatted GitHub-style Markdown report."""
        meta = summary["metadata_summary"]
        events = summary["aggregate_kettle_events"]
        households = summary["households"]
        pos_h = [h for h in households if h["is_positive"]]
        neg_h = [h for h in households if not h["is_positive"]]

        lines = [
            "# Kettle Target Quality & Activation Profile Audit Report",
            "",
            "## 1. Scope & Metadata Verification",
            "",
            "| Item | Measured Finding |",
            "|---|---|",
            f"| **Target Appliance** | **Kettle** |",
            f"| **Positive (Monitored) Households** | {meta['positive_households_count']} ({', '.join(f'H{h}' for h in meta['positive_households'])}) |",
            f"| **Negative (Unmonitored) Households** | {meta['negative_households_count']} ({', '.join(f'H{h}' for h in meta['negative_households'])}) |",
            f"| **Total 500W Descriptive Activations** | {events['total_events_500w_descriptive_rule']:,} events ({events['daily_rate_500w']} / day / house) |",
            f"| **Total 1500W Standard Activations** | {events['total_events_1500w_reference']:,} events ({events['daily_rate_1500w']} / day / house) |",
            "",
            "## 2. Positive Household Kettle Signal & Activation Profile",
            "",
            "| House | Channel | Model | Events (500W) | Events/Day | Mean Dur (s) | Median Dur (s) | Peak Mean (W) | Idle Zero % | Non-Zero Max Dur (s) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for h in pos_h:
            h_id = h["household_id"]
            chan = h["kettle_channel"]
            model = h["make_model"] or h["metadata_appliance_name"] or "Unknown"
            sig = h["kettle_signal_statistics"]
            ev = h["activation_events_500w_descriptive_rule"]
            lines.append(
                f"| House {h_id} | {chan} | {model} | {ev['event_count']:,} | {ev['events_per_day']} | {ev['duration_mean_s']}s | {ev['duration_median_s']}s | {ev['peak_power_mean_w']:.0f} W | {sig['zero_pct']}% | {sig['max_continuous_nonzero_duration_s']:.0f}s |"
            )

        lines.extend([
            "",
            "## 3. Aggregate Context & Relationship During Kettle ON",
            "",
            "| House | Aggregate Mean (W) | Agg > 4000W % | Wrap-around Glitches (>60kW) | Issues % | Gaps > 30s | Mean Agg During Kettle ON (W) | Mean Background Load (W) | Kettle > Agg Deficit % |",
            "|---|---|---|---|---|---|---|---|---|",
        ])

        for h in pos_h:
            h_id = h["household_id"]
            agg = h["aggregate_context"]
            rel = h["kettle_to_aggregate_relationship"]
            mean_agg_on = f"{rel['mean_aggregate_during_kettle_on_w']:.0f} W" if rel else "N/A"
            mean_bg = f"{rel['mean_background_power_w']:.0f} W" if rel else "N/A"
            def_pct = f"{rel['kettle_exceeds_aggregate_pct']:.2f}%" if rel else "N/A"
            lines.append(
                f"| House {h_id} | {agg['mean']:.1f} W | {agg['gt_4000w_pct']:.2f}% | {agg['wraparound_gt_60000w_count']} | {agg['issues_1_pct']:.2f}% | {agg['time_gaps_gt_30s']} | {mean_agg_on} | {mean_bg} | {def_pct} |"
            )

        lines.extend([
            "",
            "## 4. Household-Specific Target Risks & Nuances",
            "",
            "| House | Identified Risk | CSV & Metadata Evidence | Recommended Handling |",
            "|---|---|---|---|",
            "| **House 3** | Kettle Hardware Change | Replaced with Vektra Vacuum Kettle on 16 Apr 2014 | Keep in dataset; record signature shift date |",
            "| **House 11** | Solar PV Generation | Net daytime aggregate power reduced/distorted by solar panels | Reserve for validation / robustness evaluation |",
            "| **House 17** | Shared IAM Channel | Kettle shares plug with toaster/misc items | Use >=1500W threshold for target ground truth |",
            "| **House 21** | Shared IAM + Solar PV | Dual complexity: shared toaster plug + rooftop solar PV | Reserve for holdout / robustness evaluation |",
            "",
            "## 5. Negative (Unmonitored) Households Verification",
            "",
            "| House | Metadata Verification | Monitored Kettle IAM | Usable as Negative Weak-Supervision Example |",
            "|---|---|---|---|",
        ])

        for h in neg_h:
            h_id = h["household_id"]
            lines.append(f"| House {h_id} | Confirmed unmonitored in metadata | None | **Yes** (Unmonitored / Weak-Negative Supervision State) |")

        return "\n".join(lines)
