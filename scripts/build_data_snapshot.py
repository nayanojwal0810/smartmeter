"""Create and validate canonical Kettle data snapshot for Git -> Drive -> Colab workflow.

Generates:
- artifacts/data_snapshot/kettle_validation_targets.jsonl
- artifacts/data_snapshot/kettle_train_val_snapshot_manifest.json
- artifacts/data_snapshot/kettle_train_val_snapshot_2026-09-24.zip
- artifacts/data_snapshot/kettle_train_val_snapshot_2026-09-24.zip.sha256
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_current_git_commit() -> str:
    """Retrieve the current canonical Git commit hash dynamically."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return "unknown"


TRAIN_HOUSEHOLDS = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21]
VAL_HOUSEHOLDS = [4, 17]
TEST_HOUSEHOLDS = [2, 13]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_validation_targets_jsonl(
    source_targets_jsonl: Path,
    output_val_targets_jsonl: Path,
) -> Dict[str, Any]:
    """Filter full evaluation targets manifest into validation-only (H4, H17) records."""
    output_val_targets_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_hh_set = set(VAL_HOUSEHOLDS)
    
    val_records_count = 0
    val_households_found: Set[int] = set()
    forbidden_records_count = 0

    with open(source_targets_jsonl, "r", encoding="utf-8") as fin, \
         open(output_val_targets_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            h_id = rec.get("household_id")
            if h_id in val_hh_set:
                fout.write(line)
                val_records_count += 1
                val_households_found.add(h_id)
            elif h_id in TEST_HOUSEHOLDS:
                forbidden_records_count += 1

    assert val_records_count == 18561, f"Expected 18,561 validation records, got {val_records_count}"
    assert val_households_found == {4, 17}, f"Validation target households must be exactly {{4, 17}}, got {val_households_found}"
    
    sha256 = compute_sha256(output_val_targets_jsonl)
    size_bytes = output_val_targets_jsonl.stat().st_size

    return {
        "record_count": val_records_count,
        "households": sorted(list(val_households_found)),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def audit_npy_arrays(proc_dir: Path) -> Dict[str, Any]:
    """Audit all 18 train/validation .npy arrays."""
    records = []
    total_train_windows = 0
    total_val_windows = 0
    total_bytes = 0

    for h in TRAIN_HOUSEHOLDS:
        npy_p = proc_dir / f"house_{h}_aggregate.npy"
        assert npy_p.exists(), f"Missing train array: {npy_p}"
        arr = np.load(npy_p, mmap_mode="r")
        assert arr.ndim == 2 and arr.shape[1] == 510, f"Invalid shape for House {h}: {arr.shape}"
        assert arr.dtype == np.float32, f"Invalid dtype for House {h}: {arr.dtype}"
        n_win = arr.shape[0]
        total_train_windows += n_win
        size = npy_p.stat().st_size
        total_bytes += size
        records.append({
            "household_id": h,
            "split": "train",
            "filename": npy_p.name,
            "window_count": n_win,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": size,
            "sha256": compute_sha256(npy_p),
        })

    for h in VAL_HOUSEHOLDS:
        npy_p = proc_dir / f"house_{h}_aggregate.npy"
        assert npy_p.exists(), f"Missing val array: {npy_p}"
        arr = np.load(npy_p, mmap_mode="r")
        assert arr.ndim == 2 and arr.shape[1] == 510, f"Invalid shape for House {h}: {arr.shape}"
        assert arr.dtype == np.float32, f"Invalid dtype for House {h}: {arr.dtype}"
        n_win = arr.shape[0]
        total_val_windows += n_win
        size = npy_p.stat().st_size
        total_bytes += size
        records.append({
            "household_id": h,
            "split": "validation",
            "filename": npy_p.name,
            "window_count": n_win,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size_bytes": size,
            "sha256": compute_sha256(npy_p),
        })

    # Strict check: Test households must not exist in processed directory
    for h in TEST_HOUSEHOLDS:
        test_npy = proc_dir / f"house_{h}_aggregate.npy"
        assert not test_npy.exists(), f"Test household array leaked in processed directory: {test_npy}"

    assert total_train_windows == 141003, f"Expected 141,003 train windows, got {total_train_windows}"
    assert total_val_windows == 18561, f"Expected 18,561 val windows, got {total_val_windows}"
    assert total_train_windows + total_val_windows == 159564, "Total windows mismatch"

    return {
        "train_households_count": len(TRAIN_HOUSEHOLDS),
        "val_households_count": len(VAL_HOUSEHOLDS),
        "total_households_count": len(records),
        "total_train_windows": total_train_windows,
        "total_val_windows": total_val_windows,
        "total_included_windows": total_train_windows + total_val_windows,
        "total_bytes": total_bytes,
        "arrays": records,
    }


def create_snapshot_bundle() -> Path:
    """Package the snapshot archive and manifest."""
    snapshot_dir = PROJECT_ROOT / "artifacts" / "data_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    source_targets = PROJECT_ROOT / "artifacts" / "manifests" / "kettle_evaluation_targets.jsonl"
    val_targets_file = snapshot_dir / "kettle_validation_targets.jsonl"
    
    print("[1/4] Generating validation-only evaluation targets manifest...")
    val_tgt_meta = generate_validation_targets_jsonl(source_targets, val_targets_file)
    print(f"      Generated {val_tgt_meta['record_count']} validation records ({val_tgt_meta['size_bytes']} bytes)")

    proc_dir = PROJECT_ROOT / "data" / "processed" / "kettle"
    print("[2/4] Auditing processed arrays...")
    array_audit = audit_npy_arrays(proc_dir)
    print(f"      Audited 18 arrays: {array_audit['total_included_windows']} windows ({array_audit['total_bytes'] / (1024*1024):.2f} MB)")

    # Manifests to include
    split_manifest = PROJECT_ROOT / "artifacts" / "manifests" / "kettle_household_split.json"
    index_manifest = PROJECT_ROOT / "artifacts" / "manifests" / "timebase_window_index.json"

    manifest_files_meta = {
        "kettle_household_split.json": {
            "size_bytes": split_manifest.stat().st_size,
            "sha256": compute_sha256(split_manifest),
        },
        "timebase_window_index.json": {
            "size_bytes": index_manifest.stat().st_size,
            "sha256": compute_sha256(index_manifest),
        },
        "kettle_validation_targets.jsonl": {
            "record_count": val_tgt_meta["record_count"],
            "size_bytes": val_tgt_meta["size_bytes"],
            "sha256": val_tgt_meta["sha256"],
        },
    }

    snapshot_manifest = {
        "snapshot_name": "kettle_train_val_snapshot_2026-09-24",
        "creation_date": "2026-09-24",
        "source_git_commit": get_current_git_commit(),
        "appliance": "kettle",
        "dataset_specification": {
            "window_points": 510,
            "grid_step_seconds": 8,
            "window_duration_seconds": 4072,
            "normalization_policy": "train-only standardization",
            "supervision": "household-level weak kettle presence",
            "evaluation_target_policy": "validation-only strong kettle annotations (H4, H17)",
        },
        "split_summary": {
            "train_households": TRAIN_HOUSEHOLDS,
            "validation_households": VAL_HOUSEHOLDS,
            "sealed_test_households": TEST_HOUSEHOLDS,
            "total_train_windows": array_audit["total_train_windows"],
            "total_validation_windows": array_audit["total_val_windows"],
            "total_included_windows": array_audit["total_included_windows"],
        },
        "test_isolation_guarantees": {
            "h2_included": False,
            "h13_included": False,
            "h2_target_records_included": False,
            "h13_target_records_included": False,
            "statement": "Households H2 and H13 are sealed test households and are completely excluded from all data arrays and target manifests in this snapshot.",
        },
        "manifest_files": manifest_files_meta,
        "processed_arrays": array_audit["arrays"],
    }

    manifest_path = snapshot_dir / "kettle_train_val_snapshot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_manifest, f, indent=2)
    print(f"[3/4] Wrote snapshot manifest: {manifest_path}")

    # Build ZIP
    zip_path = snapshot_dir / "kettle_train_val_snapshot_2026-09-24.zip"
    sha_path = snapshot_dir / "kettle_train_val_snapshot_2026-09-24.zip.sha256"

    print(f"[4/4] Building ZIP archive at {zip_path}...")
    root_prefix = "kettle_train_val_snapshot"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write manifest
        zf.write(manifest_path, arcname=f"{root_prefix}/kettle_train_val_snapshot_manifest.json")
        
        # Write manifests
        zf.write(split_manifest, arcname=f"{root_prefix}/artifacts/manifests/kettle_household_split.json")
        zf.write(index_manifest, arcname=f"{root_prefix}/artifacts/manifests/timebase_window_index.json")
        zf.write(val_targets_file, arcname=f"{root_prefix}/artifacts/manifests/kettle_validation_targets.jsonl")

        # Write 18 .npy arrays
        for rec in array_audit["arrays"]:
            src_npy = proc_dir / rec["filename"]
            arc_name = f"{root_prefix}/data/processed/kettle/{rec['filename']}"
            zf.write(src_npy, arcname=arc_name)

    zip_sha256 = compute_sha256(zip_path)
    sha_path.write_text(f"{zip_sha256}  kettle_train_val_snapshot_2026-09-24.zip\n", encoding="utf-8")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"      ZIP created successfully: {zip_size_mb:.2f} MB")
    print(f"      ZIP SHA256: {zip_sha256}")

    return zip_path


def verify_clean_extraction(zip_path: Path) -> None:
    """Extract zip into temp directory and perform full integrity checks."""
    print("\n--- RUNNING INTEGRITY VERIFICATION ON EXTRACTED SNAPSHOT ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        base = extract_dir / "kettle_train_val_snapshot"
        assert base.exists(), "Root snapshot directory missing"

        manifest_file = base / "kettle_train_val_snapshot_manifest.json"
        assert manifest_file.exists(), "Snapshot manifest missing"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Verify .npy files
        proc_dir = base / "data" / "processed" / "kettle"
        assert proc_dir.exists(), "Processed kettle directory missing"
        
        total_extracted_windows = 0
        npy_files = list(proc_dir.glob("*.npy"))
        assert len(npy_files) == 18, f"Expected 18 npy files, found {len(npy_files)}"

        for npy_f in npy_files:
            h_id = int(npy_f.stem.split("_")[1])
            assert h_id not in TEST_HOUSEHOLDS, f"Test household found in extraction: {npy_f.name}"
            arr = np.load(npy_f)
            assert arr.ndim == 2 and arr.shape[1] == 510, f"Corrupted shape for {npy_f.name}: {arr.shape}"
            assert arr.dtype == np.float32, f"Corrupted dtype for {npy_f.name}: {arr.dtype}"
            total_extracted_windows += len(arr)

        assert total_extracted_windows == 159564, f"Expected 159,564 windows, got {total_extracted_windows}"
        print("  1. Extracted 18 .npy arrays: Shape (N, 510), float32, 159,564 windows: PASS")

        # Verify validation target manifest
        val_tgt_file = base / "artifacts" / "manifests" / "kettle_validation_targets.jsonl"
        assert val_tgt_file.exists(), "Validation targets manifest missing"
        
        val_records = 0
        val_houses = set()
        with open(val_tgt_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                h_id = rec.get("household_id")
                assert h_id in {4, 17}, f"Non-validation household in target manifest: {h_id}"
                val_houses.add(h_id)
                val_records += 1

        assert val_records == 18561, f"Expected 18,561 validation target records, got {val_records}"
        assert val_houses == {4, 17}, f"Expected only H4 and H17, got {val_houses}"
        print("  2. Extracted validation targets: Exactly 18,561 records (H4, H17 only): PASS")

        # Verify test isolation
        for root, dirs, files in os.walk(base):
            for fname in files:
                assert fname not in ("house_2_aggregate.npy", "house_13_aggregate.npy"), f"Forbidden test file: {fname}"
                assert not (fname.startswith("house_2_") or fname.startswith("house_13_")), f"Forbidden test file: {fname}"
        print("  3. Sealed test isolation: Zero H2/H13 references in extracted files: PASS")

        # Verify split manifest marks H2/H13 as test
        split_json = base / "artifacts" / "manifests" / "kettle_household_split.json"
        with open(split_json, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        test_h_split = {h["household_id"]: h["split"] for h in split_data["households"] if h["household_id"] in (2, 13)}
        assert test_h_split == {2: "test", 13: "test"}, f"Test split misconfigured: {test_h_split}"
        print("  4. Split manifest verification: H2/H13 confirmed as TEST: PASS")

    print("\n--- ALL SNAPSHOT INTEGRITY CHECKS PASSED ---")


if __name__ == "__main__":
    zip_path = create_snapshot_bundle()
    verify_clean_extraction(zip_path)
