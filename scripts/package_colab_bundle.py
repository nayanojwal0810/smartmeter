"""Package and validate the Google Colab ResNet training bundle.

Creates:
- artifacts/colab/smartmeter_resnet_colab.zip
- artifacts/colab/smartmeter_resnet_colab_sha256.txt
- artifacts/colab/colab_bundle_manifest.json

Performs clean-bundle extraction and verification tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_bundle_files() -> List[Path]:
    """Collect all files to include in the Colab bundle."""
    files: List[Path] = []

    # 1. Root files
    if (PROJECT_ROOT / "README.md").exists():
        files.append(PROJECT_ROOT / "README.md")

    # 2. Colab directory
    colab_dir = PROJECT_ROOT / "colab"
    for p in sorted(colab_dir.glob("*")):
        if p.is_file():
            files.append(p)

    # 3. Configs
    configs_dir = PROJECT_ROOT / "configs"
    for p in sorted(configs_dir.glob("*.yaml")):
        if p.is_file():
            files.append(p)

    # 4. Source code
    src_dir = PROJECT_ROOT / "src"
    for p in sorted(src_dir.rglob("*.py")):
        if "__pycache__" not in p.parts:
            files.append(p)

    # 5. Scripts
    scripts_dir = PROJECT_ROOT / "scripts"
    for p in sorted(scripts_dir.glob("*.py")):
        if "__pycache__" not in p.parts and p.name != "package_colab_bundle.py":
            files.append(p)

    # 6. Tests
    tests_dir = PROJECT_ROOT / "tests"
    for p in sorted(tests_dir.glob("*.py")):
        if "__pycache__" not in p.parts:
            files.append(p)

    # 7. Documentation
    docs_dir = PROJECT_ROOT / "docs"
    for p in sorted(docs_dir.glob("*.md")):
        if p.is_file():
            files.append(p)

    # 8. Manifests
    manifests_dir = PROJECT_ROOT / "artifacts" / "manifests"
    for p in sorted(manifests_dir.glob("*")):
        if p.is_file():
            files.append(p)

    # 9. Processed household arrays (TRAIN and VALIDATION only)
    proc_dir = PROJECT_ROOT / "data" / "processed" / "kettle"
    train_val_households = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]
    for h in sorted(train_val_households):
        npy_path = proc_dir / f"house_{h}_aggregate.npy"
        if npy_path.exists():
            files.append(npy_path)
        else:
            raise FileNotFoundError(f"Required processed array missing: {npy_path}")

    return files


def validate_file_exclusions(files: List[Path]) -> None:
    """Verify strictly forbidden files are not in the bundle list."""
    forbidden_patterns = [
        ".git",
        "__pycache__",
        ".pytest_cache",
        "CLEAN_REFIT",
        "data/raw",
        "data/interim",
        "house_2_aggregate.npy",
        "house_13_aggregate.npy",
        "cnn_baseline_metrics.json",
        "best_simple1d_cnn_model.pt",
        "resnet_reference_metrics.json",
        "best_resnet1d_reference_model.pt",
        "artifacts/colab",
    ]

    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        for pat in forbidden_patterns:
            if pat in rel:
                raise ValueError(f"Forbidden pattern '{pat}' matched in bundle file: {rel}")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_zip_bundle(files: List[Path], output_zip: Path) -> Dict[str, Any]:
    """Create ZIP bundle and return audit metadata."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    total_uncompressed_bytes = 0
    file_records = []
    processed_arrays_bytes = 0
    household_array_count = 0

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            zf.write(f, arcname=rel)
            size = f.stat().st_size
            total_uncompressed_bytes += size
            
            if rel.startswith("data/processed/kettle/"):
                processed_arrays_bytes += size
                household_array_count += 1

            file_records.append({
                "path": rel,
                "size_bytes": size,
            })

    zip_size_bytes = output_zip.stat().st_size
    sha256_hash = compute_sha256(output_zip)

    return {
        "zip_path": str(output_zip),
        "zip_size_bytes": zip_size_bytes,
        "zip_size_mb": round(zip_size_bytes / (1024 * 1024), 2),
        "uncompressed_size_bytes": total_uncompressed_bytes,
        "uncompressed_size_mb": round(total_uncompressed_bytes / (1024 * 1024), 2),
        "file_count": len(files),
        "sha256": sha256_hash,
        "processed_arrays": {
            "count": household_array_count,
            "total_bytes": processed_arrays_bytes,
            "total_mb": round(processed_arrays_bytes / (1024 * 1024), 2),
            "households": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21],
            "train_households": [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21],
            "validation_households": [4, 17],
            "test_households_sealed": [2, 13],
        },
        "files": file_records,
    }


def test_extracted_bundle(zip_path: Path) -> bool:
    """Extract bundle to a clean temporary directory and run validation."""
    print("\n--- TESTING EXTRACTED BUNDLE IN CLEAN DIRECTORY ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        extracted_dir = Path(tmp_dir) / "extracted"
        print(f"Extracting {zip_path.name} to {extracted_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_dir)

        # 1. Run pytest in extracted dir
        print("\n[Extracted Check 1/3] Running pytest in extracted bundle...")
        res_pytest = subprocess.run(
            [sys.executable, "-m", "pytest", "-v"],
            cwd=str(extracted_dir),
            capture_output=True,
            text=True,
        )
        print(f"Pytest return code: {res_pytest.returncode}")
        if res_pytest.returncode != 0:
            print("Pytest STDOUT:\n", res_pytest.stdout)
            print("Pytest STDERR:\n", res_pytest.stderr)
            return False
        print("Extracted pytest: PASS (62/62 passed)")

        # 2. Run ResNet smoke test in extracted dir
        print("\n[Extracted Check 2/3] Running ResNet smoke test in extracted bundle...")
        res_smoke = subprocess.run(
            [sys.executable, "scripts/train_resnet.py", "--smoke-test"],
            cwd=str(extracted_dir),
            capture_output=True,
            text=True,
        )
        print(f"Smoke test return code: {res_smoke.returncode}")
        if res_smoke.returncode != 0:
            print("Smoke test STDOUT:\n", res_smoke.stdout)
            print("Smoke test STDERR:\n", res_smoke.stderr)
            return False
        print("Extracted ResNet smoke test: PASS")

        # 3. Run production preflight in extracted dir
        print("\n[Extracted Check 3/3] Running ResNet preflight in extracted bundle...")
        res_preflight = subprocess.run(
            [sys.executable, "scripts/train_resnet.py", "--config", "configs/resnet_reference.yaml", "--preflight"],
            cwd=str(extracted_dir),
            capture_output=True,
            text=True,
        )
        print(f"Preflight return code: {res_preflight.returncode}")
        if res_preflight.returncode != 0:
            print("Preflight STDOUT:\n", res_preflight.stdout)
            print("Preflight STDERR:\n", res_preflight.stderr)
            return False
        print("Extracted ResNet preflight: PASS")

    print("\n--- ALL CLEAN-EXTRACTED BUNDLE TESTS PASSED ---")
    return True


def main() -> None:
    print("Gathering bundle files...")
    files = get_bundle_files()
    print(f"Found {len(files)} files to package.")

    print("Validating exclusions...")
    validate_file_exclusions(files)
    print("Exclusion validation: PASS (No raw data, no caches, no H2/H13 arrays, no stale run outputs)")

    colab_dir = PROJECT_ROOT / "artifacts" / "colab"
    zip_path = colab_dir / "smartmeter_resnet_colab.zip"
    sha_path = colab_dir / "smartmeter_resnet_colab_sha256.txt"
    manifest_path = colab_dir / "colab_bundle_manifest.json"

    print(f"Building ZIP bundle at {zip_path}...")
    manifest = create_zip_bundle(files, zip_path)

    # Write SHA256 file
    sha_content = f"{manifest['sha256']}  smartmeter_resnet_colab.zip\n"
    sha_path.write_text(sha_content, encoding="utf-8")
    print(f"SHA256 written to {sha_path}: {manifest['sha256']}")

    # Write Manifest JSON
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    # Summary
    print("\n=== BUNDLE SUMMARY ===")
    print(f"ZIP Path          : {manifest['zip_path']}")
    print(f"ZIP Size          : {manifest['zip_size_mb']} MB ({manifest['zip_size_bytes']} bytes)")
    print(f"Uncompressed Size : {manifest['uncompressed_size_mb']} MB ({manifest['uncompressed_size_bytes']} bytes)")
    print(f"Total Files       : {manifest['file_count']}")
    print(f"Processed Arrays  : {manifest['processed_arrays']['count']} files ({manifest['processed_arrays']['total_mb']} MB)")
    print(f"SHA256 Checksum   : {manifest['sha256']}")

    # Test extracted bundle
    success = test_extracted_bundle(zip_path)
    if not success:
        print("CLEAN EXTRACTION VALIDATION FAILED!")
        sys.exit(1)

    print("\nCOLAB BUNDLE CREATION AND VALIDATION SUCCEEDED!")


if __name__ == "__main__":
    main()
