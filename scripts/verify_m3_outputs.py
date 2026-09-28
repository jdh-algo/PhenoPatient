"""Independently check frozen-case coverage after the original M3 runner exits."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath


SOURCE = Path(__file__).resolve().parents[1] / "src" / "phenopatient"
sys.path.insert(0, str(SOURCE))
import utils  # noqa: E402
import virtual_clinical_interaction as m3  # noqa: E402


def verify_m3_outputs(bundle_root: Path, runtime_root: Path,
                      expected_patients: int | None = None) -> dict:
    """Require every frozen case to have an interaction and matching ledger metadata."""
    bundle_root, runtime_root = Path(bundle_root), Path(runtime_root)
    expected = defaultdict(set)
    seen_case_ids = set()
    patient_file = bundle_root / "data/final_patients/phenopatient_1000.csv"
    with patient_file.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id") or "").strip()
            raw_group = str(row.get("source_group") or "")
            group = PurePosixPath(raw_group)
            if (not case_id or not raw_group or group.is_absolute() or "\\" in raw_group
                    or any(part in (".", "..") for part in raw_group.split("/"))
                    or case_id in seen_case_ids):
                raise ValueError("invalid or duplicate frozen case_id/source_group")
            expected[group].add(case_id)
            seen_case_ids.add(case_id)
    total = sum(len(ids) for ids in expected.values())
    if not total or (expected_patients is not None and total != expected_patients):
        raise ValueError(f"frozen patient count mismatch: {total} != {expected_patients}")

    runtime_csvs = set(runtime_root.rglob("*.csv"))
    required_csvs = {runtime_root / f"{group.as_posix()}.csv" for group in expected}
    if runtime_csvs != required_csvs:
        raise ValueError(
            f"runtime CSV set differs from frozen groups: found={len(runtime_csvs)} "
            f"expected={len(required_csvs)}"
        )

    complete = 0
    incomplete = []
    for group, ids in sorted(expected.items(), key=lambda item: item[0].as_posix()):
        csv_path = runtime_root / f"{group.as_posix()}.csv"
        sidecar_dir = Path(str(csv_path) + ".module_io")
        if not sidecar_dir.is_dir():
            raise ValueError(f"runtime sidecar directory missing: {group}")
        template, rows = utils._load_existing_csv(str(csv_path))
        if template is None or not template.get(utils.COL_STAGING_SYSTEM, "").strip():
            raise ValueError(f"runtime CSV unreadable or M1 template missing: {group}")
        case_ids = [str(row.get("case_id") or "").strip() for row in rows.values()]
        if len(case_ids) != len(ids) or set(case_ids) != ids:
            raise ValueError(f"runtime case_id coverage differs from frozen patients: {group}")
        staging = utils.parse_staging_system(template[utils.COL_STAGING_SYSTEM])
        for index, row in rows.items():
            case_id = row["case_id"]
            interaction = row.get(utils.COL_INTERACTION, "") or ""
            if not interaction.strip() or "（诊断生成失败）" in interaction[-2000:]:
                incomplete.append(case_id)
                continue
            try:
                current = m3._m3_row_has_current_ledger_metadata(
                    str(csv_path), index, row, staging
                )
            except (ValueError, KeyError, OSError):
                current = False
            if current:
                complete += 1
            else:
                incomplete.append(case_id)
    if incomplete:
        raise ValueError(
            f"M3 coverage incomplete: complete={complete}/{total}; "
            f"sample_case_ids={incomplete[:5]}"
        )
    return {
        "disease_groups": len(expected),
        "expected_patients": total,
        "complete_patients": complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=1000)
    args = parser.parse_args()
    summary = verify_m3_outputs(args.bundle_root, args.runtime_root, args.expected_patients)
    print(f"M3_OUTPUTS_OK {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
