"""Materialize original-M3-compatible CSVs and sidecars without model calls."""

import argparse
import csv
import gzip
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


SOURCE = Path(__file__).resolve().parents[1] / "src" / "phenopatient"
sys.path.insert(0, str(SOURCE))
import patient_fact_ledger as pfl  # noqa: E402
import utils  # noqa: E402


def _gzip_records(path: Path) -> dict:
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            case_id = str(record.get("case_id") or "").strip()
            if not case_id or case_id in records:
                raise ValueError(f"empty or duplicate case_id in {path.name}: {case_id!r}")
            records[case_id] = record
    return records


def _safe_group(raw: str) -> PurePosixPath:
    group = PurePosixPath(raw)
    if not raw or group.is_absolute() or any(part in (".", "..") for part in raw.split("/")) or "\\" in raw:
        raise ValueError(f"unsafe source_group: {raw!r}")
    return group


def prepare_m3_runtime(bundle_root: Path, output_root: Path,
                       expected_patients: int | None = None) -> dict:
    """Restore native M3 inputs; reject mismatched case IDs and source hashes."""
    bundle_root, output_root = Path(bundle_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    data = bundle_root / "data"
    patients = defaultdict(list)
    patient_csv = data / "final_patients" / "phenopatient_1000.csv"
    with patient_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id") or "").strip()
            if not case_id:
                raise ValueError("patient CSV has empty case_id")
            group = _safe_group(str(row.get("source_group") or ""))
            patients[group].append(row)
    patient_ids = [row["case_id"] for rows in patients.values() for row in rows]
    if not patient_ids or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("patient CSV is empty or contains duplicate case_id")
    if expected_patients is not None and len(patient_ids) != expected_patients:
        raise ValueError(f"patient count mismatch: {len(patient_ids)} != {expected_patients}")

    ledgers = _gzip_records(data / "final_ledgers" / "phenopatient_1000_fact_ledgers.jsonl.gz")
    m25 = _gzip_records(data / "m25_time_models" / "phenopatient_1000_m25.jsonl.gz")
    if set(patient_ids) != set(ledgers) or set(patient_ids) != set(m25):
        raise ValueError("patient, ledger, and M2.5 case_id sets differ")
    atlas_root = data / "m1_atlases"
    atlas_files = set(atlas_root.rglob("*.csv"))
    expected_atlases = {atlas_root / f"{group.as_posix()}.csv" for group in patients}
    if atlas_files != expected_atlases:
        raise ValueError("M1 atlas files do not match patient source_group set")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m3-prepare-", dir=output_root.parent) as temporary:
        stage = Path(temporary) / "runtime"
        for group, rows in sorted(patients.items(), key=lambda item: item[0].as_posix()):
            atlas_path = atlas_root / f"{group.as_posix()}.csv"
            with atlas_path.open(encoding="utf-8-sig", newline="") as handle:
                atlas_rows = list(csv.DictReader(handle))
            if len(atlas_rows) != 1 or atlas_rows[0].get("case_id"):
                raise ValueError(f"M1 atlas must have one template row: {group}")
            template = utils._empty_row()
            template.update({key: value for key, value in atlas_rows[0].items() if key in template})
            csv_path = stage / f"{group.as_posix()}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            sorted_rows = sorted(rows, key=lambda row: row["case_id"])
            result_rows = [template]
            for row in sorted_rows:
                case_id = row["case_id"]
                if m25[case_id].get("source_group") != group.as_posix():
                    raise ValueError(f"M2.5 source_group mismatch: {case_id}")
                patient = utils._empty_row()
                patient.update({key: value for key, value in row.items() if key in patient})
                patient[utils.COL_M25_OUTPUT] = json.dumps(
                    m25[case_id]["time_model"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
                result_rows.append(patient)
            utils._save_csv(str(csv_path), result_rows)
            for index, row in enumerate(sorted_rows, start=1):
                pfl.write_fact_ledger(str(csv_path), index, ledgers[row["case_id"]])
            row_1, restored = utils._load_existing_csv(str(csv_path))
            staging = utils.parse_staging_system(row_1[utils.COL_STAGING_SYSTEM])
            for index, row in restored.items():
                try:
                    pfl.load_verified_fact_ledger_for_m3(str(csv_path), index, row, staging)
                except ValueError as exc:
                    raise ValueError(f"{row.get('case_id')}: {exc}") from exc
        stage.rename(output_root)
    return {"disease_groups": len(patients), "patients": len(patient_ids), "ledgers": len(ledgers)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=1000)
    args = parser.parse_args()
    summary = prepare_m3_runtime(args.bundle_root, args.output_root,
                                 expected_patients=args.expected_patients)
    print(f"M3_INPUTS_OK {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
