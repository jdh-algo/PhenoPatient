"""Export only the M2.5 time models from a completed M2 run, without prompts."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path


M25_OUTPUT = "模块2.5_输出"
TIME_MODEL_FIELDS = {
    "schema_version", "underlying_duration", "current_episode_duration", "symptom_timeline",
}
TIMELINE_FIELDS = {"name", "category", "phase", "time_label"}


def _validate_public_time_model(time_model: object) -> None:
    """Fail closed on unexpected fields from a private run's module I/O."""
    if not isinstance(time_model, dict) or set(time_model) != TIME_MODEL_FIELDS:
        raise ValueError("unexpected time model fields")
    if type(time_model["schema_version"]) is not int or time_model["schema_version"] != 2:
        raise ValueError("invalid schema_version")
    if (time_model["underlying_duration"] is not None
            and not isinstance(time_model["underlying_duration"], str)):
        raise ValueError("invalid underlying_duration")
    if not isinstance(time_model["current_episode_duration"], str):
        raise ValueError("invalid current_episode_duration")
    timeline = time_model["symptom_timeline"]
    if not isinstance(timeline, list):
        raise ValueError("invalid symptom_timeline")
    for entry in timeline:
        if isinstance(entry, dict):
            if set(entry) != TIMELINE_FIELDS:
                raise ValueError("unexpected timeline fields")
            values = [entry[key] for key in ("name", "category", "phase", "time_label")]
        elif isinstance(entry, list) and len(entry) == 4:
            values = entry
        else:
            raise ValueError("invalid timeline entry")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("invalid timeline values")
        if values[1] != "症状" or values[2] not in {"起病", "就诊"}:
            raise ValueError("invalid timeline category or phase")


def export_m25_time_models(source_m2: Path, sidecar_root: Path, output: Path,
                           expected_count: int | None = None) -> int:
    source_m2, sidecar_root, output = map(Path, (source_m2, sidecar_root, output))
    records = []
    seen = set()
    for csv_path in sorted(source_m2.rglob("*.csv")):
        relative = csv_path.relative_to(source_m2)
        sidecar_dir = sidecar_root / relative.parent / f"{relative.name}.module_io"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                case_id = str(row.get("case_id") or "").strip()
                if not case_id:
                    continue
                if case_id in seen:
                    raise ValueError(f"duplicate case_id: {case_id}")
                seen.add(case_id)
                sidecar = sidecar_dir / f"row_{index}.json"
                try:
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    raw = payload[M25_OUTPUT]
                    time_model = json.loads(raw) if isinstance(raw, str) else raw
                    _validate_public_time_model(time_model)
                except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{case_id}: M2.5 time model missing or invalid") from exc
                records.append({
                    "case_id": case_id,
                    "source_group": relative.with_suffix("").as_posix(),
                    "time_model": time_model,
                })
    if not records:
        raise ValueError("M2 source has no patient rows")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"M2.5 record count mismatch: {len(records)} != {expected_count}")
    records.sort(key=lambda item: item["case_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as zipped:
                for record in records:
                    zipped.write((json.dumps(record, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")) + "\n").encode("utf-8"))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-m2", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    args = parser.parse_args()
    count = export_m25_time_models(args.source_m2, args.sidecar_root, args.output,
                                   expected_count=args.expected_count)
    print(f"M2.5_EXPORT_OK count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
