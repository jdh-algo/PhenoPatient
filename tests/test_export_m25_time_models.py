import csv
import gzip
import json

import pytest

from scripts.export_m25_time_models import export_m25_time_models


def _source_fixture(tmp_path, *, include_second_sidecar=True):
    source = tmp_path / "M2"
    source.mkdir()
    csv_path = source / "disease.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "模块2.5_输出"])
        writer.writeheader()
        writer.writerow({"case_id": ""})
        writer.writerow({"case_id": "case_b", "模块2.5_输出": "@module_io/row_1.json"})
        writer.writerow({"case_id": "case_a", "模块2.5_输出": "@module_io/row_2.json"})
    sidecars = tmp_path / "sidecars" / "disease.csv.module_io"
    sidecars.mkdir(parents=True)
    time_model = {
        "schema_version": 2,
        "underlying_duration": "三天",
        "current_episode_duration": "三天",
        "symptom_timeline": [],
    }
    for index in (1, 2) if include_second_sidecar else (1,):
        (sidecars / f"row_{index}.json").write_text(
            json.dumps({
                "模块2.5_输出": json.dumps(time_model, ensure_ascii=False),
                "模块2.3_输入": "private-marker-must-not-ship",
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    return source, tmp_path / "sidecars"


def test_export_keeps_only_case_key_and_time_model_in_stable_order(tmp_path):
    source, sidecars = _source_fixture(tmp_path)
    output = tmp_path / "m25.jsonl.gz"

    assert export_m25_time_models(source, sidecars, output) == 2
    first_bytes = output.read_bytes()
    assert export_m25_time_models(source, sidecars, output) == 2
    assert output.read_bytes() == first_bytes
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert [row["case_id"] for row in rows] == ["case_a", "case_b"]
    assert all(row["source_group"] == "disease" for row in rows)
    assert all(row["time_model"]["schema_version"] == 2 for row in rows)
    assert "private-marker-must-not-ship" not in "".join(map(json.dumps, rows))
    assert all(set(row) == {"case_id", "source_group", "time_model"} for row in rows)


def test_export_rejects_missing_patient_sidecar_without_output(tmp_path):
    source, sidecars = _source_fixture(tmp_path, include_second_sidecar=False)
    output = tmp_path / "m25.jsonl.gz"

    with pytest.raises(ValueError, match="case_a.*M2.5"):
        export_m25_time_models(source, sidecars, output)

    assert not output.exists()


def test_export_count_mismatch_preserves_existing_output(tmp_path):
    source, sidecars = _source_fixture(tmp_path)
    output = tmp_path / "m25.jsonl.gz"
    output.write_bytes(b"previous release")

    with pytest.raises(ValueError, match="record count mismatch"):
        export_m25_time_models(source, sidecars, output, expected_count=3)

    assert output.read_bytes() == b"previous release"


@pytest.mark.parametrize("extra_field", ["prompt", "api_key"])
def test_export_rejects_unexpected_time_model_fields(tmp_path, extra_field):
    source, sidecars = _source_fixture(tmp_path)
    path = sidecars / "disease.csv.module_io" / "row_1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = json.loads(payload["模块2.5_输出"])
    model[extra_field] = "private-marker-must-not-ship"
    payload["模块2.5_输出"] = json.dumps(model, ensure_ascii=False)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output = tmp_path / "m25.jsonl.gz"
    with pytest.raises(ValueError, match="case_b.*M2.5"):
        export_m25_time_models(source, sidecars, output)
    assert not output.exists()


@pytest.mark.parametrize("entry", [
    ["胸痛", "症状", "起病", "H-3", "private-marker-must-not-ship"],
    {"name": "胸痛", "category": "症状", "phase": "起病", "time_label": "H-3", "api_key": "secret"},
    ["胸痛", "症状", "起病"],
])
def test_export_rejects_malformed_timeline_entry(tmp_path, entry):
    source, sidecars = _source_fixture(tmp_path)
    path = sidecars / "disease.csv.module_io" / "row_1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = json.loads(payload["模块2.5_输出"])
    model["symptom_timeline"] = [entry]
    payload["模块2.5_输出"] = json.dumps(model, ensure_ascii=False)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output = tmp_path / "m25.jsonl.gz"
    with pytest.raises(ValueError, match="case_b.*M2.5"):
        export_m25_time_models(source, sidecars, output)
    assert not output.exists()
