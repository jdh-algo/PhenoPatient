import csv
import gzip
import json
import shutil
from pathlib import Path

import pytest

from scripts.prepare_m3_runtime import prepare_m3_runtime
from scripts.verify_m3_outputs import verify_m3_outputs


DATA = Path(__file__).resolve().parents[1] / "data"


def _one_patient_bundle(tmp_path, *, stale_time_model=False):
    bundle = tmp_path / "bundle"
    with (DATA / "final_patients/phenopatient_1000.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        patient = next(reader)
    source_group = patient["source_group"]
    atlas = DATA / "m1_atlases" / f"{source_group}.csv"
    target_atlas = bundle / "data/m1_atlases" / f"{source_group}.csv"
    target_atlas.parent.mkdir(parents=True)
    shutil.copy2(atlas, target_atlas)
    target_patient = bundle / "data/final_patients/phenopatient_1000.csv"
    target_patient.parent.mkdir(parents=True)
    with target_patient.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(patient)

    with gzip.open(DATA / "m25_time_models/phenopatient_1000_m25.jsonl.gz", "rt", encoding="utf-8") as handle:
        m25 = next(json.loads(line) for line in handle if json.loads(line)["case_id"] == patient["case_id"])
    if stale_time_model:
        m25["time_model"]["current_episode_duration"] = "不匹配的病程"
    target_m25 = bundle / "data/m25_time_models/phenopatient_1000_m25.jsonl.gz"
    target_m25.parent.mkdir(parents=True)
    with gzip.open(target_m25, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(m25, ensure_ascii=False) + "\n")

    with gzip.open(DATA / "final_ledgers/phenopatient_1000_fact_ledgers.jsonl.gz", "rt", encoding="utf-8") as handle:
        ledger = next(json.loads(line) for line in handle if json.loads(line)["case_id"] == patient["case_id"])
    target_ledger = bundle / "data/final_ledgers/phenopatient_1000_fact_ledgers.jsonl.gz"
    target_ledger.parent.mkdir(parents=True)
    with gzip.open(target_ledger, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
    return bundle, source_group, patient["case_id"]


def test_prepare_restores_m3_compatible_csv_and_minimal_sidecars(tmp_path):
    import patient_fact_ledger as pfl
    import utils

    bundle, group, case_id = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"

    assert prepare_m3_runtime(bundle, output) == {"disease_groups": 1, "patients": 1, "ledgers": 1}
    csv_path = output / f"{group}.csv"
    row_1, patients = utils._load_existing_csv(str(csv_path))
    staging = utils.parse_staging_system(row_1[utils.COL_STAGING_SYSTEM])
    verified = pfl.load_verified_fact_ledger_for_m3(str(csv_path), 1, patients[1], staging)
    assert verified["ledger"]["case_id"] == case_id
    sidecar = json.loads((Path(str(csv_path) + ".module_io") / "row_1.json").read_text(encoding="utf-8"))
    assert set(sidecar) == {utils.COL_M25_OUTPUT, pfl.COL_M281_OUTPUT}
    assert not any(p.name.startswith("row_0") for p in Path(str(csv_path) + ".module_io").iterdir())


def test_prepare_fails_closed_when_time_model_does_not_match_ledger(tmp_path):
    bundle, _, _ = _one_patient_bundle(tmp_path, stale_time_model=True)
    output = tmp_path / "runtime"

    with pytest.raises(ValueError, match="input_hash stale"):
        prepare_m3_runtime(bundle, output)

    assert not output.exists()


def test_prepare_rejects_wrong_expected_count_without_output(tmp_path):
    bundle, _, _ = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"

    with pytest.raises(ValueError, match="patient count mismatch"):
        prepare_m3_runtime(bundle, output, expected_patients=2)

    assert not output.exists()


def test_verify_m3_rejects_prepared_but_unrun_patient(tmp_path):
    bundle, _, case_id = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"
    prepare_m3_runtime(bundle, output)

    with pytest.raises(ValueError, match=f"complete=0/1.*{case_id}"):
        verify_m3_outputs(bundle, output, expected_patients=1)


def test_verify_m3_rejects_missing_csv_despite_runner_success_exit(tmp_path):
    bundle, group, _ = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"
    prepare_m3_runtime(bundle, output)
    (output / f"{group}.csv").unlink()

    with pytest.raises(ValueError, match="runtime CSV set"):
        verify_m3_outputs(bundle, output, expected_patients=1)


def test_verify_m3_rejects_case_id_reused_in_another_group(tmp_path):
    bundle, _, _ = _one_patient_bundle(tmp_path)
    patient_csv = bundle / "data/final_patients/phenopatient_1000.csv"
    with patient_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        original = next(reader)
    duplicate = dict(original, source_group="other/disease")
    with patient_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([original, duplicate])

    with pytest.raises(ValueError, match="duplicate frozen case_id"):
        verify_m3_outputs(bundle, tmp_path / "runtime", expected_patients=2)


def test_verify_m3_does_not_create_missing_sidecar_directory(tmp_path):
    bundle, group, _ = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"
    prepare_m3_runtime(bundle, output)
    sidecars = Path(str(output / f"{group}.csv") + ".module_io")
    shutil.rmtree(sidecars)

    with pytest.raises(ValueError, match="sidecar directory missing"):
        verify_m3_outputs(bundle, output, expected_patients=1)
    assert not sidecars.exists()


def test_verify_m3_checks_interaction_and_current_ledger_metadata(tmp_path):
    import patient_fact_ledger as pfl
    import utils

    bundle, group, _ = _one_patient_bundle(tmp_path)
    output = tmp_path / "runtime"
    prepare_m3_runtime(bundle, output)
    csv_path = output / f"{group}.csv"
    row_1, rows = utils._load_existing_csv(str(csv_path))
    row = rows[1]
    row[utils.COL_INTERACTION] = "[诊断]: 模拟完成"
    utils._save_csv(str(csv_path), [row_1, row])

    with pytest.raises(ValueError, match="complete=0/1"):
        verify_m3_outputs(bundle, output, expected_patients=1)

    context = pfl.load_verified_fact_ledger_for_m3(
        str(csv_path), 1, row, utils.parse_staging_system(row_1[utils.COL_STAGING_SYSTEM])
    )
    pfl.write_m3_ledger_metadata(str(csv_path), 1, context["metadata"])
    assert verify_m3_outputs(bundle, output, expected_patients=1) == {
        "disease_groups": 1, "expected_patients": 1, "complete_patients": 1,
    }

    row[utils.COL_INTERACTION] = "（诊断生成失败）"
    utils._save_csv(str(csv_path), [row_1, row])
    with pytest.raises(ValueError, match="complete=0/1"):
        verify_m3_outputs(bundle, output, expected_patients=1)
