import copy
import multiprocessing
import json
from pathlib import Path

import pytest

import patient_fact_ledger as pfl
import utils
from patient_fact_ledger import (
    COL_M281_OUTPUT,
    build_fact_ledger,
    canonical_ledger_hash,
    converge_fact_ledger,
    ledger_input_hash,
    load_fact_ledger,
    module_2_81_finalize_fact_ledger,
    write_fact_ledger,
)


def _base_row():
    row = utils._empty_row()
    row[utils.COL_CASE_ID] = "case_00001"
    row[utils.COL_AGE] = "68"
    row[utils.COL_GENDER] = "男"
    row[utils.COL_DIAGNOSIS] = "慢性心力衰竭"
    row[utils.COL_STAGE] = "III级"
    row[utils.COL_PATIENT_LATERALITY] = "不适用"
    row[utils.COL_ACUITY] = "慢性急性加重"
    row[utils.COL_DURATION_TOTAL] = "5天"
    row[utils.COL_SYMPTOMS] = repr([
        ("胸闷", "III级", "3小时", "活动后", "持续"),
    ])
    row[utils.COL_SIGNS] = repr([
        ("颈静脉怒张", "III级", "", "", ""),
    ])
    row[utils.COL_LAB_TESTS] = repr([
        ("NT-proBNP升高", "III级", "", "", ""),
        ("血钾降低", "III级", "", "", ""),
    ])
    row[utils.COL_IMAGING] = repr([])
    row[utils.COL_FUNCTIONAL_TESTS] = repr([])
    row[utils.COL_ABSENT_SYMPTOMS] = repr(["发热"])
    row[utils.COL_ABSENT_SIGNS] = repr([])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["肌酐升高"])
    row[utils.COL_ABSENT_IMAGING] = repr([])
    row[utils.COL_ABSENT_FUNCTIONAL] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("胸闷", "症状", "起病", "H-3"),
        ("颈静脉怒张", "体征", "就诊", "D0"),
        ("NT-proBNP升高", "实验室检查", "就诊", "D0"),
        ("血钾降低", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("NT-proBNP升高", "实验室检查", "就诊", "D0", "> 900 pg/mL"),
        ("血钾降低", "实验室检查", "就诊", "D0", "3.0-3.5 mmol/L"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("NT-proBNP升高", "实验室检查", "就诊", "D0", "1850 pg/mL"),
        ("血钾降低", "实验室检查", "就诊", "D0", "3.2 mmol/L"),
    ])
    return row


def _staging_system():
    return {
        "scheme": "NYHA心功能分级",
        "levels": [
            {"level": 1, "name": "I级", "description": "无明显受限"},
            {"level": 3, "name": "III级", "description": "明显受限"},
        ],
    }


def _time_model():
    return {
        "schema_version": 2,
        "underlying_duration": "2年",
        "current_episode_duration": "5天",
        "symptom_timeline": [
            {"name": "胸闷", "category": "症状", "phase": "起病", "time_label": "H-3"},
        ],
    }


def _fact_by_canonical(ledger, canonical):
    matches = [fact for fact in ledger["facts"] if fact["concept"]["canonical"] == canonical]
    assert len(matches) == 1
    return matches[0]


def _concurrent_manifest_worker(csv_path, row_index, case_id, barrier, errors):
    try:
        row = _base_row()
        row[utils.COL_CASE_ID] = case_id
        ledger = build_fact_ledger(row, _staging_system(), _time_model())
        barrier.wait(timeout=10)
        write_fact_ledger(csv_path, row_index, ledger)
    except Exception as exc:
        errors.put(f"{case_id}: {type(exc).__name__}: {exc}")


def _fact_by_raw(ledger, raw):
    matches = [fact for fact in ledger["facts"] if fact["concept"]["raw"] == raw]
    assert len(matches) == 1
    return matches[0]


def _set_objective_entries(row, entries):
    by_category = {
        "体征": [],
        "实验室检查": [],
        "影像检查": [],
        "功能检查": [],
    }
    timeline = []
    quantified = []
    specific = []
    for name, category, range_text, actual_text in entries:
        by_category[category].append((name, "III级", "", "", ""))
        timeline.append((name, category, "就诊", "D0"))
        if range_text is not None:
            quantified.append((f"{name}{range_text}", category, "就诊", "D0"))
        if actual_text is not None:
            specific.append((f"{name}；实际值={actual_text}", category, "就诊", "D0"))
    row[utils.COL_SIGNS] = repr(by_category["体征"])
    row[utils.COL_LAB_TESTS] = repr(by_category["实验室检查"])
    row[utils.COL_IMAGING] = repr(by_category["影像检查"])
    row[utils.COL_FUNCTIONAL_TESTS] = repr(by_category["功能检查"])
    row[utils.COL_ABSENT_SIGNS] = repr([])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_ABSENT_IMAGING] = repr([])
    row[utils.COL_ABSENT_FUNCTIONAL] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("胸闷", "症状", "起病", "H-3"),
        *timeline,
    ])
    row[utils.COL_QUANTIFIED] = repr(quantified)
    row[utils.COL_SPECIFIC] = repr(specific)


def _fact_value(ledger, canonical):
    return _fact_by_canonical(ledger, canonical)["value"]["number"]


def _projection_patches(ledger, kind):
    return [
        patch
        for round_payload in ledger["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
        if patch["kind"] == kind
    ]


def _assert_fact_within_reference_range(ledger, canonical):
    fact = _fact_by_canonical(ledger, canonical)
    value = fact["value"]
    reference = value["reference_range"]
    assert reference["lower"] <= value["number"] <= reference["upper"]


def test_fact_ledger_uses_validator_v9_provenance_version():
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())

    assert ledger["provenance"]["code_version"] == "m2.81.validator.v9"


def test_build_fact_ledger_separates_identity_polarity_time_state_and_values():
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())

    assert ledger["schema_version"] == "fact_ledger.v1"
    assert ledger["module"] == "2.81"
    assert ledger["status"] == "converged"
    assert ledger["patient"] == {
        "age": 68,
        "sex": "男",
        "diagnosis": "慢性心力衰竭",
        "severity_scheme": "NYHA心功能分级",
        "severity_level": "III级",
        "laterality": "not_applicable",
        "acuity": "慢性急性加重",
    }
    assert ledger["course"] == {
        "underlying_duration": "2年",
        "current_episode_duration": "5天",
        "anchor": "current_visit",
    }

    chest_tightness = _fact_by_raw(ledger, "胸闷")
    assert chest_tightness["domain"] == "symptom"
    assert chest_tightness["clinical_state"] == "present"
    assert chest_tightness["interpretation"] is None
    assert chest_tightness["acquisition_state"] == "historical"
    assert chest_tightness["time"] == {
        "clinical_onset": "H-3",
        "observed_at": None,
        "phase": "起病",
    }
    assert "patient" in chest_tightness["visibility"]

    jvp = _fact_by_raw(ledger, "颈静脉怒张")
    assert jvp["domain"] == "sign"
    assert jvp["clinical_state"] == "present"
    assert jvp["acquisition_state"] == "latent"
    assert jvp["time"] == {
        "clinical_onset": None,
        "observed_at": "D0",
        "phase": "就诊",
    }
    assert jvp["visibility"] == ["physical_exam"]

    ntprobnp = _fact_by_raw(ledger, "NT-proBNP升高")
    assert ntprobnp["domain"] == "lab"
    assert ntprobnp["clinical_state"] == "present"
    assert ntprobnp["interpretation"] == "high"
    assert ntprobnp["acquisition_state"] == "latent"
    assert ntprobnp["value"] == {
        "number": 1850,
        "unit": "pg/mL",
        "reference_range": {"raw": "> 900 pg/mL", "lower": 900, "upper": None, "unit": "pg/mL"},
    }
    assert ntprobnp["time"]["observed_at"] == "D0"
    assert ntprobnp["time"]["clinical_onset"] is None
    assert ntprobnp["visibility"] == ["diagnostic_oracle", "doctor_after_order"]

    potassium = _fact_by_raw(ledger, "血钾降低")
    assert potassium["interpretation"] == "low"
    assert potassium["value"] == {
        "number": 3.2,
        "unit": "mmol/L",
        "reference_range": {"raw": "3.0-3.5 mmol/L", "lower": 3.0, "upper": 3.5, "unit": "mmol/L"},
    }

    fever = _fact_by_raw(ledger, "发热")
    assert fever["clinical_state"] == "absent"
    assert fever["interpretation"] == "negative"
    assert fever["acquisition_state"] == "historical"
    assert fever["value"] is None

    creatinine = _fact_by_raw(ledger, "肌酐升高")
    assert creatinine["clinical_state"] == "absent"
    assert creatinine["interpretation"] == "negative"
    assert creatinine["acquisition_state"] == "latent"


def test_build_fact_ledger_rejects_present_absent_conflict_because_identity_excludes_polarity():
    row = _base_row()
    row[utils.COL_ABSENT_SYMPTOMS] = repr(["胸闷"])

    with pytest.raises(ValueError, match="present/absent"):
        build_fact_ledger(row, _staging_system(), _time_model())


def test_build_fact_ledger_allows_present_high_with_absent_low_for_same_measurement():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("血钾升高", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["血钾降低"])
    row[utils.COL_TIME_ORDER] = repr([
        ("血钾升高", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    potassium_facts = [
        fact for fact in ledger["facts"]
        if fact["concept"]["canonical"] == "血钾"
    ]
    assert len(potassium_facts) == 1
    assert potassium_facts[0]["clinical_state"] == "present"
    assert potassium_facts[0]["interpretation"] == "high"
    assert potassium_facts[0]["provenance"]["compatible_absent_assertions"] == [{
        "raw": "血钾降低",
        "qualifier": "low",
        "source": utils.COL_ABSENT_LAB_TESTS,
        "source_index": 0,
    }]


@pytest.mark.parametrize(
    ("present_name", "absent_name"),
    [
        ("血钾升高。", "血钾降低。"),
        ("血钾偏高", "血钾偏低"),
    ],
)
def test_build_fact_ledger_normalizes_directional_suffixes_before_reconciliation(
    present_name, absent_name
):
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        (present_name, "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([absent_name])
    row[utils.COL_TIME_ORDER] = repr([
        (present_name, "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    potassium_facts = [
        fact for fact in ledger["facts"]
        if fact["concept"]["canonical"] == "血钾"
    ]
    assert len(potassium_facts) == 1
    assert potassium_facts[0]["clinical_state"] == "present"


def test_build_fact_ledger_rejects_same_direction_present_absent_conflict():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("血钾升高", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["血钾升高"])
    row[utils.COL_TIME_ORDER] = repr([
        ("血钾升高", "实验室检查", "就诊", "D0"),
    ])

    with pytest.raises(ValueError, match="present/absent"):
        build_fact_ledger(row, _staging_system(), _time_model())


def test_build_fact_ledger_allows_high_with_absent_marked_high():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("白细胞计数升高", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["白细胞计数显著升高"])
    row[utils.COL_TIME_ORDER] = repr([
        ("白细胞计数升高", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    wbc = _fact_by_canonical(ledger, "白细胞计数")
    assert wbc["clinical_state"] == "present"
    assert wbc["interpretation"] == "high"
    assert wbc["provenance"]["compatible_absent_assertions"] == [{
        "raw": "白细胞计数显著升高",
        "qualifier": "marked_high",
        "source": utils.COL_ABSENT_LAB_TESTS,
        "source_index": 0,
    }]


def test_build_fact_ledger_allows_abnormal_with_absent_low():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("平均红细胞体积异常", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["平均红细胞体积降低"])
    row[utils.COL_TIME_ORDER] = repr([
        ("平均红细胞体积异常", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    mcv = _fact_by_canonical(ledger, "平均红细胞体积")
    assert mcv["clinical_state"] == "present"
    assert mcv["interpretation"] == "abnormal"
    assert mcv["provenance"]["compatible_absent_assertions"] == [{
        "raw": "平均红细胞体积降低",
        "qualifier": "low",
        "source": utils.COL_ABSENT_LAB_TESTS,
        "source_index": 0,
    }]


def test_build_fact_ledger_records_absent_absent_compatible_assertions():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["血钾升高", "血钾降低"])
    row[utils.COL_TIME_ORDER] = repr([
        ("胸闷", "症状", "起病", "H-3"),
        ("颈静脉怒张", "体征", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([])
    row[utils.COL_SPECIFIC] = repr([])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    potassium = _fact_by_canonical(ledger, "血钾")
    assert potassium["clinical_state"] == "absent"
    assert potassium["provenance"]["compatible_absent_assertions"] == [{
        "raw": "血钾降低",
        "qualifier": "low",
        "source": utils.COL_ABSENT_LAB_TESTS,
        "source_index": 1,
        "compatibility": "absent_absent",
    }]


def test_build_fact_ledger_merges_urine_protein_positive_and_high():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("尿蛋白阳性", "III级", "", "", ""),
        ("尿蛋白升高", "III级", "", "", ""),
    ])
    row[utils.COL_TIME_ORDER] = repr([
        ("尿蛋白阳性", "实验室检查", "就诊", "D0"),
        ("尿蛋白升高", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    protein = _fact_by_canonical(ledger, "尿蛋白")
    assert protein["clinical_state"] == "present"
    assert protein["interpretation"] == "positive"
    assert protein["provenance"]["sources"] == [
        {"source": utils.COL_LAB_TESTS, "source_index": 0, "source_level": "III级", "raw": "尿蛋白阳性"},
        {"source": utils.COL_LAB_TESTS, "source_index": 1, "source_level": "III级", "raw": "尿蛋白升高"},
    ]


@pytest.mark.parametrize("positive_first", [False, True])
def test_build_fact_ledger_merges_serology_positive_with_quantified_high(positive_first):
    row = _base_row()
    high = ("类风湿因子升高", "III级", "", "", "")
    positive = ("类风湿因子阳性", "III级", "", "", "")
    entries = [positive, high] if positive_first else [high, positive]
    row[utils.COL_LAB_TESTS] = repr(entries)
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("类风湿因子升高", "实验室检查", "就诊", "D0"),
        ("类风湿因子阳性", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("类风湿因子升高{75,10,14,nan}IU/mL", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("类风湿因子升高；实际值=75IU/mL", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())
    rf = _fact_by_canonical(ledger, "类风湿因子")

    assert ledger["status"] == "converged"
    assert rf["concept"]["raw"] == "类风湿因子升高"
    assert rf["interpretation"] == "high"
    assert rf["value"] == {
        "number": 75,
        "unit": "IU/mL",
        "reference_range": {
            "raw": "{75,10,14,nan}IU/mL",
            "lower": 14,
            "upper": None,
            "unit": "IU/mL",
        },
    }
    assert rf["provenance"]["sources"] == [
        {
            "source": utils.COL_LAB_TESTS,
            "source_index": index,
            "source_level": "III级",
            "raw": item[0],
        }
        for index, item in enumerate(entries)
    ]


def test_build_fact_ledger_keeps_serology_high_low_conflict():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("类风湿因子升高", "III级", "", "", ""),
        ("类风湿因子降低", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("类风湿因子升高", "实验室检查", "就诊", "D0"),
        ("类风湿因子降低", "实验室检查", "就诊", "D0"),
    ])

    with pytest.raises(ValueError, match="interpretation conflict"):
        build_fact_ledger(row, _staging_system(), _time_model())


def test_build_fact_ledger_merges_absent_duplicate_sources():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([
        "白细胞计数升高",
        "白细胞计数显著升高",
    ])
    row[utils.COL_TIME_ORDER] = repr([])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    wbc = _fact_by_canonical(ledger, "白细胞计数")
    assert wbc["clinical_state"] == "absent"
    assert wbc["provenance"]["source"] == utils.COL_ABSENT_LAB_TESTS
    assert wbc["provenance"]["source_index"] == 0
    assert wbc["provenance"]["sources"] == [
        {"source": utils.COL_ABSENT_LAB_TESTS, "source_index": 0, "raw": "白细胞计数升高"},
        {"source": utils.COL_ABSENT_LAB_TESTS, "source_index": 1, "raw": "白细胞计数显著升高"},
    ]


def test_build_fact_ledger_rejects_legacy_time_model_shape():
    with pytest.raises(ValueError, match="schema_version=2"):
        build_fact_ledger(_base_row(), _staging_system(), [("胸闷", "症状", "H-3")])


def test_canonical_hash_is_stable_and_excludes_audit_final_hash():
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())
    reordered = {
        "provenance": copy.deepcopy(ledger["provenance"]),
        "audit": {"final_hash": "stale-placeholder", "rounds": []},
        "relations": copy.deepcopy(ledger["relations"]),
        "facts": copy.deepcopy(ledger["facts"]),
        "course": copy.deepcopy(ledger["course"]),
        "patient_context": copy.deepcopy(ledger["patient_context"]),
        "patient": copy.deepcopy(ledger["patient"]),
        "case_id": ledger["case_id"],
        "status": ledger["status"],
        "module": ledger["module"],
        "schema_version": ledger["schema_version"],
    }

    assert canonical_ledger_hash(ledger) == canonical_ledger_hash(reordered)

    changed = copy.deepcopy(ledger)
    changed["facts"][0]["clinical_state"] = "unknown"
    assert canonical_ledger_hash(changed) != canonical_ledger_hash(ledger)


def test_ledger_input_hash_changes_when_validator_version_changes(monkeypatch):
    original_hash = ledger_input_hash(_base_row(), _staging_system(), _time_model())

    monkeypatch.setattr(pfl, "LEDGER_CODE_VERSION", "future-validator", raising=False)

    assert ledger_input_hash(_base_row(), _staging_system(), _time_model()) != original_hash


def test_build_error_quarantine_uses_versioned_ledger_input_hash():
    row = _base_row()
    row[utils.COL_M25_OUTPUT] = json.dumps(_time_model(), ensure_ascii=False)
    row[utils.COL_ABSENT_SYMPTOMS] = repr(["胸闷"])

    _, ledger = module_2_81_finalize_fact_ledger(row, _staging_system())

    assert ledger["status"] == "quarantined"
    assert ledger["audit"]["stop_reason"] == "build_error"
    assert ledger["provenance"]["source_row_hash"] == ledger_input_hash(
        row, _staging_system(), _time_model()
    )


def test_sidecar_roundtrip_fails_closed_for_case_id_hash_and_resume_hash(tmp_path):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    row_1 = utils._empty_row()
    patient = _base_row()
    utils._save_csv(str(csv_path), [row_1, patient])

    ledger = build_fact_ledger(patient, _staging_system(), _time_model())
    summary = write_fact_ledger(str(csv_path), 1, ledger)

    assert summary == {
        "status": "converged",
        "path": "row_1.fact_ledger.json",
        "input_hash": ledger_input_hash(patient, _staging_system(), _time_model()),
        "ledger_hash": canonical_ledger_hash(ledger),
        "round_count": 0,
    }
    loaded = load_fact_ledger(str(csv_path), 1, expected_case_id="case_00001")
    assert loaded == ledger
    assert loaded["provenance"]["source_row_hash"] == summary["input_hash"]
    assert loaded["audit"]["final_hash"] == summary["ledger_hash"]

    _, existing_patients = utils._load_existing_csv(str(csv_path))
    assert COL_M281_OUTPUT not in existing_patients[1]

    with pytest.raises(ValueError, match="case_id"):
        load_fact_ledger(str(csv_path), 1, expected_case_id="case_99999")

    module_io_dir = Path(str(csv_path) + ".module_io")
    row_sidecar = module_io_dir / "row_1.json"
    row_payload = json.loads(row_sidecar.read_text(encoding="utf-8"))
    row_payload[COL_M281_OUTPUT]["ledger_hash"] = "bad-hash"
    row_sidecar.write_text(json.dumps(row_payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="ledger_hash"):
        load_fact_ledger(str(csv_path), 1, expected_case_id="case_00001")


def test_manifest_preserves_two_row_records(tmp_path):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    first = _base_row()
    second = _base_row()
    second[utils.COL_CASE_ID] = "case_00002"
    utils._save_csv(str(csv_path), [utils._empty_row(), first, second])

    first_ledger = build_fact_ledger(first, _staging_system(), _time_model())
    second_ledger = build_fact_ledger(second, _staging_system(), _time_model())
    write_fact_ledger(str(csv_path), 1, first_ledger)
    write_fact_ledger(str(csv_path), 2, second_ledger)

    manifest_path = Path(str(csv_path) + ".module_io") / "fact_ledger_manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert [record["row_index"] for record in records] == [1, 2]
    assert [record["case_id"] for record in records] == ["case_00001", "case_00002"]
    assert records[0]["ledger_hash"] == canonical_ledger_hash(first_ledger)
    assert records[1]["ledger_hash"] == canonical_ledger_hash(second_ledger)


def test_build_fact_ledger_reads_production_four_tuple_values_and_ranges_from_item_names():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("CA19-9升高", "III级", "", "", ""),
    ])
    row[utils.COL_FUNCTIONAL_TESTS] = repr([
        ("FEV1/FVC降低", "III级", "", "", ""),
    ])
    row[utils.COL_TIME_ORDER] = repr([
        ("CA19-9升高", "实验室检查", "就诊", "D0"),
        ("FEV1/FVC降低", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("CA19-9升高{100,20,37,nan}U/mL", "实验室检查", "就诊", "D0"),
        ("FEV1/FVC降低{0.60,0.05,nan,0.70}", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("CA19-9升高；实际值=125U/mL", "实验室检查", "就诊", "D0"),
        ("FEV1/FVC降低；实际值=0.62", "功能检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    ca199 = _fact_by_canonical(ledger, "CA19-9")
    assert ca199["value"] == {
        "number": 125,
        "unit": "U/mL",
        "reference_range": {"raw": "{100,20,37,nan}U/mL", "lower": 37, "upper": None, "unit": "U/mL"},
    }
    fev_ratio = _fact_by_canonical(ledger, "FEV1/FVC")
    assert fev_ratio["value"] == {
        "number": 0.62,
        "unit": None,
        "reference_range": {"raw": "{0.60,0.05,nan,0.70}", "lower": None, "upper": 0.70, "unit": None},
    }

    changed_specific = copy.deepcopy(row)
    changed_specific[utils.COL_SPECIFIC] = repr([
        ("CA19-9升高；实际值=126U/mL", "实验室检查", "就诊", "D0"),
        ("FEV1/FVC降低；实际值=0.62", "功能检查", "就诊", "D0"),
    ])
    changed_range = copy.deepcopy(row)
    changed_range[utils.COL_QUANTIFIED] = repr([
        ("CA19-9升高{100,20,38,nan}U/mL", "实验室检查", "就诊", "D0"),
        ("FEV1/FVC降低{0.60,0.05,nan,0.70}", "功能检查", "就诊", "D0"),
    ])
    original_hash = ledger_input_hash(row, _staging_system(), _time_model())
    assert ledger_input_hash(changed_specific, _staging_system(), _time_model()) != original_hash
    assert ledger_input_hash(changed_range, _staging_system(), _time_model()) != original_hash


def test_build_fact_ledger_rejects_same_present_identity_with_opposite_interpretation():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("肌酐升高", "III级", "", "", ""),
        ("肌酐降低", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("肌酐升高", "实验室检查", "就诊", "D0"),
        ("肌酐降低", "实验室检查", "就诊", "D0"),
    ])

    with pytest.raises(ValueError, match="interpretation conflict"):
        build_fact_ledger(row, _staging_system(), _time_model())


def test_build_fact_ledger_merges_null_and_explicit_present_interpretation_safely():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("肌酐", "III级", "", "", ""),
        ("肌酐升高", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("肌酐", "实验室检查", "就诊", "D0"),
        ("肌酐升高", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())

    creatinine = _fact_by_canonical(ledger, "肌酐")
    assert creatinine["concept"]["raw"] == "肌酐升高"
    assert creatinine["interpretation"] == "high"


def test_manifest_keeps_all_records_under_multiprocessing_writes(tmp_path):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    process_count = 16
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(process_count)
    errors = ctx.Queue()
    processes = []
    for offset in range(process_count):
        row_index = offset + 1
        case_id = f"case_{row_index:05d}"
        process = ctx.Process(
            target=_concurrent_manifest_worker,
            args=(str(csv_path), row_index, case_id, barrier, errors),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join(20)
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    queued_errors = []
    while not errors.empty():
        queued_errors.append(errors.get())

    assert failures == []
    assert queued_errors == []
    manifest_path = Path(str(csv_path) + ".module_io") / "fact_ledger_manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert sorted(record["row_index"] for record in records) == list(range(1, process_count + 1))
    assert sorted(record["case_id"] for record in records) == [f"case_{idx:05d}" for idx in range(1, process_count + 1)]


def test_write_and_load_reject_malformed_ledger_with_clear_value_error(tmp_path):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())
    malformed = copy.deepcopy(ledger)
    malformed.pop("audit")

    with pytest.raises(ValueError, match="audit"):
        write_fact_ledger(str(csv_path), 1, malformed)

    write_fact_ledger(str(csv_path), 1, ledger)
    ledger_path = Path(str(csv_path) + ".module_io") / "row_1.fact_ledger.json"
    on_disk = json.loads(ledger_path.read_text(encoding="utf-8"))
    on_disk.pop("audit")
    ledger_path.write_text(json.dumps(on_disk, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="audit"):
        load_fact_ledger(str(csv_path), 1, expected_case_id="case_00001")


def test_converge_fact_ledger_repairs_ag_and_bilirubin_derivations_after_two_rounds():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,0,140,140}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{30,5,0,40}mmol/L", "35mmol/L"),
        ("总胆红素升高", "实验室检查", "{60,0,60,60}umol/L", "60umol/L"),
        ("直接胆红素升高", "实验室检查", "{20,0,20,20}umol/L", "20umol/L"),
        ("间接胆红素升高", "实验室检查", "{8,2,0,80}umol/L", "8umol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert len(result["audit"]["rounds"]) == 2
    assert _fact_value(result, "阴离子间隙") == pytest.approx(20)
    assert _fact_value(result, "间接胆红素") == pytest.approx(40)
    assert result["audit"]["rounds"][0]["patches"]
    assert result["audit"]["rounds"][1]["blockers"] == []


def test_converge_fact_ledger_joint_projects_anion_gap_within_all_ranges():
    row = _base_row()
    _set_objective_entries(row, [
        ("血钠降低", "实验室检查", "{130,4,115,135}mmol/L", "124.91mmol/L"),
        ("血氯降低", "实验室检查", "{92,4,75,97}mmol/L", "94.19mmol/L"),
        ("碳酸氢根降低", "实验室检查", "{15,4,5,21.9}mmol/L", "16.23mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{22,6,16.1,40}mmol/L", "27.96mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    values = {
        name: _fact_by_canonical(result, name)["value"]["number"]
        for name in ("血钠", "血氯", "碳酸氢根", "阴离子间隙")
    }
    assert 115 <= values["血钠"] <= 135
    assert 75 <= values["血氯"] <= 97
    assert 5 <= values["碳酸氢根"] <= 21.9
    assert 16.1 <= values["阴离子间隙"] <= 40
    assert values["阴离子间隙"] == pytest.approx(
        values["血钠"] - values["血氯"] - values["碳酸氢根"], abs=0.01
    )
    assert {
        _fact_by_canonical(result, name)["interpretation"]
        for name in ("血钠", "血氯", "碳酸氢根")
    } == {"low"}
    assert _fact_by_canonical(result, "阴离子间隙")["interpretation"] == "high"
    patch_kinds = {
        patch["kind"]
        for round_payload in result["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
    }
    assert "anion_gap_joint_projection" in patch_kinds
    projection_patches = [
        patch
        for round_payload in result["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
        if patch["kind"] == "anion_gap_joint_projection"
    ]
    assert all(patch["standardized_shift"] <= 3 for patch in projection_patches)
    assert all(patch["projection_distance"] <= 4 for patch in projection_patches)


def test_anion_gap_joint_projection_rejects_excessive_standardized_shift():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,0.1,100,200}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{35,1,35,40}mmol/L", "35mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "unrepairable"


def test_anion_gap_joint_projection_rejects_direction_reversal():
    row = _base_row()
    _set_objective_entries(row, [
        ("血钠降低", "实验室检查", "{134,100,100,200}mmol/L", "134mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{35,0,35,35}mmol/L", "35mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "unrepairable"


def test_anion_gap_joint_projection_requires_finite_m27_bounds():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,100,nan,nan}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{35,0,35,35}mmol/L", "35mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "unrepairable"


def test_bnp_aliases_are_not_sodium_for_anion_gap_grouping():
    row = _base_row()
    _set_objective_entries(row, [
        ("N末端B型利钠肽原升高", "实验室检查", "{1800,0,1800,1800}pg/mL", "1800pg/mL"),
        ("N端脑钠肽前体升高", "实验室检查", "{900,0,900,900}pg/mL", "900pg/mL"),
        ("NT-proBNP升高", "实验室检查", "{2000,0,2000,2000}pg/mL", "2000pg/mL"),
        ("BNP升高", "实验室检查", "{600,0,600,600}pg/mL", "600pg/mL"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{30,0,30,30}mmol/L", "30mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "N末端B型利钠肽原") == 1800
    assert _fact_value(result, "N端脑钠肽前体") == 900
    assert _fact_value(result, "NT-proBNP") == 2000
    assert _fact_value(result, "BNP") == 600
    assert _projection_patches(result, "anion_gap_joint_projection") == []


def test_nonblood_puncture_fluid_wbc_differential_is_not_blood_constraint():
    row = _base_row()
    _set_objective_entries(row, [
        ("腹腔穿刺液白细胞计数升高", "实验室检查", "{2000,500,0,5000}10^9/L", "2000 10^9/L"),
        ("穿刺液中性粒细胞比例升高", "实验室检查", "{80,10,50,95}%", "80%"),
        ("腹腔液中性粒细胞绝对值升高", "实验室检查", "{200,500,0,3000}10^9/L", "200 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "腹腔液中性粒细胞绝对值") == 200
    assert _projection_patches(result, "wbc_differential_joint_projection") == []


@pytest.mark.parametrize(("total_name", "pct_name", "abs_name", "abs_canonical"), [
    (
        "胸腔液白细胞计数升高",
        "胸腔液中性粒细胞比例升高",
        "胸腔液中性粒细胞绝对值升高",
        "胸腔液中性粒细胞绝对值",
    ),
    (
        "胸腔积液白细胞计数升高",
        "胸腔积液中性粒细胞比例升高",
        "胸腔积液中性粒细胞绝对值升高",
        "胸腔积液中性粒细胞绝对值",
    ),
    (
        "关节液白细胞计数升高",
        "关节液中性粒细胞比例升高",
        "关节液中性粒细胞绝对值升高",
        "关节液中性粒细胞绝对值",
    ),
])
def test_common_body_fluid_wbc_differential_is_not_peripheral_blood_constraint(
        total_name, pct_name, abs_name, abs_canonical):
    row = _base_row()
    _set_objective_entries(row, [
        (total_name, "实验室检查", "{2000,500,0,5000}10^9/L", "2000 10^9/L"),
        (pct_name, "实验室检查", "{80,10,50,95}%", "80%"),
        (abs_name, "实验室检查", "{200,500,0,3000}10^9/L", "200 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, abs_canonical) == 200
    assert _projection_patches(result, "wbc_differential_joint_projection") == []


def test_unitless_fraction_wbc_percent_uses_percent_canonical_scale():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数", "实验室检查", "{10,0,10,10}10^9/L", "10 10^9/L"),
        ("中性粒细胞比例", "实验室检查", "{0.8,0.1,0.5,0.95}", "0.8"),
        ("中性粒细胞绝对值", "实验室检查", "{8,0,8,8}10^9/L", "8 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "中性粒细胞比例") == pytest.approx(0.8)
    assert _fact_value(result, "中性粒细胞绝对值") == pytest.approx(8)


def test_wbc_differential_quarantines_unsupported_total_unit():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数升高", "实验室检查", "{12000,1000,10000,20000}/uL", "12000/uL"),
        ("中性粒细胞比例升高", "实验室检查", "{80,10,60,90}%", "80%"),
        ("中性粒细胞绝对值升高", "实验室检查", "{8,1,7,12}10^9/L", "8 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"unsupported_unit"}


def test_bilirubin_direct_repair_requires_finite_m27_bounds():
    row = _base_row()
    _set_objective_entries(row, [
        ("总胆红素升高", "实验室检查", "{60,5,40,nan}umol/L", "60umol/L"),
        ("直接胆红素升高", "实验室检查", "{20,5,10,30}umol/L", "20umol/L"),
        ("间接胆红素升高", "实验室检查", "{8,5,0,nan}umol/L", "8umol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"bilirubin_triad"}


def test_wbc_direct_repair_requires_finite_m27_bounds():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数", "实验室检查", "{10,2,5,nan}10^9/L", "10 10^9/L"),
        ("中性粒细胞比例", "实验室检查", "{80,10,50,95}%", "80%"),
        ("中性粒细胞绝对值", "实验室检查", "{2,2,0,nan}10^9/L", "2 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"wbc_differential"}


def test_red_cell_direct_repair_requires_finite_m27_bounds():
    row = _base_row()
    _set_objective_entries(row, [
        ("红细胞计数", "实验室检查", "{5,0.5,4,nan}10^12/L", "5 10^12/L"),
        ("血红蛋白", "实验室检查", "{150,10,120,160}g/L", "150g/L"),
        ("红细胞压积", "实验室检查", "{45,2.5,40,50}%", "45%"),
        ("平均红细胞体积", "实验室检查", "{20,10,0,nan}fL", "20fL"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"red_cell_indices"}


def test_bilirubin_triad_joint_projects_when_single_indirect_patch_exits_range():
    row = _base_row()
    _set_objective_entries(row, [
        ("总胆红素升高", "实验室检查", "{80,5,70,90}umol/L", "80umol/L"),
        ("直接胆红素升高", "实验室检查", "{30,10,20,60}umol/L", "30umol/L"),
        ("间接胆红素升高", "实验室检查", "{10,1,5,15}umol/L", "10umol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    total = _fact_value(result, "总胆红素")
    direct = _fact_value(result, "直接胆红素")
    indirect = _fact_value(result, "间接胆红素")
    for canonical in ("总胆红素", "直接胆红素", "间接胆红素"):
        _assert_fact_within_reference_range(result, canonical)
    assert indirect == pytest.approx(total - direct, abs=0.01)
    patches = _projection_patches(result, "bilirubin_triad_joint_projection")
    assert patches
    assert {patch["basis"] for patch in patches} == {"bounded_joint_projection"}
    assert all(patch["standardized_shift"] <= 3 for patch in patches)
    assert all(patch["projection_distance"] <= 4 for patch in patches)


def test_wbc_differential_joint_projects_when_single_abs_patch_exits_range():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数升高", "实验室检查", "{20,2,18,24}10^9/L", "20 10^9/L"),
        ("中性粒细胞比例升高", "实验室检查", "{80,10,60,90}%", "80%"),
        ("中性粒细胞绝对值升高", "实验室检查", "{10,1,10,12}10^9/L", "10 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    total = _fact_value(result, "白细胞计数")
    pct = _fact_value(result, "中性粒细胞比例")
    absolute = _fact_value(result, "中性粒细胞绝对值")
    for canonical in ("白细胞计数", "中性粒细胞比例", "中性粒细胞绝对值"):
        _assert_fact_within_reference_range(result, canonical)
    assert absolute == pytest.approx(total * pct / 100, abs=0.01)
    patches = _projection_patches(result, "wbc_differential_joint_projection")
    assert patches
    assert {patch["basis"] for patch in patches} == {"bounded_joint_projection"}
    assert all(patch["standardized_shift"] <= 3 for patch in patches)
    assert all(patch["projection_distance"] <= 4 for patch in patches)


def test_red_cell_indices_joint_project_core_and_observed_indices():
    row = _base_row()
    _set_objective_entries(row, [
        ("红细胞计数", "实验室检查", "{5,0.5,4,6}10^12/L", "5 10^12/L"),
        ("血红蛋白", "实验室检查", "{150,10,120,160}g/L", "150g/L"),
        ("红细胞压积", "实验室检查", "{45,2.5,40,50}%", "45%"),
        ("平均红细胞体积", "实验室检查", "{70,10,70,80}fL", "70fL"),
        ("平均红细胞血红蛋白量", "实验室检查", "{30,2,28,32}pg", "30pg"),
        ("平均红细胞血红蛋白浓度", "实验室检查", "{370,10,350,390}g/L", "370g/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    rbc = _fact_value(result, "红细胞计数")
    hb = _fact_value(result, "血红蛋白")
    hct = _fact_value(result, "红细胞压积")
    mcv = _fact_value(result, "平均红细胞体积")
    mch = _fact_value(result, "平均红细胞血红蛋白量")
    mchc = _fact_value(result, "平均红细胞血红蛋白浓度")
    for canonical in (
            "红细胞计数", "血红蛋白", "红细胞压积",
            "平均红细胞体积", "平均红细胞血红蛋白量", "平均红细胞血红蛋白浓度"):
        _assert_fact_within_reference_range(result, canonical)
    assert mcv == pytest.approx(hct * 10 / rbc, abs=0.05)
    assert mch == pytest.approx(hb / rbc, abs=0.05)
    assert mchc == pytest.approx(hb * 100 / hct, abs=0.1)
    patches = _projection_patches(result, "red_cell_indices_joint_projection")
    assert patches
    assert {patch["basis"] for patch in patches} == {"bounded_joint_projection"}
    assert all(patch["standardized_shift"] <= 3 for patch in patches)
    assert all(patch["projection_distance"] <= 4 for patch in patches)


def test_converge_fact_ledger_repairs_wbc_differential_and_red_cell_indices():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数", "实验室检查", "{10,0,10,10}10^9/L", "10 10^9/L"),
        ("中性粒细胞比例", "实验室检查", "{80,0,80,80}%", "80%"),
        ("中性粒细胞绝对值", "实验室检查", "{2,1,0,12}10^9/L", "2 10^9/L"),
        ("红细胞计数", "实验室检查", "{5,0,5,5}10^12/L", "5 10^12/L"),
        ("血红蛋白", "实验室检查", "{150,0,150,150}g/L", "150g/L"),
        ("红细胞压积", "实验室检查", "{45,0,45,45}%", "45%"),
        ("平均红细胞体积", "实验室检查", "{20,5,10,120}fL", "20fL"),
        ("平均红细胞血红蛋白量", "实验室检查", "{1,1,0,50}pg", "1pg"),
        ("平均红细胞血红蛋白浓度", "实验室检查", "{1,1,0,500}g/L", "1g/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "中性粒细胞绝对值") == pytest.approx(8)
    assert _fact_value(result, "平均红细胞体积") == pytest.approx(90)
    assert _fact_value(result, "平均红细胞血红蛋白量") == pytest.approx(30)
    assert _fact_value(result, "平均红细胞血红蛋白浓度") == pytest.approx(333.333, abs=0.01)


def test_reticulocyte_count_is_not_rbc_and_wbc_differential_still_converges():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数", "实验室检查", "{13.61,0,13.61,13.61}10^9/L", "13.61 10^9/L"),
        ("中性粒细胞比例", "实验室检查", "{80,0,80,80}%", "80%"),
        ("中性粒细胞绝对值", "实验室检查", "{12.73,2,0,20}10^9/L", "12.73 10^9/L"),
        ("网织红细胞计数升高", "实验室检查", "{120,40,100,300}10^9/L", "145.12 10^9/L"),
        ("血红蛋白", "实验室检查", "{150,0,150,150}g/L", "150g/L"),
        ("红细胞压积", "实验室检查", "{45,0,45,45}%", "45%"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert len(result["audit"]["rounds"]) == 2
    assert _fact_value(result, "中性粒细胞绝对值") == pytest.approx(10.888)
    reticulocyte = _fact_by_canonical(result, "网织红细胞计数")
    assert reticulocyte["value"]["number"] == pytest.approx(145.12)


def test_wbc_differential_does_not_treat_band_neutrophil_percentage_as_total():
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数升高", "实验室检查", "{12.5,2,10.1,18}10^9/L", "12.5 10^9/L"),
        ("杆状核中性粒细胞比例升高", "实验室检查", "{0.09,0.03,0.01,0.2}%", "0.09%"),
        ("中性粒细胞计数升高", "实验室检查", "{8.85,2,7.1,28}10^9/L", "8.85 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "中性粒细胞计数") == pytest.approx(8.85)
    assert _fact_value(result, "杆状核中性粒细胞比例") == pytest.approx(0.09)


@pytest.mark.parametrize("subtype_name", [
    "杆状中性粒细胞比例升高",
    "带状中性粒细胞比例升高",
    "分叶中性粒细胞百分比升高",
    "BAND-NEUT%升高",
    "BAND_NEUT%升高",
    "STAB NEUT%升高",
    "SEG NEUT%升高",
    "SEGMENTED NEUT%升高",
])
def test_wbc_differential_ignores_neutrophil_subtype_aliases(subtype_name):
    row = _base_row()
    _set_objective_entries(row, [
        ("白细胞计数升高", "实验室检查", "{12.5,2,10.1,18}10^9/L", "12.5 10^9/L"),
        (subtype_name, "实验室检查", "{0.09,0.03,0.01,0.2}%", "0.09%"),
        ("中性粒细胞计数升高", "实验室检查", "{8.85,2,7.1,28}10^9/L", "8.85 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "中性粒细胞计数") == pytest.approx(8.85)


def test_wbc_differential_keeps_total_neut_percent_and_hash_aliases():
    row = _base_row()
    _set_objective_entries(row, [
        ("WBC", "实验室检查", "{10,0,10,10}10^9/L", "10 10^9/L"),
        ("NEUT%", "实验室检查", "{80,0,80,80}%", "80%"),
        ("NEUT#", "实验室检查", "{2,2,0,12}10^9/L", "2 10^9/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "NEUT#") == pytest.approx(8)


def test_converge_fact_ledger_repairs_blood_gas_and_respects_cardio_oxygen_scope():
    row = _base_row()
    _set_objective_entries(row, [
        ("动脉血pH降低", "实验室检查", "{7.40,0,6.80,7.80}", "7.40"),
        ("动脉二氧化碳分压升高", "实验室检查", "{40,0,10,120}mmHg", "40mmHg"),
        ("碳酸氢根降低", "实验室检查", "{10,5,5,40}mmol/L", "10mmol/L"),
        ("血氧饱和度SpO2降低", "体征", "{88,0,70,100}%", "88%"),
        ("动脉血氧饱和度SaO2降低", "实验室检查", "{89,0,70,100}%", "89%"),
        ("混合静脉血氧饱和度SvO2降低", "实验室检查", "{55,0,30,80}%", "55%"),
        ("动脉氧分压PaO2降低", "实验室检查", "{55,0,30,80}mmHg", "55mmHg"),
        ("心率增快", "体征", "{130,0,40,180}次/分", "130次/分"),
        ("脉率降低", "体征", "{60,0,40,180}次/分", "60次/分"),
        ("心电图心室率增快", "功能检查", "{128,0,40,180}次/分", "128次/分"),
        ("心电图提示心房颤动", "功能检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "碳酸氢根") == pytest.approx(24, abs=0.2)
    assert _fact_value(result, "血氧饱和度SpO2") == 88
    assert _fact_value(result, "动脉血氧饱和度SaO2") == 89
    assert _fact_value(result, "混合静脉血氧饱和度SvO2") == 55
    assert _fact_value(result, "动脉氧分压PaO2") == 55
    assert _fact_value(result, "脉率") == 60


def test_converge_fact_ledger_projects_spo2_and_sao2_into_overlapping_ranges():
    row = _base_row()
    _set_objective_entries(row, [
        ("血氧饱和度下降（SpO2<95%）", "体征", "{92,1,90,94}%", "93%"),
        ("动脉血氧饱和度降低", "实验室检查", "{80,5,65,92}%", "86%"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    spo2 = _fact_value(result, "血氧饱和度下降（SpO2<95%）")
    sao2 = _fact_value(result, "动脉血氧饱和度")

    assert result["status"] == "converged"
    assert 90 <= spo2 <= 94
    assert 65 <= sao2 <= 92
    assert abs(spo2 - sao2) <= 3
    assert "spo2_sao2" in {
        patch["kind"]
        for round_payload in result["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
    }


def test_converge_fact_ledger_aligns_heart_rate_pulse_and_ecg_when_no_pulse_deficit():
    row = _base_row()
    _set_objective_entries(row, [
        ("心率增快", "体征", "{120,0,40,180}次/分", "120次/分"),
        ("脉率", "体征", "{70,0,40,180}次/分", "70次/分"),
        ("心电图心室率增快", "功能检查", "{118,0,40,180}次/分", "118次/分"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "脉率") == pytest.approx(119)


def test_converge_fact_ledger_aligns_duplicate_systolic_bp_at_same_time():
    row = _base_row()
    _set_objective_entries(row, [
        ("重度收缩压升高（SBP>180mmHg）", "体征", "{195,10,181,230}mmHg", "182mmHg"),
        ("收缩压升高（SBP>140mmHg）", "体征", "{172,18,141,230}mmHg", "175mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    values = {
        fact["value"]["number"]
        for fact in result["facts"]
        if "收缩压" in fact["concept"]["canonical"]
    }

    assert result["status"] == "converged"
    assert len(values) == 1
    assert next(iter(values)) > 180
    assert "same_metric_value" in {
        patch["kind"]
        for round_payload in result["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
    }


def test_converge_fact_ledger_quarantines_duplicate_systolic_bp_without_shared_range():
    row = _base_row()
    _set_objective_entries(row, [
        ("重度收缩压升高（SBP>180mmHg）", "体征", "{185,2,181,190}mmHg", "185mmHg"),
        ("收缩压升高（SBP>140mmHg）", "体征", "{170,2,141,175}mmHg", "170mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "unrepairable"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"same_metric_value"}


def test_converge_fact_ledger_keeps_systolic_bp_at_distinct_times_separate():
    row = _base_row()
    _set_objective_entries(row, [
        ("收缩压升高（SBP>140mmHg）", "体征", "{182,0,182,182}mmHg", "182mmHg"),
        ("收缩压升高（SBP>140mmHg）复测", "体征", "{150,0,150,150}mmHg", "150mmHg"),
    ])
    ledger = build_fact_ledger(row, _staging_system(), _time_model())
    bp_facts = [fact for fact in ledger["facts"] if "收缩压" in fact["concept"]["canonical"]]
    bp_facts[1]["time"]["observed_at"] = "D-1"

    result = converge_fact_ledger(ledger)
    result_bp_facts = [
        fact for fact in result["facts"]
        if "收缩压" in fact["concept"]["canonical"]
    ]

    assert result["status"] == "converged"
    assert {fact["value"]["number"] for fact in result_bp_facts} == {150, 182}


def test_converge_fact_ledger_keeps_systolic_bp_in_distinct_positions_separate():
    row = _base_row()
    _set_objective_entries(row, [
        ("卧位收缩压升高（SBP>140mmHg）", "体征", "{182,0,182,182}mmHg", "182mmHg"),
        ("站立位收缩压升高（SBP>140mmHg）", "体征", "{150,0,150,150}mmHg", "150mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    values = {
        fact["value"]["number"]
        for fact in result["facts"]
        if "收缩压" in fact["concept"]["canonical"]
    }

    assert result["status"] == "converged"
    assert values == {150, 182}


def test_patient_context_ignores_absent_sex_and_laterality_findings():
    row = _base_row()
    row[utils.COL_GENDER] = "男"
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [])
    row[utils.COL_ABSENT_LAB_TESTS] = repr(["妊娠试验阳性"])
    row[utils.COL_ABSENT_IMAGING] = repr(["右肾积水"])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"


def test_patient_context_ignores_contralateral_comorbidity_finding():
    row = _base_row()
    row[utils.COL_GENDER] = "女"
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [])
    finding = "超声示妊娠相关输尿管扩张或右侧肾盂轻度扩张"
    row[utils.COL_IMAGING] = repr([(finding, "伴随疾病")])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_by_raw(result, finding)["provenance"]["source_level"] == "伴随疾病"


def test_patient_context_ignores_fixed_left_traube_landmark():
    row = _base_row()
    row[utils.COL_DIAGNOSIS] = "胸腔积液/脓胸"
    row[utils.COL_PATIENT_LATERALITY] = "右侧"
    _set_objective_entries(row, [
        ("左侧Traube区叩诊浊音（左下胸Traube鼓音区消失）", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"


def test_patient_context_keeps_present_paired_organ_laterality_conflict():
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("右肾积水", "影像检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


def test_patient_context_keeps_present_limb_laterality_conflict():
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("右膝关节肿胀", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


@pytest.mark.parametrize("finding", ["右下叶肺炎", "右中叶实变", "右肺下叶实变"])
def test_patient_context_keeps_present_lung_lobe_laterality_conflict(finding):
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        (finding, "影像检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


def test_patient_context_ignores_regional_and_referred_pain_side_words():
    row = _base_row()
    row[utils.COL_DIAGNOSIS] = "急性冠脉综合征/心肌梗死"
    row[utils.COL_PATIENT_LATERALITY] = "右侧"
    _set_objective_entries(row, [
        ("左肩放射痛", "体征", None, None),
        ("左上腹压痛", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"


@pytest.mark.parametrize("finding", [
    "左手无名指牵涉痛",
    "左上肢放射痛",
    "胸痛放射至左上肢",
    "胸痛向左上肢放射",
    "胸痛放散至左上肢",
    "左上肢放散痛",
])
def test_patient_context_ignores_referred_pain_in_paired_limb_locations(finding):
    row = _base_row()
    row[utils.COL_DIAGNOSIS] = "冠状动脉粥样硬化性心脏病/不稳定型心绞痛"
    row[utils.COL_PATIENT_LATERALITY] = "右侧"
    _set_objective_entries(row, [
        (finding, "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"


@pytest.mark.parametrize("finding, category", [
    ("右肾积水伴左肩放射痛", "影像检查"),
    ("右膝关节肿胀伴左上肢放射痛", "体征"),
])
def test_patient_context_keeps_objective_laterality_conflict_in_mixed_referred_pain_fact(
        finding, category):
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        (finding, category, None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


def test_patient_context_does_not_apply_bilateral_exemption_to_another_body_site():
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("双侧上肢麻木伴右肾积水", "影像检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


@pytest.mark.parametrize("finding", [
    "双侧肾积水，右肾较重",
    "双侧肾积水，右侧较重",
    "双肾积水，右侧较重",
    "双侧上肢麻木，左侧为主",
    "双侧肾积水，右侧为重",
    "双侧肾积水，右侧为著",
    "双肺炎，右下叶较重",
    "双侧肺炎，右下叶较重",
    "双肺炎，右肺下叶较重",
    "双侧肺炎，右肺下叶较重",
    "双侧肾积水，右肾盂较重",
    "双侧下肢水肿，右踝较重",
])
def test_patient_context_keeps_bilateral_exemption_for_the_same_body_site(finding):
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        (finding, "影像检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"


def test_patient_context_does_not_hide_new_unilateral_disease_after_bilateral_finding():
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("双侧肾积水伴右肾肿瘤", "影像检查", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"laterality_conflict"}


def test_fixed_right_anatomic_diagnosis_overrides_random_left_patient_laterality():
    row = _base_row()
    row[utils.COL_DIAGNOSIS] = "急性阑尾炎"
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("右下腹压痛", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert result["patient"]["laterality"] == "右侧"


def test_mirror_anatomy_diagnosis_does_not_force_right_laterality_in_ledger():
    row = _base_row()
    row[utils.COL_DIAGNOSIS] = "急性阑尾炎伴内脏反位"
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    _set_objective_entries(row, [
        ("左侧阑尾区压痛", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert result["patient"]["laterality"] == "左侧"



def test_module_281_quarantines_context_direction_and_strict_time_blockers():
    row = _base_row()
    row[utils.COL_GENDER] = "男"
    row[utils.COL_STAGE] = "危重"
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    row[utils.COL_M25_OUTPUT] = json.dumps(_time_model(), ensure_ascii=False)
    _set_objective_entries(row, [
        ("妊娠试验阳性", "实验室检查", None, None),
        ("右侧肾积水", "影像检查", None, None),
        ("血清肌酐升高", "实验室检查", "{220,0,220,220}umol/L", "220umol/L"),
        ("eGFR升高", "实验室检查", "{120,0,120,120}mL/min/1.73m2", "120mL/min/1.73m2"),
    ])

    finalized_row, ledger = module_2_81_finalize_fact_ledger(
        row, _staging_system(), max_rounds=5
    )

    assert finalized_row[COL_M281_OUTPUT]["status"] == "quarantined"
    assert ledger["status"] == "quarantined"
    kinds = {blocker["kind"] for blocker in ledger["audit"]["blockers"]}
    assert {"sex_specific_fact", "laterality_conflict", "severity_level", "creatinine_egfr_direction"} <= kinds

    bad_time_ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())
    bad_time_ledger["facts"][0]["time"]["clinical_onset"] = "三天前"
    time_result = converge_fact_ledger(bad_time_ledger)
    assert time_result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in time_result["audit"]["blockers"]} == {"time_label"}



def test_converge_fact_ledger_keeps_acute_exercise_functional_fact_but_marks_feasibility():
    row = _base_row()
    row[utils.COL_ACUITY] = "急性"
    row[utils.COL_DIAGNOSIS] = "肺栓塞"
    _set_objective_entries(row, [
        ("休克", "体征", None, None),
        ("心肺运动试验峰值摄氧量降低", "功能检查", "{12,0,10,20}mL/kg/min", "12mL/kg/min"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    exercise_fact = _fact_by_canonical(result, "心肺运动试验峰值摄氧量")
    assert exercise_fact["clinical_state"] == "present"
    assert exercise_fact["feasibility"] == "clinically_questionable"
    assert result["status"] == "converged"


def test_converge_fact_ledger_quarantines_when_derived_value_cannot_fit_m27_range():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,0,140,140}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{35,0,35,40}mmol/L", "35mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "unrepairable"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"anion_gap"}


def test_converge_fact_ledger_detects_oscillating_state_hash(monkeypatch):
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())

    monkeypatch.setattr(
        pfl,
        "_validate_joint_constraints",
        lambda _ledger: [{"kind": "synthetic", "message": "still blocked", "repairable": True}],
    )
    monkeypatch.setattr(pfl, "_minimal_patch_ledger", lambda _ledger, _blockers: [])

    result = converge_fact_ledger(ledger, max_rounds=5)

    assert result["status"] == "quarantined"
    assert result["audit"]["stop_reason"] == "oscillation"


def test_quarantined_ledger_is_written_but_load_fails_closed(tmp_path):
    csv_path = tmp_path / "肺炎.csv"
    utils._save_csv(str(csv_path), [utils._empty_row(), _base_row()])
    ledger = build_fact_ledger(_base_row(), _staging_system(), _time_model())
    ledger["status"] = "quarantined"
    ledger["audit"]["blockers"] = [{"kind": "time_label", "message": "bad time", "repairable": False}]
    ledger["audit"]["stop_reason"] = "unrepairable"

    summary = write_fact_ledger(str(csv_path), 1, ledger)

    assert summary["status"] == "quarantined"
    with pytest.raises(ValueError, match="not converged"):
        load_fact_ledger(str(csv_path), 1, expected_case_id="case_00001")


def test_anion_gap_does_not_mix_cross_time_values():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,0,140,140}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,100,100}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,20,20}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{30,5,0,40}mmol/L", "30mmol/L"),
    ])
    ledger = build_fact_ledger(row, _staging_system(), _time_model())
    _fact_by_canonical(ledger, "碳酸氢根")["time"]["observed_at"] = "D-1"

    result = converge_fact_ledger(ledger)

    assert result["status"] == "converged"
    assert _fact_value(result, "阴离子间隙") == 30
    patch_kinds = [
        patch["kind"]
        for round_payload in result["audit"]["rounds"]
        for patch in round_payload.get("patches", [])
    ]
    assert "anion_gap" not in patch_kinds


def test_serum_anion_gap_excludes_urine_electrolytes_and_ag_like_names():
    row = _base_row()
    _set_objective_entries(row, [
        ("尿钠", "实验室检查", "{80,0,80,80}mmol/L", "80mmol/L"),
        ("尿氯", "实验室检查", "{40,0,40,40}mmol/L", "40mmol/L"),
        ("尿碳酸氢根", "实验室检查", "{10,0,10,10}mmol/L", "10mmol/L"),
        ("白蛋白/球蛋白A/G降低", "实验室检查", "{1.0,0,0.5,2.5}", "1.0"),
        ("阴离子间隙升高", "实验室检查", "{35,0,0,50}mmol/L", "35mmol/L"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "阴离子间隙") == 35


def test_red_cell_indices_normalize_hb_hct_and_rbc_common_units():
    row = _base_row()
    _set_objective_entries(row, [
        ("红细胞计数", "实验室检查", "{5,0,5,5}10^6/uL", "5 10^6/uL"),
        ("血红蛋白", "实验室检查", "{15,0,15,15}g/dL", "15g/dL"),
        ("红细胞压积", "实验室检查", "{0.45,0,0.45,0.45}L/L", "0.45L/L"),
        ("平均红细胞体积", "实验室检查", "{50,5,0,120}fL", "50fL"),
        ("平均红细胞血红蛋白量", "实验室检查", "{1,1,0,50}pg", "1pg"),
        ("平均红细胞血红蛋白浓度", "实验室检查", "{1,1,0,50}g/dL", "1g/dL"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "平均红细胞体积") == pytest.approx(90)
    assert _fact_value(result, "平均红细胞血红蛋白量") == pytest.approx(30)
    assert _fact_value(result, "平均红细胞血红蛋白浓度") == pytest.approx(33.333, abs=0.01)


def test_red_cell_indices_quarantine_unsupported_units_without_auto_patch():
    row = _base_row()
    _set_objective_entries(row, [
        ("红细胞计数", "实验室检查", "{5,0,5,5}10^12/L", "5 10^12/L"),
        ("血红蛋白", "实验室检查", "{9,0,9,9}kg/L", "9kg/L"),
        ("红细胞压积", "实验室检查", "{45,0,45,45}%", "45%"),
        ("平均红细胞血红蛋白量", "实验室检查", "{1,1,0,50}pg", "1pg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert _fact_value(result, "平均红细胞血红蛋白量") == 1
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"unsupported_unit"}


def test_bilirubin_triad_converts_mg_dl_and_umol_l_when_supported():
    row = _base_row()
    _set_objective_entries(row, [
        ("总胆红素升高", "实验室检查", "{3,0,3,3}mg/dL", "3mg/dL"),
        ("直接胆红素升高", "实验室检查", "{17.104,0,17.104,17.104}umol/L", "17.104umol/L"),
        ("间接胆红素升高", "实验室检查", "{1,1,0,5}mg/dL", "1mg/dL"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "间接胆红素") == pytest.approx(2, abs=0.01)


def test_metric_grouping_preserves_nonblood_oxygen_and_pulse_deficit_exclusions():
    row = _base_row()
    _set_objective_entries(row, [
        ("尿白细胞计数", "实验室检查", "{30,0,30,30}/HPF", "30/HPF"),
        ("尿红细胞计数", "实验室检查", "{20,0,20,20}/HPF", "20/HPF"),
        ("尿肌酐", "实验室检查", "{8000,0,8000,8000}umol/L", "8000umol/L"),
        ("eGFR升高", "实验室检查", "{120,0,120,120}mL/min/1.73m2", "120mL/min/1.73m2"),
        ("混合静脉血氧饱和度SvO2降低", "实验室检查", "{55,0,55,55}%", "55%"),
        ("动脉氧分压PaO2降低", "实验室检查", "{55,0,55,55}mmHg", "55mmHg"),
        ("心率增快", "体征", "{130,0,40,180}次/分", "130次/分"),
        ("脉率降低", "体征", "{60,0,40,180}次/分", "60次/分"),
        ("心电图心室率增快", "功能检查", "{128,0,40,180}次/分", "128次/分"),
        ("脉搏短绌", "体征", None, None),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert _fact_value(result, "脉率") == 60
    assert _fact_value(result, "eGFR") == 120
    assert _fact_value(result, "混合静脉血氧饱和度SvO2") == 55
    assert _fact_value(result, "动脉氧分压PaO2") == 55


def test_rederive_fact_ledger_recomputes_derived_quantities_after_source_change():
    row = _base_row()
    _set_objective_entries(row, [
        ("血清钠", "实验室检查", "{140,0,100,160}mmol/L", "140mmol/L"),
        ("血清氯", "实验室检查", "{100,0,80,120}mmol/L", "100mmol/L"),
        ("碳酸氢根", "实验室检查", "{20,0,10,40}mmol/L", "20mmol/L"),
        ("阴离子间隙升高", "实验室检查", "{20,0,0,40}mmol/L", "20mmol/L"),
    ])
    ledger = build_fact_ledger(row, _staging_system(), _time_model())
    _fact_by_canonical(ledger, "血清钠")["value"]["number"] = 130

    patches = pfl._rederive_fact_ledger(ledger)

    assert patches == [{
        "kind": "anion_gap",
        "fact_id": _fact_by_canonical(ledger, "阴离子间隙")["fact_id"],
        "from": 20,
        "to": 10,
        "basis": "derived_quantity_recalculation",
    }]
    assert _fact_value(ledger, "阴离子间隙") == 10



def test_actual_value_without_unit_inherits_reference_range_unit():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([("总胆红素升高", "III级", "", "", "")])
    row[utils.COL_TIME_ORDER] = repr([("总胆红素升高", "实验室检查", "就诊", "D0")])
    row[utils.COL_QUANTIFIED] = repr([
        ("总胆红素升高{40,5,21,nan}umol/L", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("总胆红素升高；实际值=38", "实验室检查", "就诊", "D0"),
    ])

    ledger = build_fact_ledger(row, _staging_system(), _time_model())
    bilirubin = _fact_by_canonical(ledger, "总胆红素")

    assert bilirubin["value"]["number"] == 38
    assert bilirubin["value"]["unit"] == "umol/L"
    assert bilirubin["value"]["reference_range"]["unit"] == "umol/L"


def test_converge_fact_ledger_quarantines_bilirubin_triad_without_safe_units():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("总胆红素升高", "III级", "", "", ""),
        ("直接胆红素升高", "III级", "", "", ""),
        ("间接胆红素升高", "III级", "", "", ""),
    ])
    row[utils.COL_TIME_ORDER] = repr([
        ("总胆红素升高", "实验室检查", "就诊", "D0"),
        ("直接胆红素升高", "实验室检查", "就诊", "D0"),
        ("间接胆红素升高", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("总胆红素升高{38,0,21,nan}", "实验室检查", "就诊", "D0"),
        ("直接胆红素升高{10,0,6,nan}", "实验室检查", "就诊", "D0"),
        ("间接胆红素升高{28,0,15,nan}", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("总胆红素升高；实际值=38", "实验室检查", "就诊", "D0"),
        ("直接胆红素升高；实际值=10", "实验室检查", "就诊", "D0"),
        ("间接胆红素升高；实际值=28", "实验室检查", "就诊", "D0"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"unsupported_unit"}

@pytest.mark.parametrize("generic_first", [False, True])
def test_present_generic_abnormal_merges_with_directional_finding(generic_first):
    row = _base_row()
    directional = ("血钙降低", "III级")
    generic = ("血钙异常", "伴随疾病")
    entries = [generic, directional] if generic_first else [directional, generic]
    _set_objective_entries(row, [])
    row[utils.COL_LAB_TESTS] = repr(entries)
    row[utils.COL_TIME_ORDER] = repr([
        ("血钙降低", "实验室检查", "就诊", "D0"),
        ("血钙异常", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("血钙降低{2,0.1,1.8,2.1}mmol/L", "实验室检查", "就诊", "D0"),
        ("血钙异常", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("血钙降低；实际值=1.94mmol/L", "实验室检查", "就诊", "D0"),
        ("血钙异常", "实验室检查", "就诊", "D0"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    calcium = _fact_by_canonical(result, "血钙")

    assert result["status"] == "converged"
    assert calcium["interpretation"] == "low"
    assert calcium["value"]["number"] == pytest.approx(1.94)


def test_creatinine_clearance_is_not_serum_creatinine():
    row = _base_row()
    _set_objective_entries(row, [
        ("估算肾小球滤过率降低", "实验室检查", "{8,1,5,10}mL/min/1.73m2", "8.39mL/min/1.73m2"),
        ("血肌酐升高", "实验室检查", "{900,100,600,1200}μmol/L", "991.16μmol/L"),
        ("内生肌酐清除率降低", "实验室检查", "{3,2,0.1,8}mL/min", "0.31mL/min"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    clearance = _fact_by_canonical(result, "内生肌酐清除率")
    clearance_record = next(
        record for record in pfl._metric_records(result)
        if record["metric"] == "creatinine_clearance"
    )


    assert pfl._metric_key(clearance) == "creatinine_clearance"
    assert clearance_record["unit_supported"] is True
    assert clearance_record["canonical_unit"] == "mL/min"
    assert result["status"] == "converged"


def test_spo2_sao2_quarantines_when_ranges_cannot_approach():
    row = _base_row()
    _set_objective_entries(row, [
        ("血氧饱和度SpO2降低", "体征", "{95,0.2,95,96}%", "95%"),
        ("动脉血氧饱和度SaO2降低", "实验室检查", "{88,1,80,90}%", "90%"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"spo2_sao2"}


@pytest.mark.parametrize(
    "first,second,metric",
    [
        (
            ("重度舒张压升高（DBP>110mmHg）", "{120,3,111,140}mmHg", "112mmHg"),
            ("舒张压升高（DBP>90mmHg）", "{105,8,91,140}mmHg", "100mmHg"),
            "diastolic_bp",
        ),
        (
            ("平均动脉压降低（MAP<65mmHg）", "{58,3,40,64}mmHg", "60mmHg"),
            ("低平均动脉压（MAP<70mmHg）", "{62,4,40,69}mmHg", "68mmHg"),
            "mean_arterial_pressure",
        ),
    ],
)
def test_repeated_dbp_and_map_aliases_share_one_ledger_value(first, second, metric):
    row = _base_row()
    _set_objective_entries(row, [
        (first[0], "体征", first[1], first[2]),
        (second[0], "体征", second[1], second[2]),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    records = [
        record for record in pfl._metric_records(result)
        if record["metric"] == metric
    ]

    assert result["status"] == "converged"
    assert len(records) == 2
    assert len({record["number"] for record in records}) == 1


def test_repeated_blood_pressure_values_stay_separate_between_arms():
    row = _base_row()
    _set_objective_entries(row, [
        ("左臂收缩压升高", "体征", "{182,0,182,182}mmHg", "182mmHg"),
        ("右臂收缩压升高", "体征", "{150,0,150,150}mmHg", "150mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert {
        record["number"]
        for record in pfl._metric_records(result)
        if record["metric"] == "systolic_bp"
    } == {150, 182}


@pytest.mark.parametrize("comorbidity_first", [False, True])
def test_mixed_main_and_comorbidity_provenance_keeps_laterality_gate(comorbidity_first):
    row = _base_row()
    row[utils.COL_PATIENT_LATERALITY] = "左侧"
    finding = "右肾积水"
    entries = [(finding, "伴随疾病"), (finding, "III级")]
    if not comorbidity_first:
        entries.reverse()
    _set_objective_entries(row, [])
    row[utils.COL_IMAGING] = repr(entries)

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {
        "laterality_conflict"
    }


def test_comorbidity_scope_does_not_disable_sex_specific_gate():
    row = _base_row()
    row[utils.COL_GENDER] = "男"
    _set_objective_entries(row, [])
    row[utils.COL_IMAGING] = repr([
        ("盆腔超声示宫内妊娠囊", "伴随疾病"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {
        "sex_specific_fact"
    }


def test_generic_abnormal_with_conflicting_numeric_value_quarantines_directional_fact():
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("血钾降低", "III级", "", "", ""),
        ("血钾异常", "伴随疾病", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("血钾降低", "实验室检查", "就诊", "D0"),
        ("血钾异常", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("血钾降低{3.1,0.1,2.8,3.4}mmol/L", "实验室检查", "就诊", "D0"),
        ("血钾异常{6.2,0.1,5.8,6.6}mmol/L", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("血钾降低；实际值=3.1mmol/L", "实验室检查", "就诊", "D0"),
        ("血钾异常；实际值=6.2mmol/L", "实验室检查", "就诊", "D0"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {
        "directional_value_conflict"
    }


@pytest.mark.parametrize("generic_unit", ["mmol/l", "mEq/L"])
def test_generic_abnormal_conflicting_value_uses_safe_unit_keys(generic_unit):
    row = _base_row()
    row[utils.COL_LAB_TESTS] = repr([
        ("血钾降低", "III级", "", "", ""),
        ("血钾异常", "伴随疾病", "", "", ""),
    ])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("血钾降低", "实验室检查", "就诊", "D0"),
        ("血钾异常", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("血钾降低{3.1,0.1,2.8,3.4}mmol/L", "实验室检查", "就诊", "D0"),
        (f"血钾异常{{6.2,0.1,5.8,6.6}}{generic_unit}", "实验室检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("血钾降低；实际值=3.1mmol/L", "实验室检查", "就诊", "D0"),
        (f"血钾异常；实际值=6.2{generic_unit}", "实验室检查", "就诊", "D0"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {
        "directional_value_conflict"
    }


def test_converge_fact_ledger_quarantines_far_duplicate_sbp_projection():
    row = _base_row()
    _set_objective_entries(row, [
        ("重度收缩压升高（SBP>180mmHg）", "体征", "{250,1,200,260}mmHg", "250mmHg"),
        ("收缩压升高（SBP>140mmHg）", "体征", "{150,1,140,200}mmHg", "150mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"same_metric_value"}


def test_converge_fact_ledger_quarantines_far_spo2_sao2_projection():
    row = _base_row()
    _set_objective_entries(row, [
        ("血氧饱和度SpO2降低", "体征", "{98,1,80,100}%", "98%"),
        ("动脉血氧饱和度SaO2降低", "实验室检查", "{60,1,50,77.5}%", "60%"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "quarantined"
    assert {blocker["kind"] for blocker in result["audit"]["blockers"]} == {"spo2_sao2"}


def test_repeated_blood_pressure_values_stay_separate_between_initial_and_repeat_measurements():
    row = _base_row()
    _set_objective_entries(row, [
        ("初测收缩压升高", "体征", "{182,0,182,182}mmHg", "182mmHg"),
        ("复测收缩压升高", "体征", "{150,0,150,150}mmHg", "150mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))

    assert result["status"] == "converged"
    assert {
        record["number"]
        for record in pfl._metric_records(result)
        if record["metric"] == "systolic_bp"
    } == {150, 182}

@pytest.mark.parametrize("name", [
    "双上肢收缩压差增大（差值>20mmHg）",
    "下肢收缩压降低（踝部收缩压较上臂低>20mmHg）",
    "一侧上肢收缩压较对侧低>15mmHg",
    "踝臂收缩压差增大（踝臂差>20mmHg）",
    "上下肢收缩压差异常（下肢较上肢低>20mmHg）",
    "血压波动增大（收缩压波动>30mmHg）",
    "体位性低血压（站立后收缩压下降>20mmHg）",
    "直立性低血压（站立后收缩压下降≥20mmHg）",
    "收缩期血压体位性下降（站立3分钟收缩压下降≥20mmHg）",
    "血压进行性下降（收缩压较基础值下降≥40mmHg）",
    "脉搏奇异（吸气时收缩压下降>10mmHg）",
    "24小时动态血压示收缩压负荷升高",
    "吞咽压力测定示咽部收缩压力降低",
    "肺动脉收缩压升高（PASP>50mmHg）",
    "右心室收缩压升高（>40mmHg）",
    "心室收缩压升高（>140mmHg）",
    "食管收缩压降低（<30mmHg）",
    "括约肌收缩压降低（<40mmHg）",
    "膀胱收缩压升高（>40mmHg）",
    "尿道收缩压降低（<40mmHg）",
    "宫缩压力升高（>60mmHg）",
])
def test_bp_difference_fact_is_not_ledger_absolute_sbp_metric(name):
    fact = {"concept": {"raw": name, "canonical": name}, "domain": "sign"}

    assert pfl._metric_key(fact) is None


def test_bp_difference_facts_are_kept_but_not_forced_to_share_absolute_sbp_value():
    row = _base_row()
    relative_name = "双上肢收缩压差增大（差值>20mmHg）"
    ankle_name = "下肢收缩压降低（踝部收缩压较上臂低>20mmHg）"
    _set_objective_entries(row, [
        ("收缩压升高", "体征", "{160,0,160,160}mmHg", "160mmHg"),
        (relative_name, "体征", "{25,0,25,25}mmHg", "25mmHg"),
        (ankle_name, "体征", "{30,0,30,30}mmHg", "30mmHg"),
    ])

    result = converge_fact_ledger(build_fact_ledger(row, _staging_system(), _time_model()))
    sbp_records = [
        record for record in pfl._metric_records(result)
        if record["metric"] == "systolic_bp"
    ]

    assert result["status"] == "converged"
    assert _fact_by_raw(result, relative_name)["value"]["number"] == 25
    assert _fact_by_raw(result, ankle_name)["value"]["number"] == 30
    assert [record["number"] for record in sbp_records] == [160]
