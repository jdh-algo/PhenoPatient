import pytest

import run_phenopatient as runner

def test_seed_builder_rejects_mixed_acuity_group():
    cases = [
        {"age": 50, "gender": "男", "acuity": "急性", "icd_code": ""},
        {"age": 60, "gender": "女", "acuity": "慢性", "icd_code": ""},
    ]

    with pytest.raises(ValueError, match="急慢性"):
        runner._seed_text_for_cases("肺炎", cases)

def test_nonempty_invalid_acuity_is_rejected():
    with pytest.raises(ValueError, match="急慢性"):
        runner._normalize_acuity("亚急性", "急性")

def test_missing_acuity_uses_default():
    assert runner._normalize_acuity("", "慢性") == "慢性"

def test_injected_stage_must_exist_in_m1_levels():
    staging_system = {
        "name": "临床严重度分级",
        "levels": [
            {"level": 1, "name": "轻度", "description": "", "proportion": 0.5},
            {"level": 2, "name": "重度", "description": "", "proportion": 0.5},
        ],
    }

    with pytest.raises(ValueError, match="时期.*分级系统"):
        runner._validate_injected_stage("极重度", staging_system)

def test_injected_stage_accepts_unique_m1_level_name():
    staging_system = {
        "name": "临床严重度分级",
        "levels": [
            {"level": 1, "name": "轻度", "description": "", "proportion": 0.5},
            {"level": 2, "name": "重度", "description": "", "proportion": 0.5},
        ],
    }

    assert runner._validate_injected_stage("重度", staging_system) == "重度"


def test_injected_stage_rejects_duplicate_m1_level_names():
    staging_system = {
        "name": "临床严重度分级",
        "levels": [
            {"level": 1, "name": "重度", "description": "较重", "proportion": 0.5},
            {"level": 2, "name": "重度", "description": "更重", "proportion": 0.5},
        ],
    }

    with pytest.raises(ValueError, match="时期.*分级系统"):
        runner._validate_injected_stage("重度", staging_system)

def test_run_phenopatient_refuses_to_start_m3_when_fact_ledger_missing(monkeypatch, tmp_path):
    import os
    import sys
    import types
    import utils

    input_file = tmp_path / "patients.json"
    input_file.write_text(
        '[{"case_id":"case_1","age":60,"gender":"男","diagnosis":"肺炎","acuity":"急性"}]',
        encoding="utf-8",
    )

    def fake_process_seed_module1(seed_text, output_dir, **_kwargs):
        row_1 = utils._empty_row()
        row_1[utils.COL_SEED] = seed_text
        row_1[utils.COL_STAGING_SYSTEM] = str({
            "name": "临床严重度分级",
            "levels": [
                {"level": 1, "name": "轻度", "description": "症状轻", "proportion": 0.6},
                {"level": 2, "name": "重度", "description": "症状重", "proportion": 0.4},
            ],
        })
        utils._save_csv(os.path.join(output_dir, "肺炎.csv"), [row_1])
        return {"status": "success"}

    def fake_run_module2(csv_file, **_kwargs):
        return {"status": "success", "file": csv_file}

    m3_called = {"value": False}

    def forbidden_run_module345(*_args, **_kwargs):
        m3_called["value"] = True
        raise AssertionError("M3 started without fact ledger preflight")

    monkeypatch.setitem(sys.modules, "phenotypic_atlas", types.SimpleNamespace(
        process_seed_module1=fake_process_seed_module1,
    ))
    monkeypatch.setitem(sys.modules, "atlas_based_patient_generation", types.SimpleNamespace(
        run_module2=fake_run_module2,
    ))
    monkeypatch.setitem(sys.modules, "virtual_clinical_interaction", types.SimpleNamespace(
        run_module345=forbidden_run_module345,
    ))
    monkeypatch.setitem(sys.modules, "check_and_reset_seed", types.SimpleNamespace(
        check_and_reset=lambda *_args, **_kwargs: None,
    ))
    monkeypatch.setattr(sys, "argv", [
        "run_phenopatient.py",
        "--input_file", str(input_file),
        "--output_root", str(tmp_path / "out"),
    ])

    with pytest.raises((RuntimeError, SystemExit), match="fact ledger|账本"):
        runner.main()
    assert not m3_called["value"]


def _template_row(label):
    import utils

    row = utils._empty_row()
    row[utils.COL_SEED] = "肺炎 # 60-60 1.00 急性"
    row[utils.COL_STAGING_SYSTEM] = str({
        "name": f"{label}分级",
        "levels": [
            {"level": 1, "name": "轻度", "description": label, "proportion": 0.6},
            {"level": 2, "name": "重度", "description": label, "proportion": 0.4},
        ],
    })
    row[utils.COL_M10_OUTPUT] = f"sidecar-{label}"
    return row


def _patient_progress_row():
    import utils

    row = utils._empty_row()
    row[utils.COL_CASE_ID] = "case_existing"
    row[utils.COL_SEED] = "肺炎 # 60-60 1.00 急性"
    row[utils.COL_AGE] = "60"
    row[utils.COL_GENDER] = "男"
    row[utils.COL_STAGE] = "轻度"
    row[utils.COL_CHIEF_COMPLAINT] = "old progress"
    row[utils.COL_M25_OUTPUT] = "patient-sidecar"
    return row


def test_copy_csv_and_sidecar_same_path_is_noop(tmp_path):
    import utils
    from pathlib import Path

    csv_path = tmp_path / "same.csv"
    utils._save_csv(str(csv_path), [_template_row("same")])
    marker = Path(str(csv_path) + ".module_io") / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    runner._copy_csv_and_sidecar(str(csv_path), str(csv_path))

    assert csv_path.exists()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_sync_csv_template_refreshes_stale_target_and_discards_old_patient_progress(tmp_path):
    import utils
    from pathlib import Path

    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    utils._save_csv(str(src), [_template_row("new")])
    utils._save_csv(str(dst), [_template_row("old"), _patient_progress_row()])
    stale_marker = Path(str(dst) + ".module_io") / "row_1.json"
    assert stale_marker.exists()

    action = runner._sync_csv_template_and_sidecar(str(src), str(dst))

    row_1, patients = utils._load_existing_csv(str(dst))
    assert action == "refreshed"
    assert "new" in row_1[utils.COL_STAGING_SYSTEM]
    assert not patients
    assert not stale_marker.exists()
    assert (Path(str(dst) + ".module_io") / "row_0.json").exists()


def test_sync_csv_template_preserves_patient_progress_when_source_unchanged(tmp_path):
    import utils
    from pathlib import Path

    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    template = _template_row("same")
    utils._save_csv(str(src), [template])
    utils._save_csv(str(dst), [dict(template), _patient_progress_row()])
    patient_sidecar = Path(str(dst) + ".module_io") / "row_1.json"
    before = patient_sidecar.read_text(encoding="utf-8")

    action = runner._sync_csv_template_and_sidecar(str(src), str(dst))

    _row_1, patients = utils._load_existing_csv(str(dst))
    assert action == "unchanged"
    assert patients[1][utils.COL_CHIEF_COMPLAINT] == "old progress"
    assert patient_sidecar.read_text(encoding="utf-8") == before


def _write_inline_csv(path, rows):
    import csv
    import utils

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=utils.CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in utils.CSV_COLUMNS})


def test_sync_csv_template_treats_inline_and_sidecar_storage_as_equivalent(tmp_path):
    import utils
    from pathlib import Path

    src = tmp_path / "src_inline.csv"
    dst = tmp_path / "dst_pointer.csv"
    template = _template_row("same")
    _write_inline_csv(src, [template])
    utils._save_csv(str(dst), [dict(template), _patient_progress_row()])
    patient_sidecar = Path(str(dst) + ".module_io") / "row_1.json"
    assert not Path(str(src) + ".module_io").exists()
    before = patient_sidecar.read_text(encoding="utf-8")

    action = runner._sync_csv_template_and_sidecar(str(src), str(dst))

    _row_1, patients = utils._load_existing_csv(str(dst))
    assert action == "unchanged"
    assert patients[1][utils.COL_CHIEF_COMPLAINT] == "old progress"
    assert patient_sidecar.read_text(encoding="utf-8") == before
    assert not Path(str(src) + ".module_io").exists()


def test_template_signature_does_not_create_module_io_for_inline_source(tmp_path):
    from pathlib import Path

    src = tmp_path / "src_inline.csv"
    _write_inline_csv(src, [_template_row("inline")])
    assert not Path(str(src) + ".module_io").exists()

    signature = runner._template_signature(str(src))

    assert signature["row_0_extra_sidecar"] == {}
    assert not Path(str(src) + ".module_io").exists()
