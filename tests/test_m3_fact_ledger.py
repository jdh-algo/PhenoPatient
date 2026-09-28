import json
from pathlib import Path

import pytest

import patient_fact_ledger as pfl
import utils
import virtual_clinical_interaction as m3


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


def _base_row(case_id="case_00001"):
    row = utils._empty_row()
    row[utils.COL_CASE_ID] = case_id
    row[utils.COL_SEED] = "慢性心力衰竭"
    row[utils.COL_AGE] = "68"
    row[utils.COL_GENDER] = "男"
    row[utils.COL_DIAGNOSIS] = "慢性心力衰竭"
    row[utils.COL_STAGE] = "III级"
    row[utils.COL_PATIENT_LATERALITY] = "不适用"
    row[utils.COL_ACUITY] = "慢性急性加重"
    row[utils.COL_DURATION_TOTAL] = "5天"
    row[utils.COL_CHIEF_COMPLAINT] = "胸闷5天"
    row[utils.COL_SYMPTOMS] = repr([
        ("胸闷", "III级", "3小时", "活动后", "持续"),
    ])
    row[utils.COL_SIGNS] = repr([
        ("颈静脉怒张", "III级", "", "", ""),
    ])
    row[utils.COL_LAB_TESTS] = repr([
        ("白细胞计数升高", "III级", "", "", ""),
        ("血红蛋白降低", "III级", "", "", ""),
    ])
    row[utils.COL_IMAGING] = repr([])
    row[utils.COL_FUNCTIONAL_TESTS] = repr([
        ("6MWT缩短", "III级", "", "", ""),
    ])
    row[utils.COL_ABSENT_SYMPTOMS] = repr(["发热"])
    row[utils.COL_ABSENT_SIGNS] = repr(["肺部湿啰音"])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_ABSENT_IMAGING] = repr([])
    row[utils.COL_ABSENT_FUNCTIONAL] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        ("胸闷", "症状", "起病", "H-3"),
        ("颈静脉怒张", "体征", "就诊", "D0"),
        ("白细胞计数升高", "实验室检查", "就诊", "D0"),
        ("血红蛋白降低", "实验室检查", "就诊", "D0"),
        ("6MWT缩短", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("白细胞计数升高{12,2,10,nan}×10^9/L", "实验室检查", "就诊", "D0"),
        ("血红蛋白降低{105,10,nan,120}g/L", "实验室检查", "就诊", "D0"),
        ("6MWT缩短{280,30,nan,400}m", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("胸闷", "症状", "起病", "H-3"),
        ("颈静脉怒张", "体征", "就诊", "D0"),
        ("白细胞计数升高；实际值=13.2×10^9/L", "实验室检查", "就诊", "D0"),
        ("血红蛋白降低；实际值=105g/L", "实验室检查", "就诊", "D0"),
        ("6MWT缩短；实际值=280m", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_M25_OUTPUT] = json.dumps(_time_model(), ensure_ascii=False)
    return row


def _save_case_csv(tmp_path, row):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    row_1 = utils._empty_row()
    row_1[utils.COL_STAGING_SYSTEM] = json.dumps(_staging_system(), ensure_ascii=False)
    utils._save_csv(str(csv_path), [row_1, row])
    return csv_path, row_1


def _write_valid_ledger(csv_path, row, status="converged"):
    staging_system = utils.parse_staging_system(json.dumps(_staging_system(), ensure_ascii=False))
    ledger = pfl.build_fact_ledger(row, staging_system, _time_model())
    ledger["status"] = status
    ledger.setdefault("audit", {}).setdefault("rounds", [])
    return pfl.write_fact_ledger(str(csv_path), 1, ledger)


def _manual_fact(domain, raw, canonical=None, value=None, unit=None,
                 clinical_state="present", interpretation=None,
                 feasibility="routine"):
    return {
        "fact_id": f"{domain}:{canonical or raw}",
        "domain": domain,
        "concept": {"raw": raw, "canonical": canonical or raw},
        "clinical_state": clinical_state,
        "interpretation": interpretation,
        "acquisition_state": "latent" if domain != "symptom" else "historical",
        "value": None if value is None else {
            "number": value,
            "unit": unit,
            "reference_range": None,
        },
        "time": {"clinical_onset": None, "observed_at": "D0", "phase": "就诊"},
        "specimen": None,
        "body_site": None,
        "laterality": None,
        "context": "current_visit",
        "visibility": ["diagnostic_oracle", "doctor_after_order"],
        "feasibility": feasibility,
        "provenance": {"source": "test", "source_index": 0},
        "derived_from": [],
    }


def _manual_ledger(facts, acuity="急性", diagnosis="急性心肌梗死"):
    return {
        "schema_version": "fact_ledger.v1",
        "module": "2.81",
        "status": "converged",
        "case_id": "case_manual",
        "patient": {
            "age": 60,
            "sex": "男",
            "diagnosis": diagnosis,
            "severity_scheme": "分级",
            "severity_level": "重度",
            "laterality": "不适用",
            "acuity": acuity,
        },
        "course": {
            "underlying_duration": "2年" if acuity != "急性" else None,
            "current_episode_duration": "3小时",
            "anchor": "current_visit",
        },
        "facts": facts,
        "relations": [],
        "audit": {"rounds": [], "final_hash": "fixture"},
        "provenance": {"source_row_hash": "fixture"},
    }


def test_patient_view_hides_diagnostic_values_and_full_specific_text():
    ledger = _manual_ledger([
        _manual_fact("symptom", "胸痛", clinical_state="present"),
        _manual_fact("lab", "肌钙蛋白I升高", "肌钙蛋白I", 1.8, "ng/mL", interpretation="high"),
        _manual_fact("imaging", "胸部CT磨玻璃影", "胸部CT"),
        _manual_fact("functional", "心电图ST段抬高", "心电图"),
    ])

    view = pfl.project_patient_view(ledger)
    rendered = json.dumps(view, ensure_ascii=False)

    assert "胸痛" in rendered
    assert "1.8" not in rendered
    assert "ng/mL" not in rendered
    assert "磨玻璃影" not in rendered
    assert "ST段抬高" not in rendered
    assert "specific_text" not in view


def test_physical_exam_view_and_renderer_do_not_invent_unknown_positive_signs():
    ledger = _manual_ledger([
        _manual_fact("sign", "颈静脉怒张", "颈静脉怒张"),
        _manual_fact("lab", "NT-proBNP升高", "NT-proBNP", 1850, "pg/mL", interpretation="high"),
    ], acuity="慢性急性加重", diagnosis="慢性心力衰竭")

    view = pfl.project_physical_exam_view(ledger)
    result = m3.render_physical_exam_result(view, "肺部听诊")

    assert "颈静脉怒张" in json.dumps(view, ensure_ascii=False)
    assert "NT-proBNP" not in json.dumps(view, ensure_ascii=False)
    assert "肺部湿啰音" not in result
    assert "双肺湿啰音" not in result



def test_physical_exam_renderer_does_not_treat_murphy_sign_as_pulse():
    ledger = _manual_ledger([
        _manual_fact("sign", "Murphy征阳性", "Murphy征"),
    ], acuity="急性", diagnosis="急性胆囊炎")

    view = pfl.project_physical_exam_view(ledger)
    result = m3.render_physical_exam_result(view, "腹部触诊")

    assert "P: 75次/分" in result
    assert "Murphy征阳性: 阳性/异常" in result


def test_physical_exam_renderer_scopes_findings_to_explicit_request():
    ledger = _manual_ledger([
        _manual_fact("sign", "肺部湿啰音", "肺部湿啰音"),
        _manual_fact("sign", "颈静脉怒张", "颈静脉怒张"),
        _manual_fact("sign", "Murphy征阳性", "Murphy征"),
    ], acuity="急性", diagnosis="急性胆囊炎")

    view = pfl.project_physical_exam_view(ledger)
    lung_result = m3.render_physical_exam_result(view, "肺部听诊")
    unknown_result = m3.render_physical_exam_result(view, "眼底检查")

    assert "肺部湿啰音" in lung_result
    assert "颈静脉怒张" not in lung_result
    assert "Murphy征" not in lung_result
    assert "肺部湿啰音" not in unknown_result
    assert "颈静脉怒张" not in unknown_result
    assert "Murphy征" not in unknown_result


def test_lung_function_request_returns_functional_panel_metrics():
    ledger = _manual_ledger([
        _manual_fact("functional", "FEV1降低", "FEV1", 55, "%pred", interpretation="low"),
        _manual_fact("functional", "FVC降低", "FVC", 62, "%pred", interpretation="low"),
        _manual_fact("functional", "DLCO降低", "DLCO", 48, "%pred", interpretation="low"),
        _manual_fact("functional", "心电图ST段抬高", "心电图", None, None, interpretation="abnormal"),
    ], acuity="慢性", diagnosis="慢性阻塞性肺疾病")

    view = pfl.project_diagnostic_view(ledger, "[申请化验]：肺功能")
    rendered = m3.render_diagnostic_view(view)

    assert "FEV1" in rendered
    assert "FVC" in rendered
    assert "DLCO" in rendered
    assert "心电图" not in rendered


def test_cbc_panel_excludes_hba1c_but_keeps_true_hemoglobin():
    ledger = _manual_ledger([
        _manual_fact("lab", "血红蛋白降低", "血红蛋白", 105, "g/L", interpretation="low"),
        _manual_fact("lab", "糖化血红蛋白升高", "HbA1c", 8.2, "%", interpretation="high"),
    ], acuity="慢性", diagnosis="糖尿病")

    view = pfl.project_diagnostic_view(ledger, "[申请化验]：血常规")
    rendered = m3.render_diagnostic_view(view)

    assert "血红蛋白" in rendered
    assert "105g/L" in rendered
    assert "糖化血红蛋白" not in rendered
    assert "HbA1c" not in rendered
    assert "8.2%" not in rendered


def test_built_ledger_patient_view_carries_patient_context_and_hashes_prior_history():
    row = _base_row()
    row[utils.COL_PRIOR_VISITED] = "就诊过"
    row[utils.COL_PRIOR_VISIT_COUNT] = "2"
    row[utils.COL_PRIOR_VISIT_HISTORY] = "账本既往就诊：上周急诊做过心电图。"
    staging_system = utils.parse_staging_system(json.dumps(_staging_system(), ensure_ascii=False))

    ledger = pfl.build_fact_ledger(row, staging_system, _time_model())
    view = pfl.project_patient_view(ledger)
    rendered = json.dumps(view, ensure_ascii=False)
    changed_row = dict(row)
    changed_row[utils.COL_PRIOR_VISIT_HISTORY] = "账本既往就诊：改为门诊复查。"

    assert "活动后" in rendered
    assert "持续" in rendered
    assert "账本既往就诊" in rendered
    assert ledger["provenance"]["source_row_hash"] != pfl.ledger_input_hash(
        changed_row, staging_system, _time_model()
    )


def test_ledger_backed_patient_prompt_uses_ledger_context_not_csv_args(monkeypatch):
    ledger = _manual_ledger([
        _manual_fact("symptom", "胸痛", "胸痛", clinical_state="present"),
    ], acuity="急性", diagnosis="急性心肌梗死")
    ledger["patient_context"] = {
        "symptom_attributes": [
            {"name": "胸痛", "duration": "1小时", "trigger": "账本爬楼后", "nature": "压榨样"},
        ],
        "prior_visit": {
            "visited": "就诊过",
            "count": 1,
            "history": "账本既往：社区医院做过心电图。",
        },
    }
    patient_prompts = []

    def fake_call(prompt, tag=""):
        if tag.endswith("医生-开场"):
            return "哪里不舒服？"
        if "患者-1" in tag:
            patient_prompts.append(prompt)
            return "胸口疼。"
        if tag.endswith("医生-2"):
            return "[申请化验]：心电图"
        if tag.endswith("医生-评估"):
            return "[诊断]: 急性心肌梗死"
        raise AssertionError(f"unexpected GPT call: {tag}")

    monkeypatch.setattr(m3, "call_gpt5", fake_call)
    m3.module_3_interaction(
        seed_text="急性心肌梗死",
        age=60,
        gender="男",
        diagnosis="急性心肌梗死",
        specific_text="CSV_ONLY_SPECIFIC",
        comorbidities_text="[]",
        symptom_attrs_text="- 胸痛：CSV_ONLY_ATTR",
        prior_visited="就诊过",
        prior_visit_count=3,
        prior_visit_history="CSV_ONLY_HISTORY",
        max_turns=1,
        max_followup_turns=0,
        fact_ledger=ledger,
        tag="测试",
    )

    prompt = patient_prompts[0]
    assert "账本爬楼后" in prompt
    assert "压榨样" in prompt
    assert "账本既往" in prompt
    assert "CSV_ONLY" not in prompt


def test_fullwidth_request_markers_drive_module3_and_cap_diagnostics(monkeypatch):
    ledger = _manual_ledger([
        _manual_fact("sign", "肺部湿啰音", "肺部湿啰音"),
        _manual_fact("lab", "白细胞计数升高", "白细胞计数", 13.2, "×10^9/L", interpretation="high"),
        _manual_fact("functional", "心电图ST段抬高", "心电图", None, None, interpretation="abnormal"),
    ], acuity="急性", diagnosis="急性心肌梗死")

    def fake_call(prompt, tag=""):
        if tag.endswith("医生-开场"):
            return "哪里不舒服？"
        if "患者-1" in tag:
            return "胸痛。"
        if tag.endswith("医生-2"):
            return (
                "【申请查体】：肺部听诊\n"
                "【申请化验】：血常规、心电图、胸部CT、肝功能、肾功能、电解质、凝血功能、肌钙蛋白、尿常规"
            )
        if tag.endswith("医生-评估"):
            return "[诊断]: 急性心肌梗死"
        raise AssertionError(f"ledger-backed result role should not call GPT tag: {tag}")

    monkeypatch.setattr(m3, "call_gpt5", fake_call)
    history = m3.module_3_interaction(
        seed_text="急性心肌梗死",
        age=60,
        gender="男",
        diagnosis="急性心肌梗死",
        specific_text="[]",
        comorbidities_text="[]",
        max_turns=1,
        max_followup_turns=0,
        fact_ledger=ledger,
        tag="测试",
    )

    assert "肺部湿啰音" in history
    assert "白细胞计数" in history
    assert "心电图ST段抬高" in history
    assert "尿常规" not in history


def test_only_cbc_request_returns_lab_section_and_functional_none():
    ledger = _manual_ledger([
        _manual_fact("lab", "白细胞计数升高", "白细胞计数", 13.2, "×10^9/L", interpretation="high"),
        _manual_fact("lab", "血红蛋白降低", "血红蛋白", 105, "g/L", interpretation="low"),
        _manual_fact("functional", "6MWT缩短", "6MWT", 280, "m", interpretation="low"),
    ], acuity="慢性急性加重", diagnosis="慢性心力衰竭")

    view = pfl.project_diagnostic_view(ledger, "[申请化验]：血常规")
    rendered = m3.render_diagnostic_view(view)

    assert "【化验检查】" in rendered
    assert "白细胞计数" in rendered
    assert "13.2" in rendered
    assert "血红蛋白" in rendered
    assert "【影像检查】\n- 无" in rendered
    assert "【功能学检查】\n- 无" in rendered
    assert "6MWT" not in rendered


def test_explicit_acute_mi_ecg_request_returns_questionable_ledger_value():
    ledger = _manual_ledger([
        _manual_fact(
            "functional", "心电图ST段抬高", "心电图", None, None,
            interpretation="abnormal", feasibility="clinically_questionable"
        ),
    ], acuity="急性", diagnosis="急性心肌梗死")

    view = pfl.project_diagnostic_view(ledger, "[申请化验]：心电图")
    rendered = m3.render_diagnostic_view(view)

    assert "【功能学检查】" in rendered
    assert "心电图" in rendered
    assert "ST段抬高" in rendered
    assert "临床可行性需结合场景" in rendered


def test_explicit_acute_worsening_hf_6mwt_request_returns_ledger_value():
    ledger = _manual_ledger([
        _manual_fact(
            "functional", "6MWT缩短", "6MWT", 280, "m",
            interpretation="low", feasibility="clinically_questionable"
        ),
    ], acuity="慢性急性加重", diagnosis="慢性心力衰竭")

    view = pfl.project_diagnostic_view(ledger, "[申请化验]：6MWT")
    rendered = m3.render_diagnostic_view(view)

    assert "6MWT" in rendered
    assert "280m" in rendered
    assert "【化验检查】\n- 无" in rendered
    assert "【影像检查】\n- 无" in rendered


def test_route_clinical_requests_splits_exam_and_diagnostic_domains_and_caps_items():
    request = (
        "[申请查体]：心肺听诊、腹部触诊\n"
        "[申请化验]: 请检测：血常规、心电图、胸部CT、肝功能、肾功能、电解质、凝血功能、肌钙蛋白、尿常规"
    )

    routed = m3.route_clinical_requests(request)

    assert routed["physical_exam"] == ["心肺听诊", "腹部触诊"]
    assert routed["lab"] == ["血常规", "肝功能", "肾功能", "电解质", "凝血功能", "肌钙蛋白"]
    assert routed["imaging"] == ["胸部CT"]
    assert routed["functional"] == ["心电图"]
    assert sum(len(routed[key]) for key in ("lab", "imaging", "functional")) == 8


@pytest.mark.parametrize(
    "mode, error_pattern",
    [
        ("missing", "fact ledger"),
        ("bad_hash", "ledger_hash"),
        ("stale_input", "stale"),
        ("nonconverged", "not converged"),
    ],
)
def test_process_csv_module345_fails_closed_for_invalid_ledgers(monkeypatch, tmp_path, mode, error_pattern):
    def fail_if_model_flow_starts(**_kwargs):
        raise AssertionError("invalid ledger must fail before M3 interaction starts")

    monkeypatch.setattr(m3, "module_3_interaction", fail_if_model_flow_starts)

    row = _base_row()
    csv_path, _row_1 = _save_case_csv(tmp_path, row)

    if mode == "bad_hash":
        _write_valid_ledger(csv_path, row)
        sidecar = Path(str(csv_path) + ".module_io") / "row_1.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload[pfl.COL_M281_OUTPUT]["ledger_hash"] = "bad"
        sidecar.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    elif mode == "stale_input":
        _write_valid_ledger(csv_path, row)
        row[utils.COL_STAGE] = "I级"
        row_1 = utils._empty_row()
        row_1[utils.COL_STAGING_SYSTEM] = json.dumps(_staging_system(), ensure_ascii=False)
        utils._save_csv(str(csv_path), [row_1, row])
    elif mode == "nonconverged":
        _write_valid_ledger(csv_path, row, status="quarantined")

    result = m3.process_csv_module345(str(csv_path), num_workers=1)

    assert result["status"] == "error"
    assert error_pattern in result["error"]


def test_completed_m3_without_ledger_metadata_is_rerun_and_metadata_is_persisted(monkeypatch, tmp_path):
    row = _base_row()
    row[utils.COL_INTERACTION] = "[医生]: 哪里不舒服？\n[患者]: 胸闷。\n[医生诊断]: [诊断]: 慢性心力衰竭"
    csv_path, _row_1 = _save_case_csv(tmp_path, row)
    summary = _write_valid_ledger(csv_path, row)
    calls = []

    def fake_interaction(**kwargs):
        calls.append(kwargs)
        assert kwargs["fact_ledger"]["case_id"] == row[utils.COL_CASE_ID]
        return "[医生]: 哪里不舒服？\n[患者]: 胸闷。\n[医生诊断]: [诊断]: 慢性心力衰竭"

    monkeypatch.setattr(m3, "module_3_interaction", fake_interaction)
    result = m3.process_csv_module345(str(csv_path), num_workers=1)

    assert result["status"] == "success"
    assert len(calls) == 1
    sidecar = Path(str(csv_path) + ".module_io") / "row_1.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload[m3.M3_LEDGER_METADATA_KEY]["ledger_hash"] == summary["ledger_hash"]
    assert payload[m3.M3_LEDGER_METADATA_KEY]["path"] == summary["path"]



def _save_three_m3_ready_rows(tmp_path):
    csv_path = tmp_path / "慢性心力衰竭.csv"
    row_1 = utils._empty_row()
    row_1[utils.COL_STAGING_SYSTEM] = json.dumps(_staging_system(), ensure_ascii=False)
    rows = [_base_row(f"case_{idx:05d}") for idx in range(1, 4)]
    utils._save_csv(str(csv_path), [row_1] + rows)
    staging_system = utils.parse_staging_system(row_1[utils.COL_STAGING_SYSTEM])
    for row_index, row in enumerate(rows, start=1):
        ledger = pfl.build_fact_ledger(row, staging_system, _time_model())
        ledger["status"] = "converged"
        ledger.setdefault("audit", {}).setdefault("rounds", [])
        pfl.write_fact_ledger(str(csv_path), row_index, ledger)
    return csv_path, rows


def _assert_csv_preserves_case_order(csv_path, expected_case_ids):
    _row_1, loaded = utils._load_existing_csv(str(csv_path))
    assert sorted(loaded) == list(range(1, len(expected_case_ids) + 1))
    assert [loaded[idx][utils.COL_CASE_ID] for idx in range(1, len(expected_case_ids) + 1)] == expected_case_ids
    for idx, case_id in enumerate(expected_case_ids, start=1):
        pfl.load_fact_ledger(str(csv_path), idx, expected_case_id=case_id)


def test_process_csv_module345_preserves_all_existing_rows_when_patient3_updates_first_then_errors(monkeypatch, tmp_path):
    csv_path, rows = _save_three_m3_ready_rows(tmp_path)
    expected_case_ids = [row[utils.COL_CASE_ID] for row in rows]
    snapshots = []
    real_save = utils._save_csv

    def recording_save(path, all_rows):
        snapshots.append([row.get(utils.COL_CASE_ID, "") for row in all_rows[1:]])
        real_save(path, all_rows)

    def fake_worker(task):
        patient_idx = task[1]
        row_data = dict(task[2])
        update_queue = task[-1]
        if patient_idx == 3:
            row_data[utils.COL_INTERACTION] = "patient3 partial interaction"
            update_queue.put((patient_idx, dict(row_data)))
            row_data[m3._M345_ERROR_ROW_KEY] = "forced patient3 failure"
        return row_data

    monkeypatch.setattr(m3, "_save_csv", recording_save)
    monkeypatch.setattr(m3, "_generate_single_patient_m345", fake_worker)

    result = m3.process_csv_module345(str(csv_path), num_workers=3)

    assert result["status"] == "error"
    assert snapshots, "expected at least one intermediate save"
    assert all(snapshot == expected_case_ids for snapshot in snapshots)
    _assert_csv_preserves_case_order(csv_path, expected_case_ids)


def test_batch_m345_update_preserves_all_existing_rows_when_patient3_updates_first(monkeypatch, tmp_path):
    csv_path, rows = _save_three_m3_ready_rows(tmp_path)
    expected_case_ids = [row[utils.COL_CASE_ID] for row in rows]
    state, tasks = m3._prepare_csv_m345(
        str(csv_path),
        max_interaction_turns=1,
        max_followup_turns=0,
        global_queue=m3._queue_module.Queue(),
    )
    assert state is not None
    assert [task[1] for task in tasks] == [1, 2, 3]

    snapshots = []
    real_save = utils._save_csv

    def recording_save(path, all_rows):
        snapshots.append([row.get(utils.COL_CASE_ID, "") for row in all_rows[1:]])
        real_save(path, all_rows)

    monkeypatch.setattr(m3, "_save_csv", recording_save)
    updated = dict(rows[2])
    updated[utils.COL_INTERACTION] = "patient3 partial interaction"

    m3._apply_update_m345(state, 3, updated)

    assert snapshots[-1] == expected_case_ids
    _assert_csv_preserves_case_order(csv_path, expected_case_ids)
