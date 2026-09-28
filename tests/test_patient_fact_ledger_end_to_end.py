import json

import pytest
import requests

import atlas_based_patient_generation as m2
import patient_fact_ledger as pfl
import phenotypic_atlas as m1
import utils


@pytest.fixture(autouse=True)
def block_network_and_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            f"no-API smoke attempted a network/model call: args={args!r}, kwargs={kwargs!r}"
        )

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)

    for module in (utils, m1, m2):
        for name in (
            "call_gpt5",
            "call_gpt54",
            "call_evidence_api",
            "batch_evidence_queries",
            "_get_evidence_api",
            "_call_llm_stream_once",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def _staging_system():
    return {
        "scheme": "临床严重度分级",
        "levels": [
            {"level": 1, "name": "轻度", "description": "轻症"},
            {"level": 2, "name": "中度", "description": "中症"},
            {"level": 3, "name": "重度", "description": "重症"},
        ],
    }


def _patient_row(case_id, diagnosis, acuity, stage, duration, symptom, symptom_time):
    row = utils._empty_row()
    row[utils.COL_CASE_ID] = case_id
    row[utils.COL_AGE] = "60"
    row[utils.COL_GENDER] = "男"
    row[utils.COL_DIAGNOSIS] = diagnosis
    row[utils.COL_STAGE] = stage
    row[utils.COL_PATIENT_LATERALITY] = "不适用"
    row[utils.COL_ACUITY] = acuity
    row[utils.COL_DURATION_TOTAL] = duration
    row[utils.COL_SYMPTOMS] = repr([(symptom, stage, "", "", "")])
    row[utils.COL_SIGNS] = repr([("肺部湿啰音", stage, "", "", "")])
    row[utils.COL_LAB_TESTS] = repr([("C反应蛋白升高", stage, "", "", "")])
    row[utils.COL_IMAGING] = repr([])
    row[utils.COL_FUNCTIONAL_TESTS] = repr([("6MWT缩短", stage, "", "", "")])
    row[utils.COL_ABSENT_SYMPTOMS] = repr([])
    row[utils.COL_ABSENT_SIGNS] = repr([])
    row[utils.COL_ABSENT_LAB_TESTS] = repr([])
    row[utils.COL_ABSENT_IMAGING] = repr([])
    row[utils.COL_ABSENT_FUNCTIONAL] = repr([])
    row[utils.COL_TIME_ORDER] = repr([
        (symptom, "症状", "起病", symptom_time),
        ("肺部湿啰音", "体征", "就诊", "D0"),
        ("C反应蛋白升高", "实验室检查", "就诊", "D0"),
        ("6MWT缩短", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_QUANTIFIED] = repr([
        ("C反应蛋白升高> 10 mg/L", "实验室检查", "就诊", "D0"),
        ("6MWT缩短{280,30,nan,400}m", "功能检查", "就诊", "D0"),
    ])
    row[utils.COL_SPECIFIC] = repr([
        ("C反应蛋白升高；实际值=50mg/L", "实验室检查", "就诊", "D0"),
        ("6MWT缩短；实际值=280m", "功能检查", "就诊", "D0"),
    ])
    return row


def _time_model(underlying_duration, current_episode_duration, symptom, symptom_time):
    return {
        "schema_version": 2,
        "underlying_duration": underlying_duration,
        "current_episode_duration": current_episode_duration,
        "symptom_timeline": [
            {
                "name": symptom,
                "category": "症状",
                "phase": "起病",
                "time_label": symptom_time,
            }
        ],
    }


def _functional_fact(ledger):
    matches = [fact for fact in ledger["facts"] if fact["domain"] == "functional"]
    assert len(matches) == 1
    return matches[0]


def test_fact_ledger_no_api_end_to_end_for_acute_and_exacerbation(tmp_path):
    cases = [
        (
            _patient_row("acute-001", "急性冠脉综合征", "急性", "中度", "6小时", "胸痛", "H-3"),
            _time_model(None, "6小时", "胸痛", "H-3"),
        ),
        (
            _patient_row("exacerb-001", "慢性心力衰竭", "慢性急性加重", "重度", "5天", "胸闷加重", "H-3"),
            _time_model("2年", "5天", "胸闷加重", "H-3"),
        ),
    ]

    routed = pfl.route_clinical_requests(
        "[申请查体]：心肺查体\n[申请化验]：C反应蛋白、胸部CT、6MWT"
    )
    assert routed == {
        "physical_exam": ["心肺查体"],
        "lab": ["C反应蛋白"],
        "imaging": ["胸部CT"],
        "functional": ["6MWT"],
    }

    for index, (row, time_model) in enumerate(cases, start=1):
        csv_path = tmp_path / f"case_{index}.csv"
        utils._save_csv(str(csv_path), [utils._empty_row(), row])

        ledger = pfl.build_fact_ledger(row, _staging_system(), time_model)
        ledger = pfl.converge_fact_ledger(ledger)
        assert ledger["status"] == "converged", ledger.get("audit")

        summary = pfl.write_fact_ledger(str(csv_path), 1, ledger)
        loaded = pfl.load_fact_ledger(
            str(csv_path), 1, expected_case_id=row[utils.COL_CASE_ID]
        )
        assert summary["ledger_hash"] == pfl.canonical_ledger_hash(loaded)
        assert summary["input_hash"] == pfl.ledger_input_hash(
            row, _staging_system(), time_model
        )
        assert loaded["provenance"]["source_row_hash"] == summary["input_hash"]
        assert loaded["audit"]["final_hash"] == summary["ledger_hash"]

        patient_view = pfl.project_patient_view(loaded)
        physical_view = pfl.project_physical_exam_view(loaded)
        crp_only_view = pfl.project_diagnostic_view(loaded, "[申请化验]：C反应蛋白")
        functional_view = pfl.project_diagnostic_view(loaded, "[申请化验]：6MWT")

        assert patient_view["course"]["underlying_duration"] == time_model["underlying_duration"]
        assert patient_view["course"]["current_episode_duration"] == time_model["current_episode_duration"]
        assert patient_view["symptoms"][0]["time"]["clinical_onset"] == time_model["symptom_timeline"][0]["time_label"]
        assert physical_view["facts"]

        objective_functional = _functional_fact(loaded)
        assert objective_functional["time"] == {
            "clinical_onset": None,
            "observed_at": "D0",
            "phase": "就诊",
        }
        assert objective_functional["feasibility"] == "clinically_questionable"

        assert crp_only_view["facts"]["lab"]
        assert crp_only_view["facts"]["imaging"] == []
        assert crp_only_view["facts"]["functional"] == []

        assert functional_view["facts"]["lab"] == []
        assert functional_view["facts"]["imaging"] == []
        assert functional_view["facts"]["functional"]
        assert "6MWT" in json.dumps(
            functional_view["facts"]["functional"], ensure_ascii=False
        )
