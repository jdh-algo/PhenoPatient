#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize an existing M2 run with the current local validators and no API calls."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import atlas_based_patient_generation as m2
import patient_fact_ledger as pfl
import run_m1_m2_batch as batch
import utils


def _forbid_api(*_args, **_kwargs):
    raise RuntimeError("offline M2 finalizer forbids model/API calls")


@contextmanager
def install_api_guard():
    """Temporarily fail closed if a future refactor reaches any model helper."""
    originals = []
    for module in (utils, m2):
        for name in ("call_gpt5", "call_gpt54"):
            if hasattr(module, name):
                originals.append((module, name, getattr(module, name)))
                setattr(module, name, _forbid_api)
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _absent_objective_by_category(row):
    return {
        "体征": m2.parse_list_from_response(row.get(utils.COL_ABSENT_SIGNS, "") or ""),
        "实验室检查": m2.parse_list_from_response(
            row.get(utils.COL_ABSENT_LAB_TESTS, "") or ""
        ),
        "影像检查": m2.parse_list_from_response(
            row.get(utils.COL_ABSENT_IMAGING, "") or ""
        ),
        "功能检查": m2.parse_list_from_response(
            row.get(utils.COL_ABSENT_FUNCTIONAL, "") or ""
        ),
    }


def _deterministic_specific_values(case_id, quantified_items, absent_by_category):
    seed_material = (
        f"{pfl.LEDGER_CODE_VERSION}\0{case_id}\0"
        f"{repr(quantified_items)}\0module_2_8"
    )
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    state = random.getstate()
    random.seed(seed)
    try:
        specific_items = m2._specific_values_with_shared_objective_values(
            quantified_items
        )
        specific_items = m2._harmonize_acid_base_specific_values(
            quantified_items, specific_items
        )
        m2._validate_objective_value_consistency(
            quantified_items, specific_items, absent_by_category
        )
        return specific_items
    finally:
        random.setstate(state)


_PRIOR_LEDGER_REUSE_BLOCKERS = {"anion_gap", "anion_gap_joint_projection"}
_DOMAIN_TO_CATEGORY = {
    "symptom": "症状",
    "sign": "体征",
    "lab": "实验室检查",
    "imaging": "影像检查",
    "functional": "功能检查",
}


def _nonconverged_error(updated, blockers):
    return ValueError(
        f"case_id={updated.get(utils.COL_CASE_ID)!r} M2.81 did not converge: "
        f"{json.dumps(blockers, ensure_ascii=False)}"
    )


def _prior_reuse_kinds_allowed(blockers):
    return bool(blockers) and all(
        blocker.get("kind") in _PRIOR_LEDGER_REUSE_BLOCKERS
        for blocker in blockers
    )


def _blocker_fact_ids(blockers):
    fact_ids = set()
    for blocker in blockers:
        fact_id = blocker.get("fact_id")
        if fact_id:
            fact_ids.add(fact_id)
        fact_ids.update(source for source in blocker.get("sources") or [] if source)
    return fact_ids


def _validated_anion_gap_reuse_fact_ids(ledger, blockers):
    if not _prior_reuse_kinds_allowed(blockers):
        return None
    fact_ids = _blocker_fact_ids(blockers)
    if len(fact_ids) != 4:
        return None
    records_by_fact_id = {
        record["fact"].get("fact_id"): record
        for record in pfl._metric_records(ledger)
        if record.get("fact", {}).get("fact_id")
    }
    records = []
    for fact_id in sorted(fact_ids):
        record = records_by_fact_id.get(fact_id)
        if record is None:
            return None
        records.append(record)
    if {record.get("metric") for record in records} != {
            "sodium", "chloride", "bicarbonate", "anion_gap"}:
        return None
    group_keys = {record.get("group_key") for record in records}
    if len(group_keys) != 1:
        return None
    for blocker in blockers:
        blocker_ids = set()
        if blocker.get("fact_id"):
            blocker_ids.add(blocker["fact_id"])
        blocker_ids.update(source for source in blocker.get("sources") or [] if source)
        if not blocker_ids or not blocker_ids <= fact_ids:
            return None
    return fact_ids


def _facts_by_id(ledger):
    return {
        fact.get("fact_id"): fact
        for fact in (ledger or {}).get("facts") or []
        if fact.get("fact_id")
    }


def _fact_signature(fact):
    concept = fact.get("concept") or {}
    return (
        concept.get("raw"),
        fact.get("domain"),
        json.dumps(fact.get("time") or {}, ensure_ascii=False, sort_keys=True),
    )


def _same_fact_identity(current_fact, prior_fact):
    return _fact_signature(current_fact) == _fact_signature(prior_fact)


def _unit_key_from_value(value):
    if not isinstance(value, dict):
        return ""
    reference = value.get("reference_range") or {}
    return pfl._unit_key(value.get("unit") or reference.get("unit"))


def _prior_value_fits_current_range(current_fact, prior_fact):
    current_value = current_fact.get("value") if isinstance(current_fact, dict) else None
    prior_value = prior_fact.get("value") if isinstance(prior_fact, dict) else None
    if not isinstance(current_value, dict) or not isinstance(prior_value, dict):
        return False
    number = prior_value.get("number")
    if number is None:
        return False
    try:
        number = float(number)
    except (TypeError, ValueError):
        return False
    if not number == number:
        return False
    if _unit_key_from_value(current_value) != _unit_key_from_value(prior_value):
        return False
    return pfl._value_within_range(current_fact, number)


def _format_prior_specific_value(current_fact, prior_fact):
    prior_value = prior_fact.get("value") or {}
    number = float(prior_value["number"])
    unit = (prior_value.get("unit") or (prior_value.get("reference_range") or {}).get("unit") or "").strip()
    if number.is_integer():
        number_text = str(int(number))
    else:
        number_text = f"{number:.12g}"
    return f"{number_text}{unit}"


def _specific_key_from_fact(fact):
    concept = fact.get("concept") or {}
    category = _DOMAIN_TO_CATEGORY.get(fact.get("domain"))
    observed_at = (fact.get("time") or {}).get("observed_at")
    if not concept.get("raw") or not category or not observed_at:
        return None
    return (concept.get("raw"), category, observed_at)


def _specific_key_from_item(item):
    if not isinstance(item, (list, tuple)) or len(item) < 4:
        return None
    raw = str(item[0] or "").split("；实际值=", 1)[0].strip()
    return (raw, str(item[1] or "").strip(), str(item[3] or "").strip())


def _specific_items_with_prior_values(updated, current_ledger, prior_ledger, fact_ids):
    current_facts = _facts_by_id(current_ledger)
    prior_facts = _facts_by_id(prior_ledger)
    replacements = {}
    for fact_id in sorted(fact_ids):
        current_fact = current_facts.get(fact_id)
        prior_fact = prior_facts.get(fact_id)
        if not current_fact or not prior_fact:
            raise ValueError(f"prior ledger missing fact_id={fact_id}")
        if not _same_fact_identity(current_fact, prior_fact):
            raise ValueError(f"prior ledger fact identity mismatch for fact_id={fact_id}")
        if not _prior_value_fits_current_range(current_fact, prior_fact):
            raise ValueError(f"prior ledger value out of current M2.7 range for fact_id={fact_id}")
        key = _specific_key_from_fact(current_fact)
        if key is None or key in replacements:
            raise ValueError(f"cannot map fact_id={fact_id} back to one specific timeline item")
        replacements[key] = _format_prior_specific_value(current_fact, prior_fact)

    specific_items = m2.parse_list_from_response(updated.get(utils.COL_SPECIFIC, "") or "")
    replaced = set()
    rewritten = []
    for item in specific_items:
        normalized = tuple(str(value).strip() for value in item)
        key = _specific_key_from_item(normalized)
        if key in replacements:
            normalized = (
                f"{key[0]}；实际值={replacements[key]}",
                *normalized[1:],
            )
            replaced.add(key)
        rewritten.append(normalized)
    if replaced != set(replacements):
        missing = sorted(set(replacements) - replaced)
        raise ValueError(f"could not backfill prior values into COL_SPECIFIC: {missing}")
    return rewritten, len(replaced)


def _retry_with_prior_ledger_values(updated, ledger, prior_converged_ledger,
                                    quantified_items, absent_by_category,
                                    staging_system):
    blockers = ledger.get("audit", {}).get("blockers") or []
    if prior_converged_ledger is None:
        return None
    fact_ids = _validated_anion_gap_reuse_fact_ids(ledger, blockers)
    if prior_converged_ledger.get("status") != "converged" or fact_ids is None:
        return None
    try:
        specific_items, reused_count = _specific_items_with_prior_values(
            updated, ledger, prior_converged_ledger, fact_ids
        )
        validated_specific = m2._validated_existing_specific_timeline(
            quantified_items, repr(specific_items), absent_by_category
        )
        if validated_specific is None:
            return None
        retry_row = copy.deepcopy(updated)
        retry_row[utils.COL_SPECIFIC] = repr(validated_specific)
        retry_row, retry_ledger = pfl.module_2_81_finalize_fact_ledger(
            retry_row, staging_system, max_rounds=5
        )
        if retry_ledger.get("status") != "converged" or reused_count != 4:
            return None
        return retry_row, retry_ledger, reused_count
    except (TypeError, ValueError, KeyError):
        return None


def _selected_symptom_items(row):
    return [
        item for item in m2.parse_list_from_response(row.get(utils.COL_SYMPTOMS, "") or "")
        if isinstance(item, (list, tuple)) and item and str(item[0]).strip()
    ]


def _symptom_names(symptom_items):
    names = []
    for item in symptom_items:
        name = str(item[0]).strip()
        if name and name not in names:
            names.append(name)
    return names


def _chief_match_key(text):
    return "".join(
        char for char in str(text or "").strip().strip('。，、.， "\'')
        if not char.isspace()
    )


def _unsafe_chief_text(text):
    folded = str(text or "").strip()
    return any(token in folded for token in ("否认", "无", "未诉", "未见", "没有"))


def _safe_m26_suggested_chief(row, symptom_names):
    text = str(row.get(utils.COL_M26_OUTPUT, "") or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    suggested = str(
        ((payload.get("current_chief") or {}).get("symptom") if isinstance(payload, dict) else "")
        or ""
    ).strip().strip('。，、.， "\'')
    suggested_key = _chief_match_key(suggested)
    if not suggested_key or suggested_key in {"无", "无症状", "无明显症状"}:
        return ""
    for name in symptom_names:
        if _chief_match_key(name) == suggested_key:
            return name
    if _unsafe_chief_text(suggested):
        return ""
    return ""


def _time_label_sort_key(label):
    text = str(label or "").strip().upper()
    if not text:
        return (99, float("inf"))
    unit_order = {"H": 0, "D": 1, "W": 2, "M": 3, "Y": 4}
    unit = text[:1]
    try:
        magnitude = abs(float(text[1:]))
    except ValueError:
        magnitude = float("inf")
    return (unit_order.get(unit, 98), magnitude)


def _deterministic_chief_suggestion(symptom_names, time_ordered_items):
    if not symptom_names:
        return ""
    symptom_set = set(symptom_names)
    candidates = []
    for order, item in enumerate(time_ordered_items or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        name = str(item[0]).strip()
        category = str(item[1]).strip()
        if category == "症状" and name in symptom_set:
            label = item[-1] if len(item) >= 4 else ""
            candidates.append((_time_label_sort_key(label), order, name))
    if candidates:
        return min(candidates)[2]
    return symptom_names[0]


def _fill_missing_chief_complaint_after_ledger(updated):
    if str(updated.get(utils.COL_CHIEF_COMPLAINT, "") or "").strip():
        return False
    symptom_items = _selected_symptom_items(updated)
    symptom_names = _symptom_names(symptom_items)
    time_ordered_items = m2.parse_list_from_response(updated.get(utils.COL_TIME_ORDER, "") or "")
    suggested = _safe_m26_suggested_chief(updated, symptom_names)
    if not suggested:
        suggested = _deterministic_chief_suggestion(symptom_names, time_ordered_items)
    chief = m2.module_2_9_derive_chief_complaint(
        updated.get(utils.COL_TIME_ORDER, "") or "",
        seed_text="",
        tag="offline-finalizer-2.9",
        symptoms_list=symptom_items,
        diagnosis=updated.get(utils.COL_DIAGNOSIS, "") or "",
        patient_stage=updated.get(utils.COL_STAGE, "") or "",
        suggested=suggested,
        time_ordered_items=time_ordered_items,
    )
    updated[utils.COL_CHIEF_COMPLAINT] = chief or ""
    return bool(chief)


def finalize_patient_row_noapi(row, staging_system, prior_converged_ledger=None):
    """Revalidate M2.7/M2.8, rebuild M2.81, then fill missing M2.9 chief complaint locally."""
    updated = copy.deepcopy(row)
    final_time_order = m2.parse_list_from_response(
        updated.get(utils.COL_TIME_ORDER, "") or ""
    )
    original_quantified = m2._validate_quantified_timeline(
        final_time_order,
        m2.parse_list_from_response(updated.get(utils.COL_QUANTIFIED, "") or ""),
    )
    absent_by_category = _absent_objective_by_category(updated)
    quantified_items = m2._harmonize_objective_metric_ranges(
        original_quantified, absent_by_category
    )
    updated[utils.COL_QUANTIFIED] = repr(quantified_items)

    existing_specific = m2._validated_existing_specific_timeline(
        quantified_items,
        updated.get(utils.COL_SPECIFIC, "") or "",
        absent_by_category,
    )
    regenerated = existing_specific is None
    if regenerated:
        specific_items = _deterministic_specific_values(
            str(updated.get(utils.COL_CASE_ID, "") or ""),
            quantified_items,
            absent_by_category,
        )
    else:
        specific_items = existing_specific
    updated[utils.COL_SPECIFIC] = repr(specific_items)

    updated, ledger = pfl.module_2_81_finalize_fact_ledger(
        updated, staging_system, max_rounds=5
    )
    prior_ledger_values_reused = 0
    chief_complaint_filled = False
    if ledger.get("status") != "converged":
        retry = _retry_with_prior_ledger_values(
            updated, ledger, prior_converged_ledger, quantified_items,
            absent_by_category, staging_system
        )
        if retry is not None:
            updated, ledger, prior_ledger_values_reused = retry
        else:
            blockers = ledger.get("audit", {}).get("blockers") or []
            raise _nonconverged_error(updated, blockers)
    chief_complaint_filled = _fill_missing_chief_complaint_after_ledger(updated)
    stats = {
        "quantified_changed": quantified_items != original_quantified,
        "specific_regenerated": regenerated,
        "prior_ledger_values_reused": prior_ledger_values_reused,
        "chief_complaint_filled": chief_complaint_filled,
    }
    return updated, ledger, stats


@contextmanager
def exclusive_run_lock(run_root):
    lock_path = Path(run_root) / ".resume.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"run is still active; could not acquire {lock_path}"
            ) from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _load_prior_converged_ledger(csv_path, row_index, case_id):
    row_payload_path = Path(utils._module_io_abs(str(csv_path), row_index))
    if not row_payload_path.exists():
        return None
    row_payload = utils._read_json_file(str(row_payload_path), default=None)
    if not isinstance(row_payload, dict):
        raise ValueError(f"malformed prior fact ledger sidecar: {row_payload_path}")
    summary = row_payload.get(pfl.COL_M281_OUTPUT)
    if not isinstance(summary, dict) or summary.get("status") != "converged":
        return None
    return pfl.load_fact_ledger(str(csv_path), row_index, expected_case_id=case_id)


def _prepare_run(run_root, expected_cases, expected_groups, patients_per_group):
    m1_dir = Path(run_root) / "M1"
    m2_dir = Path(run_root) / "M2"
    csv_files = sorted(m2_dir.rglob("*.csv"))
    if len(csv_files) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} M2 CSV files, found {len(csv_files)}"
        )
    if len(list(m1_dir.rglob("*.csv"))) != expected_groups:
        raise ValueError(f"expected {expected_groups} M1 CSV files")

    plans = []
    case_ids = []
    aggregate = {
        "csv_files": len(csv_files),
        "patient_rows": 0,
        "quantified_changed": 0,
        "specific_regenerated": 0,
        "prior_ledger_values_reused": 0,
        "chief_complaint_filled": 0,
        "chief_complaint_filled_case_ids": [],
    }
    for csv_path in csv_files:
        row_1, patients = utils._load_existing_csv(str(csv_path))
        if row_1 is None:
            raise ValueError(f"missing M1 row: {csv_path}")
        expected_indices = set(range(1, patients_per_group + 1))
        if set(patients) != expected_indices:
            raise ValueError(
                f"{csv_path}: expected patient rows {sorted(expected_indices)}, "
                f"found {sorted(patients)}"
            )
        staging_system = m2._validate_m1_row_for_m2(row_1)
        finalized_rows = []
        for row_index in sorted(patients):
            source_row = patients[row_index]
            case_id = str(source_row.get(utils.COL_CASE_ID, "") or "").strip()
            if not case_id:
                raise ValueError(f"{csv_path} row {row_index}: missing case_id")
            prior_ledger = _load_prior_converged_ledger(csv_path, row_index, case_id)
            if prior_ledger is None:
                finalized, ledger, stats = finalize_patient_row_noapi(
                    source_row, staging_system
                )
            else:
                finalized, ledger, stats = finalize_patient_row_noapi(
                    source_row, staging_system, prior_converged_ledger=prior_ledger
                )
            case_ids.append(case_id)
            aggregate["patient_rows"] += 1
            aggregate["quantified_changed"] += int(stats["quantified_changed"])
            aggregate["specific_regenerated"] += int(stats["specific_regenerated"])
            aggregate["prior_ledger_values_reused"] += int(stats.get("prior_ledger_values_reused", 0))
            if stats.get("chief_complaint_filled", False):
                aggregate["chief_complaint_filled"] += 1
                aggregate["chief_complaint_filled_case_ids"].append(case_id)
            finalized_rows.append(finalized)
        plans.append((csv_path, row_1, finalized_rows))

    if aggregate["patient_rows"] != expected_cases:
        raise ValueError(
            f"expected {expected_cases} patient rows, found {aggregate['patient_rows']}"
        )
    if len(set(case_ids)) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} unique case_id values, found {len(set(case_ids))}"
        )
    return plans, aggregate


def _commit_plans(plans):
    for csv_path, row_1, patients in plans:
        m2._save_csv_with_fact_ledgers(
            str(csv_path), [row_1, *patients]
        )


def _finalize_status(run_root, aggregate, expected_cases, expected_groups,
                     patients_per_group, transaction_id, expected_chief_fills):
    status_path = Path(run_root) / "status.json"
    status = utils._read_json_file(str(status_path), default={}) or {}
    counts = batch._count_complete_m2(
        str(Path(run_root) / "M2"), patients_per_group
    )
    required = {
        "m2_csv_files": expected_groups,
        "m2_patient_rows": expected_cases,
        "m2_completed": expected_cases,
        "m2_unique_case_ids": expected_cases,
        "m281_completed": expected_cases,
        "m281_failed": 0,
        "m281_quarantined": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": counts.get(key)}
        for key, expected in required.items()
        if counts.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "post-commit validation failed: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    previous_result = status.get("m2_result")
    if previous_result and "pre_offline_finalization_m2_result" not in status:
        status["pre_offline_finalization_m2_result"] = previous_result
    status.update(counts)
    status.update({
        "status": "completed",
        "phase": "completed",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "m2_result": {
            "status": "success",
            "mode": "offline_existing_m2_finalization",
            "no_api": True,
            "ledger_code_version": pfl.LEDGER_CODE_VERSION,
        },
        "offline_finalization": {
            **aggregate,
            "no_api": True,
            "ledger_code_version": pfl.LEDGER_CODE_VERSION,
            "transaction_id": transaction_id,
            "expected_chief_fills": expected_chief_fills,
            "model_sidecar_boundary": _MODEL_SIDECAR_BOUNDARY,
            "runner_detection_boundary": _RUNNER_DETECTION_BOUNDARY,
        },
    })
    utils._atomic_write_json_file(str(status_path), status)
    return status


_FINALIZER_PREFIX = ".offline_finalizer"
_JOURNAL_NAME = f"{_FINALIZER_PREFIX}_transaction.json"
_MODEL_SIDECAR_BOUNDARY = {
    "m27_m28_raw_model_sidecars": "preserve_existing_no_fabrication",
    "finalized_fields": "M2.7/M2.8 values are revalidated; M2.81 ledger is rebuilt",
}
_RUNNER_DETECTION_BOUNDARY = {
    "gates": ["run_resume.pid live-process check", ".resume.lock flock for commit"],
    "undetectable": "future runners that omit both gates cannot be identified by this finalizer",
}


def _remove_path(path):
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _cleanup_offline_artifacts(run_root, keep=()):
    keep = {Path(item).resolve() for item in keep}
    for path in Path(run_root).iterdir():
        if not path.name.startswith(_FINALIZER_PREFIX):
            continue
        if path.resolve() in keep:
            continue
        _remove_path(path)


def _fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_durable(path, payload):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _write_journal(run_root, payload):
    journal_path = Path(run_root) / _JOURNAL_NAME
    _write_json_durable(journal_path, payload)
    return journal_path


def _restore_status(run_root, status_backup, status_existed):
    status_path = Path(run_root) / "status.json"
    status_backup = Path(status_backup) if status_backup else None
    if status_existed and status_backup and status_backup.exists():
        os.replace(status_backup, status_path)
        _fsync_dir(Path(run_root))
    elif not status_existed:
        status_path.unlink(missing_ok=True)
        if status_backup and status_backup.exists():
            status_backup.unlink()
        _fsync_dir(Path(run_root))


def _failed_new_m2_path(run_root, journal_payload):
    txn_id = str(journal_payload.get("txn_id") or f"unknown.{os.getpid()}")
    return Path(run_root) / f"{_FINALIZER_PREFIX}_failed_new_m2.{txn_id}"


def _rollback_from_journal(run_root, journal_payload):
    backup_value = journal_payload.get("backup_m2")
    backup_m2 = (
        Path(backup_value) if isinstance(backup_value, str) and backup_value.strip()
        else None
    )
    live_m2 = Path(run_root) / "M2"
    failed_new_m2 = _failed_new_m2_path(run_root, journal_payload)
    moved_live_to_failed = False
    if backup_m2 is not None and backup_m2.name and backup_m2.exists():
        _remove_path(failed_new_m2)
        if live_m2.exists():
            os.replace(live_m2, failed_new_m2)
            _fsync_dir(Path(run_root))
            moved_live_to_failed = True
        try:
            os.replace(backup_m2, live_m2)
            _fsync_dir(Path(run_root))
        except Exception:
            if moved_live_to_failed and failed_new_m2.exists() and not live_m2.exists():
                os.replace(failed_new_m2, live_m2)
                _fsync_dir(Path(run_root))
            raise
        if failed_new_m2.exists():
            _remove_path(failed_new_m2)
    _restore_status(
        run_root,
        journal_payload.get("status_backup"),
        bool(journal_payload.get("status_existed")),
    )


def _status_proves_committed(run_root, journal_payload):
    live_m2 = Path(run_root) / "M2"
    status_path = Path(run_root) / "status.json"
    status = utils._read_json_file(str(status_path), default=None)
    expected_txn_id = str(journal_payload.get("txn_id") or "")
    offline = status.get("offline_finalization", {}) if isinstance(status, dict) else {}
    return (
        live_m2.is_dir()
        and bool(expected_txn_id)
        and isinstance(status, dict)
        and status.get("status") == "completed"
        and status.get("phase") == "completed"
        and status.get("m2_result", {}).get("mode") == "offline_existing_m2_finalization"
        and offline.get("no_api") is True
        and offline.get("transaction_id") == expected_txn_id
    )


def _recover_committed_transaction(run_root, journal_payload):
    if not _status_proves_committed(run_root, journal_payload):
        raise RuntimeError(
            "status_committed transaction does not verify live M2 plus offline status; "
            "leaving recovery materials in place"
        )


def _recover_interrupted_transaction(run_root):
    journal_path = Path(run_root) / _JOURNAL_NAME
    if journal_path.exists():
        payload = utils._read_json_file(str(journal_path), default={}) or {}
        phase = payload.get("phase")
        if phase == "status_committed" or (phase == "new_installed" and _status_proves_committed(run_root, payload)):
            _recover_committed_transaction(run_root, payload)
        else:
            _rollback_from_journal(run_root, payload)
        journal_path.unlink(missing_ok=True)
    _cleanup_offline_artifacts(run_root)


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _reject_live_resume_pid(run_root):
    pid_path = Path(run_root) / "run_resume.pid"
    if not pid_path.exists():
        return
    text = pid_path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        pid = int(text)
    except ValueError:
        return
    if pid > 0 and _pid_is_running(pid):
        raise RuntimeError(
            f"run_resume.pid points to live process {pid}; refusing offline M2 finalization"
        )


def _copy_run_to_staging(run_root, *, inside_run_root):
    txn_id = f"{int(time.time() * 1000000)}.{os.getpid()}"
    if inside_run_root:
        staging_root = Path(run_root) / f"{_FINALIZER_PREFIX}_staging.{txn_id}"
    else:
        staging_root = Path(run_root).parent / f".{Path(run_root).name}{_FINALIZER_PREFIX}_staging.{txn_id}"
    try:
        staging_root.mkdir()
        shutil.copytree(Path(run_root) / "M1", staging_root / "M1")
        shutil.copytree(Path(run_root) / "M2", staging_root / "M2")
    except Exception:
        _remove_path(staging_root)
        raise
    return staging_root, txn_id


def _clear_fact_ledger_manifests(m2_dir):
    for manifest_path in Path(m2_dir).rglob(pfl.MANIFEST_FILENAME):
        manifest_path.unlink(missing_ok=True)
        Path(f"{manifest_path}.lock").unlink(missing_ok=True)
    for ledger_path in Path(m2_dir).rglob("row_*.fact_ledger.json"):
        ledger_path.unlink(missing_ok=True)


def _read_manifest_records(manifest_path):
    records = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _validate_manifest_for_csv(csv_path, patients_per_group):
    csv_path = Path(csv_path)
    _row_1, patients = utils._load_existing_csv(str(csv_path))
    expected_indices = list(range(1, patients_per_group + 1))
    if sorted(patients) != expected_indices:
        raise RuntimeError(
            f"{csv_path}: staging validation failed: expected row_index {expected_indices}, "
            f"found {sorted(patients)}"
        )
    manifest_path = Path(utils._module_io_dir(str(csv_path))) / pfl.MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(f"{csv_path}: staging validation failed: missing fact ledger manifest")
    records = _read_manifest_records(manifest_path)
    actual_indices = [record.get("row_index") for record in records]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"{csv_path}: staging validation failed: manifest row_index sequence "
            f"{actual_indices} != {expected_indices}"
        )
    seen_case_ids = set()
    referenced_ledger_files = set()
    for record in records:
        row_index = record["row_index"]
        row = patients[row_index]
        case_id = str(row.get(utils.COL_CASE_ID, "") or "").strip()
        if not case_id:
            raise RuntimeError(f"{csv_path}: staging validation failed: missing case_id at row {row_index}")
        if record.get("case_id") != case_id:
            raise RuntimeError(
                f"{csv_path}: staging validation failed: manifest case_id mismatch at row {row_index}"
            )
        if case_id in seen_case_ids:
            raise RuntimeError(f"{csv_path}: staging validation failed: duplicate case_id {case_id}")
        seen_case_ids.add(case_id)
        row_payload = utils._read_json_file(utils._module_io_abs(str(csv_path), row_index), default=None)
        if not isinstance(row_payload, dict):
            raise RuntimeError(f"{csv_path}: staging validation failed: missing sidecar row {row_index}")
        summary = row_payload.get(pfl.COL_M281_OUTPUT)
        if not isinstance(summary, dict):
            raise RuntimeError(f"{csv_path}: staging validation failed: missing sidecar summary row {row_index}")
        for key in ("path", "input_hash", "ledger_hash", "status", "round_count"):
            if record.get(key) != summary.get(key):
                raise RuntimeError(
                    f"{csv_path}: staging validation failed: manifest/sidecar {key} mismatch row {row_index}"
                )
        if summary.get("status") != "converged":
            raise RuntimeError(f"{csv_path}: staging validation failed: non-converged ledger row {row_index}")
        ledger_path = Path(utils._module_io_dir(str(csv_path))) / str(summary.get("path") or "")
        referenced_ledger_files.add(ledger_path.name)
        ledger = utils._read_json_file(str(ledger_path), default=None)
        if not isinstance(ledger, dict):
            raise RuntimeError(f"{csv_path}: staging validation failed: missing ledger row {row_index}")
        if ledger.get("case_id") != case_id:
            raise RuntimeError(f"{csv_path}: staging validation failed: ledger case_id mismatch row {row_index}")
        input_hash = ledger.get("provenance", {}).get("source_row_hash")
        if input_hash != summary.get("input_hash"):
            raise RuntimeError(f"{csv_path}: staging validation failed: ledger input_hash mismatch row {row_index}")
        ledger_hash = pfl.canonical_ledger_hash(ledger)
        if ledger_hash != summary.get("ledger_hash") or ledger.get("audit", {}).get("final_hash") != ledger_hash:
            raise RuntimeError(f"{csv_path}: staging validation failed: ledger hash mismatch row {row_index}")
    actual_ledger_files = {
        path.name for path in Path(utils._module_io_dir(str(csv_path))).glob("row_*.fact_ledger.json")
    }
    if actual_ledger_files != referenced_ledger_files:
        raise RuntimeError(
            f"{csv_path}: staging validation failed: unreferenced fact ledger files "
            f"{sorted(actual_ledger_files - referenced_ledger_files)}"
        )


def _default_expected_chief_fills(expected_cases):
    return 8 if int(expected_cases) == 1000 else 0


def _resolve_expected_chief_fills(expected_cases, expected_chief_fills):
    if expected_chief_fills is None:
        return _default_expected_chief_fills(expected_cases)
    return int(expected_chief_fills)


def _assert_expected_chief_fills(aggregate, expected_chief_fills):
    actual = int(aggregate.get("chief_complaint_filled", 0))
    if actual != int(expected_chief_fills):
        raise RuntimeError(
            "chief_complaint_filled count mismatch: "
            + json.dumps({"expected": int(expected_chief_fills), "actual": actual}, ensure_ascii=False)
        )


def _validate_staged_m2(staging_root, expected_cases, expected_groups, patients_per_group):
    counts = batch._count_complete_m2(str(Path(staging_root) / "M2"), patients_per_group)
    required = {
        "m2_csv_files": expected_groups,
        "m2_patient_rows": expected_cases,
        "m2_completed": expected_cases,
        "m2_unique_case_ids": expected_cases,
        "m281_completed": expected_cases,
        "m281_failed": 0,
        "m281_quarantined": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": counts.get(key)}
        for key, expected in required.items()
        if counts.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "staging validation failed: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    csv_files = sorted((Path(staging_root) / "M2").rglob("*.csv"))
    if len(csv_files) != expected_groups:
        raise RuntimeError(
            f"staging validation failed: expected {expected_groups} CSVs, found {len(csv_files)}"
        )
    for csv_path in csv_files:
        _validate_manifest_for_csv(csv_path, patients_per_group)
    return counts


def _transaction_paths(run_root, staging_root, txn_id):
    return {
        "txn_id": txn_id,
        "journal": Path(run_root) / _JOURNAL_NAME,
        "backup_m2": Path(run_root) / f"{_FINALIZER_PREFIX}_backup_m2.{txn_id}",
        "status_backup": Path(run_root) / f"{_FINALIZER_PREFIX}_status_backup.{txn_id}.json",
        "staging_root": Path(staging_root),
    }


def _journal_payload(paths, phase, status_existed):
    return {
        "phase": phase,
        "txn_id": paths["txn_id"],
        "staging_root": str(paths["staging_root"]),
        "backup_m2": str(paths["backup_m2"]),
        "status_backup": str(paths["status_backup"]),
        "status_existed": status_existed,
    }


def _backup_status(run_root, status_backup):
    status_path = Path(run_root) / "status.json"
    status_existed = status_path.exists()
    if status_existed:
        shutil.copy2(status_path, status_backup)
        _fsync_file(status_backup)
        _fsync_dir(Path(run_root))
    return status_existed


def _install_staged_m2(run_root, staging_root, paths, status_existed):
    _write_journal(run_root, _journal_payload(paths, "prepared", status_existed))
    os.replace(Path(run_root) / "M2", paths["backup_m2"])
    _fsync_dir(Path(run_root))
    _write_journal(run_root, _journal_payload(paths, "original_moved", status_existed))
    os.replace(Path(staging_root) / "M2", Path(run_root) / "M2")
    _fsync_dir(Path(run_root))
    _write_journal(run_root, _journal_payload(paths, "new_installed", status_existed))
    return paths["journal"], paths["backup_m2"], paths["status_backup"]


def _finalizer_report(commit, run_root, aggregate, validation_counts=None):
    report = {
        "mode": "commit" if commit else "dry-run",
        "run_root": str(run_root),
        "ledger_code_version": pfl.LEDGER_CODE_VERSION,
        "model_sidecar_boundary": _MODEL_SIDECAR_BOUNDARY,
        "runner_detection_boundary": _RUNNER_DETECTION_BOUNDARY,
        **aggregate,
    }
    if validation_counts is not None:
        report["staging_verification"] = validation_counts
    return report


def _finalize_run_inner(run_root, *, commit, expected_cases, expected_groups,
                        patients_per_group, expected_chief_fills):
    if commit:
        _reject_live_resume_pid(run_root)
        _recover_interrupted_transaction(run_root)
    staging_root = None
    paths = None
    journal = None
    backup_m2 = None
    status_backup = None
    preserve_recovery = False
    try:
        staging_root, txn_id = _copy_run_to_staging(run_root, inside_run_root=commit)
        plans, aggregate = _prepare_run(
            staging_root, expected_cases, expected_groups, patients_per_group
        )
        aggregate["expected_chief_fills"] = expected_chief_fills
        _assert_expected_chief_fills(aggregate, expected_chief_fills)
        _clear_fact_ledger_manifests(staging_root / "M2")
        _commit_plans(plans)
        validation_counts = _validate_staged_m2(
            staging_root, expected_cases, expected_groups, patients_per_group
        )
        report = _finalizer_report(commit, run_root, aggregate, validation_counts)
        if not commit:
            return report
        paths = _transaction_paths(run_root, staging_root, txn_id)
        status_existed = _backup_status(run_root, paths["status_backup"])
        journal = paths["journal"]
        backup_m2 = paths["backup_m2"]
        status_backup = paths["status_backup"]
        try:
            _install_staged_m2(run_root, staging_root, paths, status_existed)
            status = _finalize_status(
                run_root, aggregate, expected_cases, expected_groups,
                patients_per_group, paths["txn_id"], expected_chief_fills
            )
            _write_journal(run_root, _journal_payload(paths, "status_committed", status_existed))
        except Exception:
            payload = None
            if journal is not None and Path(journal).exists():
                payload = utils._read_json_file(str(journal), default={}) or None
            if payload:
                try:
                    _rollback_from_journal(run_root, payload)
                except Exception:
                    preserve_recovery = True
                    raise
            raise
        report["verification"] = {
            key: status.get(key)
            for key in (
                "status", "phase", "m2_csv_files", "m2_patient_rows",
                "m2_completed", "m2_unique_case_ids", "m281_completed",
                "m281_failed", "m281_quarantined",
            )
        }
        return report
    finally:
        if staging_root is not None:
            _remove_path(staging_root)
        if not preserve_recovery:
            if journal is not None:
                Path(journal).unlink(missing_ok=True)
            if backup_m2 is not None:
                _remove_path(backup_m2)
            if status_backup is not None:
                Path(status_backup).unlink(missing_ok=True)
            if commit:
                _cleanup_offline_artifacts(run_root)


def finalize_run(run_root, *, commit=False, expected_cases=1000,
                 expected_groups=50, patients_per_group=20,
                 expected_chief_fills=None):
    with install_api_guard():
        expected_chief_fills = _resolve_expected_chief_fills(
            expected_cases, expected_chief_fills
        )
        run_root = Path(run_root).resolve()
        if not run_root.is_dir():
            raise ValueError(f"run root does not exist: {run_root}")
        _reject_live_resume_pid(run_root)
        if not commit:
            return _finalize_run_inner(
                run_root,
                commit=False,
                expected_cases=expected_cases,
                expected_groups=expected_groups,
                patients_per_group=patients_per_group,
                expected_chief_fills=expected_chief_fills,
            )
        with exclusive_run_lock(run_root):
            _reject_live_resume_pid(run_root)
            return _finalize_run_inner(
                run_root,
                commit=True,
                expected_cases=expected_cases,
                expected_groups=expected_groups,
                patients_per_group=patients_per_group,
                expected_chief_fills=expected_chief_fills,
            )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Rebuild existing M2.7/M2.8/M2.81 outputs without API calls."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--expected-cases", type=int, default=1000)
    parser.add_argument("--expected-groups", type=int, default=50)
    parser.add_argument("--patients-per-group", type=int, default=20)
    parser.add_argument("--expected-chief-fills", type=int, default=None)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    report = finalize_run(
        args.run_root,
        commit=args.commit,
        expected_cases=args.expected_cases,
        expected_groups=args.expected_groups,
        patients_per_group=args.patients_per_group,
        expected_chief_fills=args.expected_chief_fills,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
