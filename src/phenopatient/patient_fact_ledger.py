# -*- coding: utf-8 -*-
"""Patient fact ledger compilation and sidecar persistence for module 2.81."""

import copy
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import threading

import utils

COL_M281_OUTPUT = '模块2.81_输出'
PENDING_LEDGER_ROW_KEY = '_m281_fact_ledger'
LEDGER_SCHEMA_VERSION = 'fact_ledger.v1'
LEDGER_MODULE = '2.81'
LEDGER_CODE_VERSION = 'm2.81.validator.v9'
MANIFEST_FILENAME = 'fact_ledger_manifest.jsonl'
M3_LEDGER_METADATA_KEY = '模块3_账本元数据'

_NONBLOOD_SPECIMEN_TERMS = (
    '尿', '尿液', '痰', '脑脊液', 'CSF',
    '胸水', '胸腔积液', '胸腔液', '胸腔穿刺液',
    '腹水', '腹腔积液', '腹腔穿刺液', '穿刺液', '腹腔液',
    '关节液',
)

_NUMERIC_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)')
_MANIFEST_LOCKS = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()

_POSITIVE_COLUMNS = (
    (utils.COL_SYMPTOMS, '症状', 'symptom'),
    (utils.COL_SIGNS, '体征', 'sign'),
    (utils.COL_LAB_TESTS, '实验室检查', 'lab'),
    (utils.COL_IMAGING, '影像检查', 'imaging'),
    (utils.COL_FUNCTIONAL_TESTS, '功能检查', 'functional'),
)
_NEGATIVE_COLUMNS = (
    (utils.COL_ABSENT_SYMPTOMS, '症状', 'symptom'),
    (utils.COL_ABSENT_SIGNS, '体征', 'sign'),
    (utils.COL_ABSENT_LAB_TESTS, '实验室检查', 'lab'),
    (utils.COL_ABSENT_IMAGING, '影像检查', 'imaging'),
    (utils.COL_ABSENT_FUNCTIONAL, '功能检查', 'functional'),
)


def ledger_input_hash(row: dict, staging_system: dict, time_model: dict) -> str:
    payload = {
        'ledger_code_version': LEDGER_CODE_VERSION,
        'row': _selected_input_fields(row),
        'staging_system': _canonical_jsonable(staging_system or {}),
        'time_model': _canonical_jsonable(_validate_time_model(time_model)),
        'normalized_values': _canonical_jsonable(_build_value_index(row)),
    }
    return _sha256_json(payload)


def canonical_ledger_hash(ledger: dict) -> str:
    payload = _canonical_ledger_payload(ledger)
    return _sha256_json(payload)


def build_fact_ledger(row: dict, staging_system: dict, time_model: dict) -> dict:
    normalized_time_model = _validate_time_model(time_model)
    source_hash = ledger_input_hash(row, staging_system, normalized_time_model)
    time_index = _build_time_index(row, normalized_time_model)
    value_index = _build_value_index(row)
    facts = []
    identity_facts = {}

    for column, category, domain in _POSITIVE_COLUMNS:
        for source_index, raw_item in enumerate(_parse_items(row.get(column, ''))):
            raw_name = _phenotype_name(raw_item)
            if not raw_name:
                continue
            fact = _build_fact(
                raw_name=raw_name,
                category=category,
                domain=domain,
                clinical_state='present',
                source_column=column,
                source_index=source_index,
                source_level=_phenotype_source_level(raw_item),
                time_index=time_index,
                value_index=value_index,
            )
            _add_fact(facts, identity_facts, fact)

    for column, category, domain in _NEGATIVE_COLUMNS:
        for source_index, raw_item in enumerate(_parse_items(row.get(column, ''))):
            raw_name = _phenotype_name(raw_item)
            if not raw_name:
                continue
            fact = _build_fact(
                raw_name=raw_name,
                category=category,
                domain=domain,
                clinical_state='absent',
                source_column=column,
                source_index=source_index,
                source_level=_phenotype_source_level(raw_item),
                time_index=time_index,
                value_index={},
            )
            _add_fact(facts, identity_facts, fact)

    facts.sort(key=lambda item: item['fact_id'])
    ledger = {
        'schema_version': LEDGER_SCHEMA_VERSION,
        'module': LEDGER_MODULE,
        'status': 'converged',
        'case_id': _clean(row.get(utils.COL_CASE_ID, '')),
        'patient': _build_patient(row, staging_system),
        'course': {
            'underlying_duration': normalized_time_model.get('underlying_duration'),
            'current_episode_duration': normalized_time_model.get('current_episode_duration'),
            'anchor': 'current_visit',
        },
        'patient_context': _build_patient_context(row),
        'facts': facts,
        'relations': [],
        'audit': {'rounds': [], 'final_hash': None},
        'provenance': {'source_row_hash': source_hash, 'code_version': LEDGER_CODE_VERSION},
    }
    ledger['audit']['final_hash'] = canonical_ledger_hash(ledger)
    return ledger


def module_2_81_finalize_fact_ledger(row: dict, staging_system: dict,
                                     max_rounds: int = 5) -> tuple[dict, dict]:
    updated_row = dict(row or {})
    try:
        time_model = _time_model_from_row(updated_row)
        ledger = build_fact_ledger(updated_row, staging_system, time_model)
        ledger.setdefault('provenance', {})['code_version'] = LEDGER_CODE_VERSION
        ledger['provenance']['severity_levels'] = _severity_level_names(staging_system)
        ledger = converge_fact_ledger(ledger, max_rounds=max_rounds)
    except Exception as exc:
        ledger = _quarantined_shell_ledger(updated_row, staging_system, exc)
    updated_row[COL_M281_OUTPUT] = _ledger_row_summary(ledger)
    updated_row[PENDING_LEDGER_ROW_KEY] = ledger
    return updated_row, ledger


def converge_fact_ledger(ledger: dict, max_rounds: int = 5) -> dict:
    working = copy.deepcopy(ledger)
    _normalize_fact_ledger(working)
    working['status'] = 'pending'
    working.setdefault('audit', {})
    working['audit']['rounds'] = []
    working['audit'].pop('blockers', None)
    working['audit'].pop('stop_reason', None)
    max_rounds = max(1, int(max_rounds))
    seen_hashes = {_convergence_state_hash(working)}

    for round_index in range(1, max_rounds + 1):
        _normalize_fact_ledger(working)
        before_hash = _convergence_state_hash(working)
        blockers = _validate_joint_constraints(working)
        if not blockers:
            after_hash = _convergence_state_hash(working)
            working['audit']['rounds'].append({
                'round': round_index,
                'phase': 'revalidate',
                'blockers': [],
                'patches': [],
                'before_hash': before_hash,
                'after_hash': after_hash,
            })
            working['status'] = 'converged'
            working['audit']['blockers'] = []
            working['audit']['stop_reason'] = None
            working['audit']['final_hash'] = canonical_ledger_hash(working)
            return working

        unrepairable = [blocker for blocker in blockers if not blocker.get('repairable')]
        if unrepairable:
            working['audit']['rounds'].append({
                'round': round_index,
                'phase': 'validate',
                'blockers': blockers,
                'patches': [],
                'before_hash': before_hash,
                'after_hash': before_hash,
            })
            return _quarantine_ledger(working, blockers, 'unrepairable')

        patches = _minimal_patch_ledger(working, blockers)
        patches.extend(_rederive_fact_ledger(working))
        _normalize_fact_ledger(working)
        after_hash = _convergence_state_hash(working)
        working['audit']['rounds'].append({
            'round': round_index,
            'phase': 'minimal_patch',
            'blockers': blockers,
            'patches': patches,
            'before_hash': before_hash,
            'after_hash': after_hash,
        })
        if after_hash in seen_hashes or (not patches and after_hash == before_hash):
            return _quarantine_ledger(working, _validate_joint_constraints(working), 'oscillation')
        seen_hashes.add(after_hash)

    return _quarantine_ledger(working, _validate_joint_constraints(working), 'max_rounds')


def _time_model_from_row(row):
    payload = row.get(utils.COL_M25_OUTPUT, '')
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError('M2.5 time model sidecar missing')
        payload = json.loads(payload)
    return _validate_time_model(payload)


def _ledger_row_summary(ledger):
    return {
        'status': ledger.get('status'),
        'path': '',
        'input_hash': ledger.get('provenance', {}).get('source_row_hash', ''),
        'ledger_hash': canonical_ledger_hash(ledger),
        'round_count': len(ledger.get('audit', {}).get('rounds') or []),
    }


def _quarantined_shell_ledger(row, staging_system, exc):
    try:
        source_hash = ledger_input_hash(
            row or {}, staging_system, _time_model_from_row(row or {})
        )
    except Exception:
        source_hash = _sha256_json({
            'ledger_code_version': LEDGER_CODE_VERSION,
            'row': _selected_input_fields(row or {}),
            'staging_system': _canonical_jsonable(staging_system or {}),
        })
    ledger = {
        'schema_version': LEDGER_SCHEMA_VERSION,
        'module': LEDGER_MODULE,
        'status': 'quarantined',
        'case_id': _clean((row or {}).get(utils.COL_CASE_ID, '')) or 'unknown',
        'patient': _build_patient(row or {}, staging_system),
        'course': {'underlying_duration': None, 'current_episode_duration': None, 'anchor': 'current_visit'},
        'patient_context': _build_patient_context(row),
        'facts': [],
        'relations': [],
        'audit': {
            'rounds': [],
            'blockers': [{
                'kind': 'build_error',
                'message': str(exc),
                'repairable': False,
            }],
            'stop_reason': 'build_error',
            'final_hash': None,
        },
        'provenance': {
            'source_row_hash': source_hash,
            'code_version': LEDGER_CODE_VERSION,
            'severity_levels': _severity_level_names(staging_system),
        },
    }
    ledger['audit']['final_hash'] = canonical_ledger_hash(ledger)
    return ledger


def _quarantine_ledger(ledger, blockers, stop_reason):
    ledger['status'] = 'quarantined'
    ledger.setdefault('audit', {})['blockers'] = blockers
    ledger['audit']['stop_reason'] = stop_reason
    ledger['audit']['final_hash'] = canonical_ledger_hash(ledger)
    return ledger


def _normalize_fact_ledger(ledger):
    ledger.setdefault('audit', {})
    ledger.setdefault('provenance', {})
    facts = ledger.get('facts') or []
    acuity = str(ledger.get('patient', {}).get('acuity') or '').strip()
    for fact in facts:
        fact.setdefault('feasibility', 'routine')
        if _is_questionable_acute_functional_fact(fact, acuity):
            fact['feasibility'] = 'clinically_questionable'
    facts.sort(key=lambda item: item.get('fact_id', ''))
    ledger['facts'] = facts


def _rederive_fact_ledger(ledger):
    records = _metric_records(ledger)
    blockers = []
    blockers.extend(_same_metric_value_blockers(records))
    blockers.extend(_anion_gap_blockers(records))
    blockers.extend(_bilirubin_blockers(records))
    blockers.extend(_wbc_blockers(records))
    blockers.extend(_red_cell_blockers(records))
    blockers.extend(_blood_gas_blockers(records))
    blockers.extend(_heart_rate_blockers(ledger, records))
    blockers.extend(_oxygen_saturation_blockers(records))
    return _apply_blocker_patches(
        ledger, [blocker for blocker in blockers if blocker.get('repairable')],
        basis='derived_quantity_recalculation'
    )


def _convergence_state_hash(ledger):
    payload = copy.deepcopy(ledger)
    payload.pop('audit', None)
    payload.pop('status', None)
    if isinstance(payload.get('facts'), list):
        payload['facts'] = sorted(payload['facts'], key=lambda item: item.get('fact_id', ''))
    return _sha256_json(_canonical_jsonable(payload))


def _validate_joint_constraints(ledger):
    records = _metric_records(ledger)
    blockers = []
    blockers.extend(_patient_context_blockers(ledger))
    blockers.extend(_time_blockers(ledger))
    blockers.extend(_directional_value_conflict_blockers(ledger))
    blockers.extend(_same_metric_value_blockers(records))
    blockers.extend(_anion_gap_blockers(records))
    blockers.extend(_bilirubin_blockers(records))
    blockers.extend(_wbc_blockers(records))
    blockers.extend(_red_cell_blockers(records))
    blockers.extend(_blood_gas_blockers(records))
    blockers.extend(_creatinine_egfr_blockers(records))
    blockers.extend(_heart_rate_blockers(ledger, records))
    blockers.extend(_oxygen_saturation_blockers(records))
    return blockers


def _minimal_patch_ledger(ledger, blockers):
    return _apply_blocker_patches(
        ledger, blockers, basis='joint_numeric_constraint'
    )


def _apply_blocker_patches(ledger, blockers, basis):
    patches = []
    fact_by_id = {fact.get('fact_id'): fact for fact in ledger.get('facts') or []}
    for blocker in blockers:
        fact = fact_by_id.get(blocker.get('fact_id'))
        expected = blocker.get('expected')
        if fact is None or expected is None or not blocker.get('repairable'):
            continue
        value = fact.get('value')
        if not isinstance(value, dict) or value.get('number') is None:
            continue
        if not _value_within_range(fact, expected):
            continue
        old = value.get('number')
        if _numbers_close(old, expected, blocker.get('tolerance', 0.01)):
            continue
        value['number'] = _normalize_number(round(float(expected), 3))
        fact['derived_from'] = sorted(set((fact.get('derived_from') or []) + blocker.get('sources', [])))
        patches.append({
            'kind': blocker.get('kind'),
            'fact_id': fact.get('fact_id'),
            'from': old,
            'to': value['number'],
            'basis': blocker.get('basis', basis),
        })
        for key in ('standardized_shift', 'projection_distance'):
            if blocker.get(key) is not None:
                patches[-1][key] = blocker[key]
    return patches


def _metric_measurement_context(fact, text, metric):
    context = [_clean(fact.get('context'))]
    context_groups = (
        (('静息', '安静状态', '休息状态'), 'rest'),
        (('初测', '首次测量', '首次测得', '第一次测量'), 'initial_measurement'),
        (('复测', '再测', '重复测量', '再次测量'), 'repeat_measurement'),
        (('治疗后', '处理后', '干预后'), 'post_treatment'),
        (('降压后', '降压治疗后'), 'post_antihypertensive'),
        (('运动后', '活动后', '运动负荷', '负荷后'), 'exercise'),
        (('吸氧后', '氧疗后', '给氧后'), 'on_oxygen'),
        (('未吸氧', '室内空气', '空气下'), 'room_air'),
        (('仰卧位', '卧位'), 'supine'),
        (('站立位', '立位'), 'standing'),
        (('坐位',), 'sitting'),
    )
    for aliases, label in context_groups:
        if any(alias in text for alias in aliases):
            context.append(label)
    if metric in {'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure'}:
        if _has_any(text, ('左臂', '左上肢')):
            context.append('left_arm')
        elif _has_any(text, ('右臂', '右上肢')):
            context.append('right_arm')
    return '+'.join(sorted(set(filter(None, context))))


def _metric_records(ledger):
    records = []
    for fact in ledger.get('facts') or []:
        if fact.get('clinical_state') != 'present':
            continue
        value = fact.get('value')
        if not isinstance(value, dict) or value.get('number') is None:
            continue
        metric = _metric_key(fact)
        if not metric:
            continue
        text = _fact_text(fact)
        raw_number = float(value['number'])
        unit = value.get('unit')
        canonical_input_unit = (
            'fraction'
            if _is_unitless_fraction_metric_fact(metric, fact, raw_number)
            else unit
        )
        normalized = _canonical_metric_value(metric, raw_number, canonical_input_unit)
        time_payload = fact.get('time') or {}
        measurement_context = _metric_measurement_context(fact, text, metric)
        time_key = (
            _clean(time_payload.get('observed_at')),
            _clean(time_payload.get('clinical_onset')),
            _clean(time_payload.get('phase')),
        )
        specimen_group = _specimen_group(fact, text, metric)
        records.append({
            'metric': metric,
            'fact': fact,
            'number': normalized['number'],
            'raw_number': raw_number,
            'unit': unit,
            'canonical_input_unit': canonical_input_unit,
            'unit_key': normalized['unit_key'],
            'unit_supported': normalized['supported'],
            'canonical_unit': normalized['canonical_unit'],
            'interpretation': fact.get('interpretation'),
            'text': text,
            'specimen_group': specimen_group,
            'group_key': (
                measurement_context,
                time_key,
                specimen_group,
                _clean(fact.get('body_site')),
                _clean(fact.get('laterality')),
            ),
            'physio_group_key': (
                measurement_context,
                time_key,
                _clean(fact.get('body_site')),
                _clean(fact.get('laterality')),
            ),
        })
    return records



_UNIT_REQUIRED_METRICS = {
    'sodium', 'chloride', 'bicarbonate', 'anion_gap',
    'bilirubin_total', 'bilirubin_direct', 'bilirubin_indirect',
    'wbc_total', 'neutrophil_abs', 'lymphocyte_abs', 'monocyte_abs',
    'eosinophil_abs', 'basophil_abs',
    'rbc', 'hemoglobin', 'mcv', 'mch', 'mchc', 'paco2', 'creatinine_clearance',
    'heart_rate', 'pulse_rate', 'ecg_heart_rate',
    'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
}


def _metric_requires_explicit_unit(metric):
    return metric in _UNIT_REQUIRED_METRICS

def _canonical_metric_delta(metric, raw_delta, unit):
    normalized = _canonical_metric_value(metric, raw_delta, unit)
    unit_key = _unit_key(unit)
    if metric in {'spo2', 'sao2'} and unit_key in {'', '%', 'percent'}:
        normalized['number'] = float(raw_delta)
    return normalized


def _canonical_metric_value(metric, raw_number, unit):
    unit_key = _unit_key(unit)
    supported = True
    canonical_unit = None
    number = float(raw_number)

    if metric in {'sodium', 'chloride', 'bicarbonate', 'anion_gap'}:
        canonical_unit = 'mmol/L'
        supported = unit_key in {'mmol/l', 'mmol/L'.lower(), 'meq/l'}
    elif metric.startswith('bilirubin_'):
        canonical_unit = 'umol/L'
        if unit_key in {'umol/l', 'μmol/l', 'µmol/l'}:
            pass
        elif unit_key == 'mg/dl':
            number = number * 17.104
        else:
            supported = False
    elif metric in {'wbc_total', 'neutrophil_abs', 'lymphocyte_abs', 'monocyte_abs', 'eosinophil_abs', 'basophil_abs'}:
        canonical_unit = '10^9/L'
        supported = unit_key in {'10^9/l', 'x10^9/l', '×10^9/l', '109/l', 'g/l'}
    elif metric.endswith('_pct'):
        canonical_unit = '%'
        if unit_key in {'', '%', 'percent'}:
            pass
        elif unit_key in {'fraction', 'l/l'}:
            number = number * 100.0
        else:
            supported = False
    elif metric == 'rbc':
        canonical_unit = '10^12/L'
        supported = unit_key in {'10^12/l', 'x10^12/l', '×10^12/l', '1012/l', '10^6/ul', '10^6/µl', '10^6/μl', 'm/ul', 'm/µl', 'm/μl'}
    elif metric == 'hemoglobin':
        canonical_unit = 'g/L'
        if unit_key in {'g/l'}:
            pass
        elif unit_key == 'g/dl':
            number = number * 10.0
        else:
            supported = False
    elif metric == 'hematocrit':
        canonical_unit = '%'
        if unit_key in {'%', 'percent'}:
            pass
        elif unit_key in {'', 'l/l', 'fraction'}:
            if number <= 1.5:
                number = number * 100.0
        else:
            supported = False
    elif metric == 'mcv':
        canonical_unit = 'fL'
        supported = unit_key in {'fl'}
    elif metric == 'mch':
        canonical_unit = 'pg'
        supported = unit_key in {'pg'}
    elif metric == 'mchc':
        canonical_unit = 'g/L'
        if unit_key in {'g/l'}:
            pass
        elif unit_key == 'g/dl':
            number = number * 10.0
        else:
            supported = False
    elif metric in {'blood_ph'}:
        canonical_unit = None
        supported = unit_key in {'', 'ph'}
    elif metric in {'paco2'}:
        canonical_unit = 'mmHg'
        supported = unit_key in {'mmhg'}
    elif metric == 'creatinine_clearance':
        canonical_unit = 'mL/min'
        supported = unit_key in {'ml/min', 'ml/分'}
    elif metric in {
            'heart_rate', 'pulse_rate', 'ecg_heart_rate',
            'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure'}:
        canonical_unit = 'bpm'
        if metric in {'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure'}:
            canonical_unit = 'mmHg'
            supported = unit_key in {'mmhg', '毫米汞柱'}
        else:
            supported = unit_key in {'次/分', 'bpm', '/min', '次/min'}
    elif metric in {'spo2', 'sao2'}:
        canonical_unit = '%'
        if unit_key in {'', '%', 'percent'}:
            if number <= 1.5:
                number = number * 100.0
        elif unit_key in {'fraction', 'l/l'}:
            number = number * 100.0
        else:
            supported = False
    elif metric in {'serum_creatinine', 'egfr'}:
        canonical_unit = unit or None
        supported = True
    if _metric_requires_explicit_unit(metric) and not unit_key:
        supported = False
    return {
        'number': number,
        'unit_key': unit_key,
        'supported': supported,
        'canonical_unit': canonical_unit,
    }


def _is_unitless_fraction_metric_fact(metric, fact, raw_number):
    if (
            not metric
            or (not metric.endswith('_pct') and metric != 'hematocrit')
            or float(raw_number) > 1.5):
        return False
    value = fact.get('value') or {}
    reference = value.get('reference_range') or {}
    if _unit_key(value.get('unit')) or _unit_key(reference.get('unit')):
        return False
    lower = reference.get('lower')
    upper = reference.get('upper')
    if lower is None or upper is None:
        return False
    try:
        range_values = [float(lower), float(upper)]
    except (TypeError, ValueError):
        return False
    if any(not math.isfinite(item) or abs(item) > 1.5 for item in range_values):
        return False
    brace_match = re.search(r'\{([^{}]+)\}', _clean(reference.get('raw')))
    if brace_match:
        for part in brace_match.group(1).split(','):
            try:
                value_part = float(part.strip())
            except ValueError:
                continue
            if math.isfinite(value_part) and abs(value_part) > 1.5:
                return False
    return True


def _unit_key(unit):
    text = _clean(unit)
    text = text.replace('μ', 'µ')
    text = text.replace('／', '/')
    text = re.sub(r'\s+', '', text)
    text = text.replace('升', 'L')
    text = text.replace('微升', 'uL')
    text = text.replace('µL', 'µl')
    text = text.replace('μL', 'µl')
    text = text.lower()
    text = text.replace('μ', 'µ')
    text = text.replace('ul', 'ul')
    if text in {'mmol·l-1', 'mmol/l'}:
        return 'mmol/l'
    if text in {'umol/l', 'µmol/l'}:
        return 'umol/l'
    if text in {'%', '％'}:
        return '%'
    return text


def _specimen_group(fact, text, metric):
    explicit = _clean(fact.get('specimen'))
    if explicit:
        return explicit
    if _has_any(text, ('尿', '尿液')):
        return 'urine'
    if _is_nonblood_specimen_text(text):
        return 'nonblood'
    if metric in {
        'sodium', 'chloride', 'bicarbonate', 'anion_gap',
        'bilirubin_total', 'bilirubin_direct', 'bilirubin_indirect',
        'wbc_total', 'neutrophil_abs', 'lymphocyte_abs', 'monocyte_abs',
        'eosinophil_abs', 'basophil_abs', 'neutrophil_pct', 'lymphocyte_pct',
        'monocyte_pct', 'eosinophil_pct', 'basophil_pct', 'rbc', 'hemoglobin',
        'hematocrit', 'mcv', 'mch', 'mchc', 'blood_ph', 'paco2',
        'serum_creatinine', 'egfr', 'sao2',
    }:
        return 'blood'
    return ''


def _by_metric(records, metric):
    return [record for record in records if record['metric'] == metric]


def _one(records, metric):
    matches = _by_metric(records, metric)
    return matches[0] if matches else None


def _groups_with(records, metrics, key='group_key'):
    grouped = {}
    wanted = set(metrics)
    for record in records:
        if record['metric'] not in wanted:
            continue
        grouped.setdefault(record[key], {}).setdefault(record['metric'], []).append(record)
    return grouped.values()


_SAME_VALUE_METRICS = {
    'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
}


def _same_metric_value_blockers(records):
    blockers = []
    for group in _groups_with(records, _SAME_VALUE_METRICS):
        for metric in sorted(_SAME_VALUE_METRICS):
            aliases = group.get(metric) or []
            if len(aliases) < 2:
                continue
            unit_blockers = _unsupported_unit_blockers(
                'same_metric_value', aliases,
                'Unsupported unit in repeated objective measurement',
            )
            blockers.extend(unit_blockers)
            if unit_blockers:
                continue
            values = [float(record['number']) for record in aliases]
            if max(values) - min(values) <= 1e-6:
                continue
            bounds = [_canonical_record_bounds(record) for record in aliases]
            if any(bound is None for bound in bounds):
                target = aliases[-1]
                blockers.append(_blocker(
                    'same_metric_value', target, None, target['raw_number'],
                    aliases[:-1], 1e-6,
                    'Repeated objective measurement lacks a safe shared range',
                    repairable=False,
                ))
                continue
            if any(not all(math.isfinite(value) for value in bound) for bound in bounds):
                target = aliases[-1]
                blockers.append(_blocker(
                    'same_metric_value', target, None, target['raw_number'],
                    aliases[:-1], 1e-6,
                    'Repeated objective measurement lacks finite repair ranges',
                    repairable=False,
                ))
                continue
            scales = [
                max(_canonical_record_scale(record, bound), 1e-6)
                for record, bound in zip(aliases, bounds)
            ]
            equations = []
            for index in range(1, len(aliases)):
                coefficients = [0.0] * len(aliases)
                coefficients[0] = 1.0
                coefficients[index] = -1.0
                equations.append((coefficients, 0.0))
            projection = _bounded_affine_projection_values(values, bounds, scales, equations)
            if projection is not None:
                projection = _validated_projection(
                    aliases, values, projection['values'], scales
                )
            if projection is None:
                target = aliases[-1]
                blockers.append(_blocker(
                    'same_metric_value', target, None, target['raw_number'],
                    aliases[:-1], 1e-6,
                    'Repeated objective measurements cannot be safely projected to one value',
                    repairable=False,
                ))
                continue
            for index, (target, expected) in enumerate(zip(aliases, projection['values'])):
                if _numbers_close(target['number'], expected, 1e-6):
                    continue
                blocker = _derived_blocker(
                    'same_metric_value', target, expected,
                    [source for source in aliases if source is not target], 1e-6,
                    'Repeated objective aliases must share one value at the same time and context',
                )
                blocker['standardized_shift'] = _normalize_number(
                    projection['standardized_shifts'][index]
                )
                blocker['projection_distance'] = _normalize_number(projection['distance'])
                blockers.append(blocker)
    return blockers


def _one_from(group, metric):
    values = group.get(metric) or []
    return values[0] if values else None


def _unsupported_unit_blockers(kind, records, message):
    blockers = []
    seen = set()
    for record in records:
        if record.get('unit_supported', True):
            continue
        fact_id = record['fact'].get('fact_id')
        if fact_id in seen:
            continue
        seen.add(fact_id)
        blockers.append({
            'kind': 'unsupported_unit',
            'fact_id': fact_id,
            'metric': record.get('metric'),
            'actual': _normalize_number(record.get('raw_number')),
            'expected': None,
            'sources': [],
            'tolerance': None,
            'repairable': False,
            'message': f"{message}: {record.get('metric')} unit={record.get('unit')!r}",
            'constraint': kind,
        })
    return blockers


def _blocker(kind, target, expected, actual, sources, tolerance, message, repairable=None):
    if repairable is None:
        repairable = (
            expected is not None
            and target.get('unit_supported', True)
            and _value_within_range(target['fact'], expected)
        )
    return {
        'kind': kind,
        'fact_id': target['fact'].get('fact_id'),
        'actual': _normalize_number(actual),
        'expected': _normalize_number(round(float(expected), 3)) if expected is not None else None,
        'sources': [source['fact'].get('fact_id') for source in sources],
        'tolerance': tolerance,
        'repairable': bool(repairable),
        'message': message,
    }


def _derived_blocker(kind, target, canonical_expected, sources, canonical_tolerance, message):
    expected = _from_canonical_metric_value(target, canonical_expected)
    tolerance = _delta_from_canonical_metric_value(target, canonical_tolerance)
    if expected is None or tolerance is None:
        return {
            'kind': 'unsupported_unit',
            'fact_id': target['fact'].get('fact_id'),
            'metric': target.get('metric'),
            'actual': _normalize_number(target.get('raw_number')),
            'expected': None,
            'sources': [source['fact'].get('fact_id') for source in sources],
            'tolerance': None,
            'repairable': False,
            'message': f"Cannot convert expected {kind} into target unit {target.get('unit')!r}",
            'constraint': kind,
        }
    repairable = (
        _value_within_range(target['fact'], expected)
        and _records_have_finite_repair_ranges([target] + list(sources), target)
    )
    return _blocker(
        kind, target, expected, target.get('raw_number'), sources, tolerance,
        message, repairable=repairable
    )


def _records_have_finite_repair_ranges(records, target):
    for record in records:
        bounds = _canonical_record_bounds(record)
        if bounds is None or not all(math.isfinite(value) for value in bounds):
            return False
        value = record.get('number')
        if value is None or value < bounds[0] - 1e-9 or value > bounds[1] + 1e-9:
            return False
        if record is target and _canonical_record_scale(record, bounds) <= 0:
            return False
    return True


def _from_canonical_metric_value(record, canonical_number):
    if not record.get('unit_supported', True):
        return None
    metric = record.get('metric')
    unit_key = record.get('unit_key')
    number = float(canonical_number)
    if metric and metric.endswith('_pct') and unit_key in {'fraction', 'l/l'}:
        return number / 100.0
    if metric and metric.startswith('bilirubin_') and unit_key == 'mg/dl':
        return number / 17.104
    if metric == 'mchc' and unit_key == 'g/dl':
        return number / 10.0
    if metric == 'hematocrit' and unit_key in {'l/l', 'fraction'}:
        return number / 100.0
    if metric in {'spo2', 'sao2'} and unit_key in {'l/l', 'fraction'}:
        return number / 100.0
    return number


def _delta_from_canonical_metric_value(record, canonical_delta):
    if not record.get('unit_supported', True):
        return None
    metric = record.get('metric')
    unit_key = record.get('unit_key')
    delta = float(canonical_delta)
    if metric and metric.endswith('_pct') and unit_key in {'fraction', 'l/l'}:
        return delta / 100.0
    if metric and metric.startswith('bilirubin_') and unit_key == 'mg/dl':
        return delta / 17.104
    if metric == 'mchc' and unit_key == 'g/dl':
        return delta / 10.0
    if metric == 'hematocrit' and unit_key in {'l/l', 'fraction'}:
        return delta / 100.0
    if metric in {'spo2', 'sao2'} and unit_key in {'l/l', 'fraction'}:
        return delta / 100.0
    return delta


def _value_within_range(fact, number):
    value = fact.get('value') or {}
    ref = value.get('reference_range') or {}
    lower = ref.get('lower')
    upper = ref.get('upper')
    if lower is not None and float(number) < float(lower) - 1e-9:
        return False
    if upper is not None and float(number) > float(upper) + 1e-9:
        return False
    return True


def _numbers_close(left, right, tolerance):
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= max(float(tolerance), abs(float(right)) * 1e-6)


def _canonical_record_bounds(record):
    reference = (record.get('fact', {}).get('value') or {}).get('reference_range') or {}
    unit = (
        record.get('canonical_input_unit')
        if record.get('canonical_input_unit') == 'fraction' and not reference.get('unit')
        else reference.get('unit') or record.get('unit')
    )
    bounds = []
    for raw_bound, fallback in (
            (reference.get('lower'), -math.inf),
            (reference.get('upper'), math.inf)):
        if raw_bound is None:
            bounds.append(fallback)
            continue
        normalized = _canonical_metric_value(record.get('metric'), raw_bound, unit)
        if not normalized.get('supported'):
            return None
        bounds.append(float(normalized['number']))
    return tuple(bounds)


def _canonical_record_scale(record, bounds):
    reference = (record.get('fact', {}).get('value') or {}).get('reference_range') or {}
    raw = _clean(reference.get('raw'))
    brace_match = re.search(r'\{([^{}]+)\}', raw)
    if brace_match:
        parts = [part.strip() for part in brace_match.group(1).split(',')]
        if len(parts) == 4 and parts[1].lower() != 'nan':
            try:
                raw_sd = float(parts[1])
            except ValueError:
                raw_sd = 0.0
            if raw_sd > 0:
                unit = (
                    record.get('canonical_input_unit')
                    if record.get('canonical_input_unit') == 'fraction' and not reference.get('unit')
                    else reference.get('unit') or record.get('unit')
                )
                normalized = _canonical_metric_delta(record.get('metric'), raw_sd, unit)
                if normalized.get('supported') and normalized['number'] > 0:
                    return float(normalized['number'])
    lower, upper = bounds
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return (upper - lower) / 4.0
    return 0.0


_DIRECTIONAL_NORMAL_INTERVALS = {
    'sodium': (135.0, 145.0),
    'chloride': (98.0, 106.0),
    'bicarbonate': (22.0, 29.0),
    'anion_gap': (8.0, 16.0),
}


def _candidate_preserves_interpretation(record, candidate):
    interval = _DIRECTIONAL_NORMAL_INTERVALS.get(record.get('metric'))
    interpretation = record.get('interpretation')
    if interval is None or interpretation not in {'high', 'low'}:
        return True
    normal_lower, normal_upper = interval
    if interpretation == 'high':
        return candidate >= normal_upper - 1e-9
    return candidate <= normal_lower + 1e-9


def _bounded_linear_projection(records, coefficients):
    """Project numeric records onto one linear equality without leaving M2.7 ranges."""
    if len(records) != len(coefficients) or not records:
        return None
    values = [float(record['number']) for record in records]
    bounds = [_canonical_record_bounds(record) for record in records]
    if any(bound is None for bound in bounds):
        return None
    if any(not all(math.isfinite(value) for value in bound) for bound in bounds):
        return None
    for value, (lower, upper) in zip(values, bounds):
        if value < lower - 1e-9 or value > upper + 1e-9:
            return None

    min_lhs = sum(
        coefficient * (lower if coefficient >= 0 else upper)
        for coefficient, (lower, upper) in zip(coefficients, bounds)
    )
    max_lhs = sum(
        coefficient * (upper if coefficient >= 0 else lower)
        for coefficient, (lower, upper) in zip(coefficients, bounds)
    )
    if min_lhs > 1e-9 or max_lhs < -1e-9:
        return None

    solution = list(values)
    scales = [
        _canonical_record_scale(record, bound)
        for record, bound in zip(records, bounds)
    ]
    free = set()
    for index, scale in enumerate(scales):
        if scale > 0:
            free.add(index)

    for _ in range(len(records) + 1):
        residual = sum(
            coefficient * value
            for coefficient, value in zip(coefficients, solution)
        )
        if abs(residual) <= 1e-9:
            return _validated_projection(records, values, solution, scales)
        denominator = sum(
            (coefficients[index] * scales[index]) ** 2
            for index in free
        )
        if denominator <= 0:
            return None
        proposed = {
            index: solution[index] - residual * coefficients[index]
            * scales[index] ** 2 / denominator
            for index in free
        }
        violations = []
        for index, candidate in proposed.items():
            lower, upper = bounds[index]
            if candidate < lower:
                violations.append(((lower - candidate) / scales[index], index, lower))
            elif candidate > upper:
                violations.append(((candidate - upper) / scales[index], index, upper))
        if violations:
            _, index, boundary = max(violations)
            solution[index] = boundary
            free.remove(index)
            continue
        for index, candidate in proposed.items():
            solution[index] = candidate
        if abs(sum(
            coefficient * value
            for coefficient, value in zip(coefficients, solution)
        )) > 1e-7:
            return None
        return _validated_projection(records, values, solution, scales)
    return None


def _validated_projection(records, original, solution, scales):
    standardized_shifts = [
        0.0 if _numbers_close(candidate, initial, 1e-9)
        else (abs(candidate - initial) / scale if scale > 0 else math.inf)
        for candidate, initial, scale in zip(solution, original, scales)
    ]
    distance = math.sqrt(sum(shift ** 2 for shift in standardized_shifts))
    if max(standardized_shifts, default=0.0) > 3.0 or distance > 4.0:
        return None
    if any(
            not _candidate_preserves_interpretation(record, candidate)
            for record, candidate in zip(records, solution)):
        return None
    return {
        'values': solution,
        'standardized_shifts': standardized_shifts,
        'distance': distance,
    }


def _bounded_affine_projection_values(values, bounds, scales, equations):
    """Project values onto affine equations with finite box bounds."""
    if not values or not equations:
        return None
    if any(bound is None for bound in bounds):
        return None
    if any(not all(math.isfinite(value) for value in bound) for bound in bounds):
        return None
    if any(scale <= 0 or not math.isfinite(scale) for scale in scales):
        return None
    for value, (lower, upper) in zip(values, bounds):
        if value < lower - 1e-9 or value > upper + 1e-9 or lower > upper:
            return None

    best = None
    dimensions = len(values)
    for status in itertools.product((0, -1, 1), repeat=dimensions):
        solution = list(values)
        free = []
        for index, state in enumerate(status):
            if state == 0:
                free.append(index)
            elif state < 0:
                solution[index] = bounds[index][0]
            else:
                solution[index] = bounds[index][1]

        adjusted_rhs = []
        free_rows = []
        for coefficients, rhs in equations:
            fixed = sum(
                coefficients[index] * solution[index]
                for index in range(dimensions)
                if index not in free
            )
            adjusted_rhs.append(rhs - fixed)
            free_rows.append([coefficients[index] for index in free])

        if free:
            free_values = [values[index] for index in free]
            free_scales = [scales[index] for index in free]
            candidate_free = _solve_weighted_affine_projection(
                free_rows, adjusted_rhs, free_values, free_scales
            )
            if candidate_free is None:
                continue
            candidate_valid = True
            for index, candidate in zip(free, candidate_free):
                lower, upper = bounds[index]
                if candidate < lower - 1e-7 or candidate > upper + 1e-7:
                    candidate_valid = False
                    break
                solution[index] = min(max(candidate, lower), upper)
            if not candidate_valid:
                continue
        elif any(abs(rhs) > 1e-7 for rhs in adjusted_rhs):
            continue

        if any(abs(sum(
                coefficient * value
                for coefficient, value in zip(coefficients, solution)
        ) - rhs) > 1e-6 for coefficients, rhs in equations):
            continue
        distance = math.sqrt(sum(
            ((candidate - initial) / scale) ** 2
            for candidate, initial, scale in zip(solution, values, scales)
        ))
        if best is None or distance < best['distance']:
            best = {'values': solution, 'distance': distance}
    return best


def _solve_weighted_affine_projection(rows, rhs, values, scales):
    if not rows:
        return values
    residual = [
        target - sum(coefficient * value for coefficient, value in zip(row, values))
        for row, target in zip(rows, rhs)
    ]
    if all(abs(item) <= 1e-9 for item in residual):
        return list(values)
    gram = []
    for left in rows:
        gram.append([
            sum(
                left[index] * right[index] * scales[index] ** 2
                for index in range(len(values))
            )
            for right in rows
        ])
    multipliers = _solve_linear_system(gram, residual)
    if multipliers is None:
        return None
    return [
        value + scales[index] ** 2 * sum(
            row[index] * multiplier
            for row, multiplier in zip(rows, multipliers)
        )
        for index, value in enumerate(values)
    ]


def _solve_linear_system(matrix, vector):
    size = len(vector)
    if size == 0:
        return []
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= 1e-12:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[-1] for row in augmented]


def _bounded_log_projection(records, equations):
    if any(record['number'] <= 0 for record in records):
        return None
    canonical_bounds = [_canonical_record_bounds(record) for record in records]
    if any(bound is None for bound in canonical_bounds):
        return None
    if any(not all(math.isfinite(value) for value in bound) for bound in canonical_bounds):
        return None
    canonical_scales = [
        _canonical_record_scale(record, bound)
        for record, bound in zip(records, canonical_bounds)
    ]
    if any(scale <= 0 or not math.isfinite(scale) for scale in canonical_scales):
        return None
    adjusted_bounds = []
    for value, lower_upper, scale in zip(
            [record['number'] for record in records], canonical_bounds, canonical_scales):
        lower, upper = lower_upper
        guarded_lower = max(lower, value - 3.0 * scale)
        guarded_upper = min(upper, value + 3.0 * scale)
        if (
                not math.isfinite(guarded_lower)
                or not math.isfinite(guarded_upper)
                or guarded_upper <= 0
                or guarded_lower > guarded_upper):
            return None
        adjusted_bounds.append((
            math.log(max(guarded_lower, min(value, guarded_upper) * 1e-6, 1e-12)),
            math.log(guarded_upper),
        ))
    log_values = [math.log(record['number']) for record in records]
    log_scales = [
        max(scale / max(record['number'], 1e-12), 1e-9)
        for record, scale in zip(records, canonical_scales)
    ]
    log_projection = _bounded_affine_projection_values(
        log_values, adjusted_bounds, log_scales, equations
    )
    if log_projection is None:
        return None
    canonical_solution = [math.exp(value) for value in log_projection['values']]
    return _validated_projection(
        records,
        [record['number'] for record in records],
        canonical_solution,
        canonical_scales,
    )


def _projection_blockers(kind, records, projection, message):
    blockers = []
    for target, expected, standardized_shift in zip(
            records, projection['values'], projection['standardized_shifts']):
        if _numbers_close(target['number'], expected, 1e-6):
            continue
        sources = [record for record in records if record is not target]
        blocker = _derived_blocker(
            kind, target, expected, sources, 1e-6, message
        )
        blocker['basis'] = 'bounded_joint_projection'
        blocker['standardized_shift'] = _normalize_number(round(standardized_shift, 6))
        blocker['projection_distance'] = _normalize_number(round(projection['distance'], 6))
        blockers.append(blocker)
    return blockers


def _joint_projection_blockers(kind, records, coefficients, message):
    projection = _bounded_linear_projection(records, coefficients)
    if projection is None:
        return None
    return _projection_blockers(kind, records, projection, message)


def _anion_gap_blockers(records):
    blockers = []
    for group in _groups_with(records, {'sodium', 'chloride', 'bicarbonate', 'anion_gap'}):
        sodium = _one_from(group, 'sodium')
        chloride = _one_from(group, 'chloride')
        bicarbonate = _one_from(group, 'bicarbonate')
        ag = _one_from(group, 'anion_gap')
        present = [record for record in (sodium, chloride, bicarbonate, ag) if record]
        if len(present) < 4:
            continue
        unit_blockers = _unsupported_unit_blockers('anion_gap', present, 'Unsupported unit in serum anion gap group')
        blockers.extend(unit_blockers)
        if unit_blockers:
            continue
        expected = sodium['number'] - chloride['number'] - bicarbonate['number']
        if _numbers_close(ag['number'], expected, 1.0):
            continue
        direct_blocker = _derived_blocker(
            'anion_gap', ag, expected, [sodium, chloride, bicarbonate], 1.0,
            'AG must equal Na-Cl-HCO3 for the same observed_at/context/specimen group'
        )
        if direct_blocker.get('repairable'):
            blockers.append(direct_blocker)
            continue
        joint_blockers = _joint_projection_blockers(
            'anion_gap_joint_projection',
            [sodium, chloride, bicarbonate, ag],
            [1.0, -1.0, -1.0, -1.0],
            'Na, Cl, HCO3 and AG jointly projected within their M2.7 ranges',
        )
        blockers.extend(joint_blockers if joint_blockers is not None else [direct_blocker])
    return blockers


def _bilirubin_blockers(records):
    blockers = []
    for group in _groups_with(records, {'bilirubin_total', 'bilirubin_direct', 'bilirubin_indirect'}):
        total = _one_from(group, 'bilirubin_total')
        direct = _one_from(group, 'bilirubin_direct')
        indirect = _one_from(group, 'bilirubin_indirect')
        present = [record for record in (total, direct, indirect) if record]
        if len(present) < 3:
            continue
        blockers.extend(_unsupported_unit_blockers('bilirubin_triad', present, 'Unsupported bilirubin unit'))
        if any(not record.get('unit_supported', True) for record in present):
            continue
        expected = total['number'] - direct['number']
        if _numbers_close(indirect['number'], expected, 1.0):
            continue
        direct_blocker = _derived_blocker(
            'bilirubin_triad', indirect, expected, [total, direct], 1.0,
            'Indirect bilirubin must equal total-direct bilirubin within the same observed_at/context/specimen group'
        )
        if direct_blocker.get('repairable'):
            blockers.append(direct_blocker)
            continue
        joint_blockers = _joint_projection_blockers(
            'bilirubin_triad_joint_projection',
            [total, direct, indirect],
            [1.0, -1.0, -1.0],
            'Total, direct and indirect bilirubin jointly projected within their M2.7 ranges',
        )
        blockers.extend(joint_blockers if joint_blockers is not None else [direct_blocker])
    return blockers


def _wbc_blockers(records):
    blockers = []
    metrics = {'wbc_total'} | {f'{prefix}_{suffix}' for prefix in ('neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil') for suffix in ('pct', 'abs')}
    for group in _groups_with(records, metrics):
        total = _one_from(group, 'wbc_total')
        if total is None:
            continue
        for prefix in ('neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil'):
            pct = _one_from(group, f'{prefix}_pct')
            absolute = _one_from(group, f'{prefix}_abs')
            present = [record for record in (total, pct, absolute) if record]
            if len(present) < 3:
                continue
            blockers.extend(_unsupported_unit_blockers('wbc_differential', present, 'Unsupported WBC differential unit'))
            if any(not record.get('unit_supported', True) for record in present):
                continue
            expected = total['number'] * pct['number'] / 100.0
            tolerance = max(0.2, abs(expected) * 0.05)
            if _numbers_close(absolute['number'], expected, tolerance):
                continue
            direct_blocker = _derived_blocker(
                'wbc_differential', absolute, expected, [total, pct], tolerance,
                'WBC differential absolute value must equal WBC*pct/100 within the same observed_at/context/specimen group'
            )
            if direct_blocker.get('repairable'):
                blockers.append(direct_blocker)
                continue
            projection = _bounded_log_projection(
                [total, pct, absolute],
                [([1.0, 1.0, -1.0], math.log(100.0))],
            )
            if projection is None:
                blockers.append(direct_blocker)
            else:
                blockers.extend(_projection_blockers(
                    'wbc_differential_joint_projection',
                    [total, pct, absolute],
                    projection,
                    'WBC total, differential percent and absolute count jointly projected within their M2.7 ranges',
                ))
    return blockers


def _red_cell_blockers(records):
    blockers = []
    metrics = {'rbc', 'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc'}
    for group in _groups_with(records, metrics):
        rbc = _one_from(group, 'rbc')
        hb = _one_from(group, 'hemoglobin')
        hct = _one_from(group, 'hematocrit')
        if not all((rbc, hb, hct)) or rbc['number'] == 0 or hct['number'] == 0:
            continue
        core = [rbc, hb, hct]
        targets = [_one_from(group, metric) for metric in ('mcv', 'mch', 'mchc')]
        present = core + [target for target in targets if target]
        blockers.extend(_unsupported_unit_blockers('red_cell_indices', present, 'Unsupported red-cell index unit'))
        if any(not record.get('unit_supported', True) for record in present):
            continue
        checks = [
            ('mcv', hct['number'] * 10.0 / rbc['number'], 1.0, 'MCV=Hct%*10/RBC'),
            ('mch', hb['number'] / rbc['number'], 1.0, 'MCH=Hb(g/L)/RBC'),
            ('mchc', hb['number'] * 100.0 / hct['number'], 5.0, 'MCHC=Hb(g/L)*100/Hct%'),
        ]
        direct_blockers = []
        for metric, expected, tolerance, message in checks:
            target = _one_from(group, metric)
            if target is None or _numbers_close(target['number'], expected, tolerance):
                continue
            direct_blockers.append(_derived_blocker('red_cell_indices', target, expected, core, tolerance, message))
        if not direct_blockers:
            continue
        if all(blocker.get('repairable') for blocker in direct_blockers):
            blockers.extend(direct_blockers)
            continue
        projection_records = core + [target for target in targets if target]
        record_index = {record['metric']: index for index, record in enumerate(projection_records)}
        equations = []
        if 'mcv' in record_index:
            coefficients = [0.0] * len(projection_records)
            coefficients[record_index['rbc']] = 1.0
            coefficients[record_index['hematocrit']] = -1.0
            coefficients[record_index['mcv']] = 1.0
            equations.append((coefficients, math.log(10.0)))
        if 'mch' in record_index:
            coefficients = [0.0] * len(projection_records)
            coefficients[record_index['rbc']] = 1.0
            coefficients[record_index['hemoglobin']] = -1.0
            coefficients[record_index['mch']] = 1.0
            equations.append((coefficients, 0.0))
        if 'mchc' in record_index:
            coefficients = [0.0] * len(projection_records)
            coefficients[record_index['hemoglobin']] = -1.0
            coefficients[record_index['hematocrit']] = 1.0
            coefficients[record_index['mchc']] = 1.0
            equations.append((coefficients, math.log(100.0)))
        projection = _bounded_log_projection(projection_records, equations)
        if projection is None:
            blockers.extend(direct_blockers)
        else:
            blockers.extend(_projection_blockers(
                'red_cell_indices_joint_projection',
                projection_records,
                projection,
                'RBC, Hb, Hct and observed red-cell indices jointly projected within their M2.7 ranges',
            ))
    return blockers


def _blood_gas_blockers(records):
    blockers = []
    for group in _groups_with(records, {'blood_ph', 'paco2', 'bicarbonate'}):
        ph = _one_from(group, 'blood_ph')
        pco2 = _one_from(group, 'paco2')
        bicarbonate = _one_from(group, 'bicarbonate')
        present = [record for record in (ph, pco2, bicarbonate) if record]
        if len(present) < 3 or pco2['number'] <= 0:
            continue
        blockers.extend(_unsupported_unit_blockers('blood_gas_hh', present, 'Unsupported blood gas unit'))
        if any(not record.get('unit_supported', True) for record in present):
            continue
        expected = 0.03 * pco2['number'] * (10 ** (ph['number'] - 6.1))
        tolerance = max(2.0, abs(expected) * 0.1)
        if _numbers_close(bicarbonate['number'], expected, tolerance):
            continue
        blockers.append(_derived_blocker(
            'blood_gas_hh', bicarbonate, expected, [ph, pco2], tolerance,
            'HCO3 must satisfy Henderson-Hasselbalch with pH and PaCO2 in the same observed_at/context/specimen group'
        ))
    return blockers


def _creatinine_egfr_blockers(records):
    blockers = []
    for group in _groups_with(records, {'serum_creatinine', 'egfr'}):
        creatinine = _one_from(group, 'serum_creatinine')
        egfr = _one_from(group, 'egfr')
        if not creatinine or not egfr:
            continue
        cr_high = creatinine.get('interpretation') == 'high'
        cr_low = creatinine.get('interpretation') == 'low'
        egfr_high = egfr.get('interpretation') == 'high'
        egfr_low = egfr.get('interpretation') == 'low'
        if (cr_high and egfr_high) or (cr_low and egfr_low):
            blockers.append({
                'kind': 'creatinine_egfr_direction',
                'fact_id': egfr['fact'].get('fact_id'),
                'actual': _normalize_number(egfr['raw_number']),
                'expected': None,
                'sources': [creatinine['fact'].get('fact_id')],
                'tolerance': None,
                'repairable': False,
                'message': 'Serum creatinine and eGFR directions conflict in the same observed_at/context/specimen group; CKD-EPI is not inferred locally',
            })
    return blockers


def _heart_rate_blockers(ledger, records):
    if _has_pulse_deficit_context(ledger):
        return []
    blockers = []
    for group in _groups_with(records, {'heart_rate', 'pulse_rate', 'ecg_heart_rate'}, key='physio_group_key'):
        hr = _one_from(group, 'heart_rate')
        pulse = _one_from(group, 'pulse_rate')
        ecg = _one_from(group, 'ecg_heart_rate')
        present = [record for record in (hr, pulse, ecg) if record]
        blockers.extend(_unsupported_unit_blockers('heart_rate', present, 'Unsupported heart-rate unit'))
        if any(not record.get('unit_supported', True) for record in present):
            continue
        if hr and ecg and not _numbers_close(ecg['number'], hr['number'], 5.0):
            blockers.append(_derived_blocker(
                'heart_rate_ecg', ecg, hr['number'], [hr], 5.0,
                'ECG heart rate should match heart rate without pulse-deficit context'
            ))
        base_values = [record['number'] for record in (hr, ecg) if record]
        if pulse and base_values:
            expected = sum(base_values) / len(base_values)
            if not _numbers_close(pulse['number'], expected, 5.0):
                blockers.append(_derived_blocker(
                    'heart_rate_pulse', pulse, expected, [record for record in (hr, ecg) if record], 5.0,
                    'Pulse should match heart/ECG rate without pulse-deficit context'
                ))
    return blockers


def _bounded_difference_projection(left, right, tolerance):
    records = [left, right]
    values = [float(record['number']) for record in records]
    bounds = [_canonical_record_bounds(record) for record in records]
    if any(bound is None for bound in bounds):
        return None
    if any(not all(math.isfinite(value) for value in bound) for bound in bounds):
        return None
    if any(
            value < lower - 1e-9 or value > upper + 1e-9
            for value, (lower, upper) in zip(values, bounds)):
        return None
    scales = [
        max(_canonical_record_scale(record, bound), 1e-6)
        for record, bound in zip(records, bounds)
    ]
    difference = values[0] - values[1]
    rhs = float(tolerance) if difference > 0 else -float(tolerance)
    projection = _bounded_affine_projection_values(
        values, bounds, scales, [([1.0, -1.0], rhs)]
    )
    if projection is None:
        return None
    return _validated_projection(records, values, projection['values'], scales)


def _oxygen_saturation_blockers(records):
    blockers = []
    for group in _groups_with(records, {'spo2', 'sao2'}, key='physio_group_key'):
        spo2 = _one_from(group, 'spo2')
        sao2 = _one_from(group, 'sao2')
        present = [record for record in (spo2, sao2) if record]
        if len(present) < 2:
            continue
        blockers.extend(_unsupported_unit_blockers(
            'spo2_sao2', present, 'Unsupported oxygen saturation unit'
        ))
        if any(not record.get('unit_supported', True) for record in present):
            continue
        if _numbers_close(spo2['number'], sao2['number'], 3.0):
            continue
        projection = _bounded_difference_projection(spo2, sao2, 3.0)
        if projection is None:
            blockers.append(_blocker(
                'spo2_sao2', sao2, None, sao2['raw_number'], [spo2], 3.0,
                'SpO2 and SaO2 ranges cannot satisfy the allowed difference',
                repairable=False,
            ))
            continue
        for target, expected, source in (
                (spo2, projection['values'][0], sao2),
                (sao2, projection['values'][1], spo2)):
            if _numbers_close(target['number'], expected, 1e-6):
                continue
            blockers.append(_derived_blocker(
                'spo2_sao2', target, expected, [source], 1e-6,
                'SpO2 and SaO2 should remain close; SvO2 and PaO2 are out of scope',
            ))
    return blockers


def _directional_value_conflict_blockers(ledger):
    blockers = []
    for fact in ledger.get('facts') or []:
        for conflict in (fact.get('provenance') or {}).get('value_conflicts') or []:
            blockers.append({
                'kind': 'directional_value_conflict',
                'fact_id': fact.get('fact_id'),
                'actual': conflict,
                'expected': None,
                'sources': [],
                'tolerance': None,
                'repairable': False,
                'message': 'Generic abnormal numeric value conflicts with directional finding',
            })
    return blockers


def _fact_is_comorbidity_only(fact):
    provenance = fact.get('provenance') or {}
    source_levels = [_clean(provenance.get('source_level'))]
    source_levels.extend(
        _clean(item.get('source_level'))
        for item in provenance.get('sources') or []
        if isinstance(item, dict)
    )
    return bool(source_levels) and all(level == '伴随疾病' for level in source_levels)


def _patient_context_blockers(ledger):
    blockers = []
    patient = ledger.get('patient') or {}
    sex = str(patient.get('sex') or '').strip()
    level = str(patient.get('severity_level') or '').strip()
    levels = ledger.get('provenance', {}).get('severity_levels') or []
    if levels and levels.count(level) != 1:
        blockers.append({
            'kind': 'severity_level',
            'message': f'severity level must match exactly one M1 level: {level}',
            'repairable': False,
        })
    laterality = str(patient.get('laterality') or '').strip()
    for fact in ledger.get('facts') or []:
        if fact.get('clinical_state') == 'absent' or fact.get('interpretation') == 'negative':
            continue
        text = _fact_text(fact)
        comorbidity_only = _fact_is_comorbidity_only(fact)
        if sex == '男' and _has_any(text, ('妊娠', '孕妇', '子宫', '卵巢', '宫颈', '阴道', '外阴')):
            blockers.append({'kind': 'sex_specific_fact', 'fact_id': fact.get('fact_id'), 'message': text, 'repairable': False})
        if sex == '女' and _has_any(text, ('前列腺', '睾丸', '阴茎', '阴囊', '精液')):
            blockers.append({'kind': 'sex_specific_fact', 'fact_id': fact.get('fact_id'), 'message': text, 'repairable': False})
        if not comorbidity_only and laterality in {'左侧', 'left'} and _mentions_laterality(text, 'right'):
            blockers.append({'kind': 'laterality_conflict', 'fact_id': fact.get('fact_id'), 'message': text, 'repairable': False})
        if not comorbidity_only and laterality in {'右侧', 'right'} and _mentions_laterality(text, 'left'):
            blockers.append({'kind': 'laterality_conflict', 'fact_id': fact.get('fact_id'), 'message': text, 'repairable': False})
    return blockers


def _mentions_laterality(text, side):
    paired_organs = (
        '肾盂|肾盏|输尿管|肾|胸腔|胸膜|肺上叶|肺中叶|肺下叶|肺|'
        '上叶|中叶|下叶|卵巢|附件|睾丸|阴囊|'
        '膝关节|膝|踝关节|踝|腕关节|腕|肘关节|肘|髋关节|髋|'
        '上肢|下肢|大腿|小腿|肢体|足|手'
    )
    side_word = '右' if side == 'right' else '左'
    pattern = rf'(?:{side_word}(?:侧)?(?P<site>{paired_organs})|{side_word}侧)'
    for match in re.finditer(pattern, text):
        if side == 'left' and 'TRAUBE' in text[match.start():match.end() + 16].upper():
            continue
        if _is_bilateral_laterality(text, match, paired_organs):
            continue
        if not _is_referred_pain_laterality(text, match.start(), match.end()):
            return True
    return False


def _is_bilateral_laterality(text, laterality_match, paired_organs):
    comparison = text[laterality_match.end():laterality_match.end() + 8]
    if not re.match(r'^(?:较重|更重|为重|为著|为主|较明显|更明显|为甚|重于对侧)', comparison):
        return False
    bilateral_mentions = list(re.finditer(
        rf'双(?:侧)?(?P<site>{paired_organs})',
        text[:laterality_match.start()],
    ))
    if not bilateral_mentions:
        return False
    antecedent = bilateral_mentions[-1]
    between = text[antecedent.end():laterality_match.start()]
    if len(between) > 12 or re.search(r'[伴并且及和与]', between):
        return False
    body_site = laterality_match.group('site')
    if not body_site:
        return True
    bilateral_site = antecedent.group('site')
    if body_site == bilateral_site:
        return True
    parent_sites = {
        '肺上叶': '肺', '肺中叶': '肺', '肺下叶': '肺',
        '上叶': '肺', '中叶': '肺', '下叶': '肺',
        '肾盂': '肾', '肾盏': '肾',
        '手': '上肢', '腕': '上肢', '腕关节': '上肢',
        '肘': '上肢', '肘关节': '上肢',
        '足': '下肢', '踝': '下肢', '踝关节': '下肢',
        '膝': '下肢', '膝关节': '下肢', '髋': '下肢', '髋关节': '下肢',
        '大腿': '下肢', '小腿': '下肢',
    }
    return parent_sites.get(body_site) == bilateral_site


def _is_referred_pain_laterality(text, start, end):
    before = text[max(0, start - 8):start]
    after = text[end:end + 12]
    if re.search(r'(?:放射|放散|牵涉)(?:至|到|向)?$', before):
        return True
    if before.endswith('向') and re.match(r'^[^伴并且及和与，,；;。]{0,8}(?:放射|放散|牵涉)', after):
        return True
    return bool(re.match(
        r'^[^伴并且及和与，,；;。]{0,10}(?:牵涉痛|放射痛|放散痛|牵涉性疼痛|放射性疼痛|放散性疼痛)',
        after,
    ))


def _time_blockers(ledger):
    blockers = []
    for fact in ledger.get('facts') or []:
        time_payload = fact.get('time') or {}
        for key in ('clinical_onset', 'observed_at'):
            label = time_payload.get(key)
            if label is None or _valid_time_label(label):
                continue
            blockers.append({
                'kind': 'time_label',
                'fact_id': fact.get('fact_id'),
                'message': f'invalid {key}: {label}',
                'repairable': False,
            })
    return blockers


def _valid_time_label(label):
    text = str(label or '').strip().upper()
    return bool(re.match(r'^(?:H|D|W|M|Y)(?:0|-[0-9]+(?:\.[0-9]+)?)$', text))


def _is_relative_blood_pressure_metric_text(text, folded):
    if (
            not any(term in text for term in ('收缩压', '舒张压', '平均动脉压'))
            and not any(term in folded for term in ('SBP', 'DBP', 'MAP'))):
        return False
    if any(term in text for term in (
            '收缩压力', '舒张压力', '收缩压负荷', '舒张压负荷', '血压负荷', '动态血压',
            '肺动脉', 'PASP', '右心室', '左心室', '心室', '食管', '咽部', '括约肌',
            '膀胱', '尿道', '宫缩', '子宫收缩',
            '收缩压差', '舒张压差', '平均动脉压差', '血压差', '压差', '差值', '差异', '差距',
            '波动', '体位性', '直立性', '站立后', '站立3分钟', '较基础值', '基础值',
            '吸气时', '脉搏奇异', '踝臂', '上下肢', '双上肢', '两上肢', '双侧上肢',
            '左右上肢', '对侧')):
        return True
    if '较' in text and any(term in text for term in (
            '上臂', '上肢', '下肢', '踝部', '基础值', '基线', '左臂', '右臂',
            '左上肢', '右上肢')):
        return True
    return False


def _metric_key(fact):
    text = _fact_text(fact)
    folded = text.upper().replace(' ', '')
    domain = fact.get('domain')
    if 'A/G' in folded or '白蛋白/球蛋白' in text:
        return None
    if _is_nonblood_text(text):
        if _has_any(text, (
                '白细胞', 'WBC', '红细胞', 'RBC', '肌酐', '钠', 'NA', '氯', 'CL',
                '碳酸氢', 'HCO3', '阴离子间隙', 'AG', '中性粒', 'NEUT', '淋巴',
                'LYMPH', '单核', 'MONO', '嗜酸', 'EOS', '嗜碱', 'BASO')):
            return None
    if ('阴离子间隙' in text or folded in {'AG', 'AG升高', 'AG降低'}) and '校正' not in text:
        return 'anion_gap'
    if _is_sodium_metric_text(text, folded):
        return 'sodium'
    if '氯' in text or folded in {'CL', 'CL-', '血清CL'}:
        return 'chloride'
    if '碳酸氢根' in text or 'HCO3' in folded:
        return 'bicarbonate'
    if '总胆红素' in text or 'TBIL' in folded:
        return 'bilirubin_total'
    if '直接胆红素' in text or 'DBIL' in folded:
        return 'bilirubin_direct'
    if '间接胆红素' in text or 'IBIL' in folded:
        return 'bilirubin_indirect'
    if ('白细胞计数' in text or folded.startswith('WBC')) and not _is_nonblood_text(text):
        return 'wbc_total'
    differential = _wbc_differential_prefix(text, folded)
    if differential:
        if _has_any(text, ('比例', '百分比')) or '%' in str((fact.get('value') or {}).get('unit') or ''):
            return f'{differential}_pct'
        if '绝对值' in text or '计数' in text or '#' in text:
            return f'{differential}_abs'
    if '平均红细胞血红蛋白浓度' in text or folded.startswith('MCHC'):
        return 'mchc'
    if '平均红细胞血红蛋白量' in text or (folded.startswith('MCH') and not folded.startswith('MCHC')):
        return 'mch'
    if '平均红细胞体积' in text or folded.startswith('MCV'):
        return 'mcv'
    if '网织红细胞' in text or 'RETIC' in folded:
        return None
    if ('红细胞计数' in text or folded.startswith('RBC')) and not _is_nonblood_text(text):
        return 'rbc'
    if '糖化血红蛋白' in text or folded.startswith('HBA1C'):
        return 'hba1c'
    if '血红蛋白' in text or folded.startswith('HB') or folded.startswith('HGB'):
        return 'hemoglobin'
    if '血小板' in text or folded.startswith('PLT'):
        return 'platelet'
    if '红细胞压积' in text or folded.startswith('HCT'):
        return 'hematocrit'
    if ('PH' in folded or '酸碱度' in text) and '尿' not in text:
        return 'blood_ph'
    if '二氧化碳分压' in text or 'PACO2' in folded or 'PCO2' in folded:
        return 'paco2'
    if '肌酐清除率' in text or 'CREATININECLEARANCE' in folded or folded.startswith('CCR'):
        return 'creatinine_clearance'
    if ('血清肌酐' in text or ('肌酐' in text and '尿肌酐' not in text)) \
            and not _is_nonblood_text(text) \
            and not _has_any(text, ('肌酐清除率', '肌酐比', '肌酐比值')):
        return 'serum_creatinine'
    if 'EGFR' in folded or '估算肾小球滤过率' in text:
        return 'egfr'
    if '奇脉' not in text and not _is_relative_blood_pressure_metric_text(text, folded):
        if '收缩压' in text or 'SBP' in folded:
            return 'systolic_bp'
        if '舒张压' in text or 'DBP' in folded:
            return 'diastolic_bp'
        if '平均动脉压' in text or 'MAP' in folded:
            return 'mean_arterial_pressure'
    if ('心电图' in text or 'ECG' in folded) and _has_any(text, ('心率', '心室率')):
        return 'ecg_heart_rate'
    if '脉率' in text:
        return 'pulse_rate'
    if '心率' in text and '胎心' not in text:
        return 'heart_rate'
    if 'SVO2' in folded or '静脉血氧饱和度' in text or 'PAO2' in folded or '氧分压' in text:
        return None
    if 'SAO2' in folded or '动脉血氧饱和度' in text:
        return 'sao2'
    if 'SPO2' in folded or '血氧饱和度' in text:
        return 'spo2'
    return None


def _wbc_differential_prefix(text, folded):
    compact = re.sub(r'[^A-Z0-9]+', '', str(folded or '').upper())
    chinese_subtype = (
        '中性粒' in text
        and _has_any(text, ('杆状', '带状', '分叶', '未成熟', '幼稚'))
    )
    if chinese_subtype or _has_any(
            compact,
            ('BANDNEUT', 'BANDCELL', 'STABNEUT', 'SEGMENTEDNEUT',
             'SEGNEUT', 'IMMATURENEUT', 'IMMATUREGRANULOCYTE')):
        return None
    pairs = (
        ('neutrophil', ('中性粒', 'NEUT')),
        ('lymphocyte', ('淋巴', 'LYMPH')),
        ('monocyte', ('单核', 'MONO')),
        ('eosinophil', ('嗜酸', 'EOS')),
        ('basophil', ('嗜碱', 'BASO')),
    )
    for key, aliases in pairs:
        if any(alias in text or alias in folded for alias in aliases):
            return key
    return None


def _is_nonblood_text(text):
    return _is_nonblood_specimen_text(text)


def _is_nonblood_specimen_text(text):
    return _has_any(text, _NONBLOOD_SPECIMEN_TERMS)


def _is_sodium_metric_text(text, folded):
    if _has_any(text, ('利钠肽', '脑钠肽')) or 'BNP' in folded:
        return False
    if folded in {'NA', 'NA+', '血清NA', '血NA', 'SERUMNA', 'SERUMNA+'}:
        return True
    return _has_any(text, ('血钠', '血清钠', '钠离子', '血清钠离子'))


def _fact_text(fact):
    concept = fact.get('concept') or {}
    return f"{concept.get('raw') or ''} {concept.get('canonical') or ''}"


def _has_any(text, needles):
    folded = str(text or '').upper()
    return any(str(needle).upper() in folded for needle in needles)


def _has_pulse_deficit_context(ledger):
    for fact in ledger.get('facts') or []:
        text = _fact_text(fact)
        if _has_any(text, ('心房颤动', '房颤', '脉搏短绌')):
            return True
    return False


def _is_questionable_acute_functional_fact(fact, acuity):
    if fact.get('domain') != 'functional' or acuity not in {'急性', '慢性急性加重'}:
        return False
    text = _fact_text(fact)
    return _has_any(text, ('运动', '步行', '6MWT', '六分钟步行', '负荷', '肺功能', 'FEV1', 'FVC'))


def _severity_level_names(staging_system):
    if not isinstance(staging_system, dict):
        return []
    levels = staging_system.get('levels') or []
    names = []
    for level in levels:
        if isinstance(level, dict):
            name = _clean(level.get('name'))
            if name:
                names.append(name)
    return names



def _ensure_converged_ledger(ledger: dict) -> None:
    if not isinstance(ledger, dict):
        raise ValueError('fact ledger must be a dict')
    if ledger.get('status') != 'converged':
        raise ValueError('fact ledger status is not converged')
    if not isinstance(ledger.get('facts'), list):
        raise ValueError('fact ledger facts missing or invalid')


def _project_fact(fact: dict, include_value: bool = True) -> dict:
    concept = fact.get('concept') if isinstance(fact.get('concept'), dict) else {}
    projected = {
        'fact_id': fact.get('fact_id'),
        'domain': fact.get('domain'),
        'raw': _clean(concept.get('raw')),
        'canonical': _clean(concept.get('canonical')),
        'clinical_state': fact.get('clinical_state'),
        'interpretation': fact.get('interpretation'),
        'time': copy.deepcopy(fact.get('time') or {}),
        'context': fact.get('context'),
        'feasibility': fact.get('feasibility') or 'routine',
    }
    if include_value:
        projected['value'] = copy.deepcopy(fact.get('value'))
    return projected


def _is_home_measurable_fact(fact: dict) -> bool:
    if fact.get('domain') != 'sign':
        return False
    text = _fact_text(fact)
    return _has_any(text, ('体温', '血压', '心率', '脉率', '呼吸频率', '呼吸次数', 'SPO2', '血氧饱和度'))


def project_patient_view(ledger: dict) -> dict:
    """Project facts visible to the simulated patient without diagnostic values."""
    _ensure_converged_ledger(ledger)
    patient = copy.deepcopy(ledger.get('patient') or {})
    course = copy.deepcopy(ledger.get('course') or {})
    symptom_facts = []
    home_measurements = []
    for fact in ledger.get('facts') or []:
        domain = fact.get('domain')
        if domain == 'symptom':
            symptom_facts.append(_project_fact(fact, include_value=False))
        elif _is_home_measurable_fact(fact):
            home_measurements.append(_project_fact(fact, include_value=True))
    patient_context = copy.deepcopy(ledger.get('patient_context') or {})
    return {
        'schema_version': 'm3_patient_view.v1',
        'case_id': ledger.get('case_id'),
        'patient': patient,
        'course': {
            'underlying_duration': course.get('underlying_duration'),
            'current_episode_duration': course.get('current_episode_duration'),
            'anchor': course.get('anchor'),
        },
        'patient_context': {
            'symptom_attributes': patient_context.get('symptom_attributes') or [],
            'prior_visit': patient_context.get('prior_visit') or {},
        },
        'symptoms': symptom_facts,
        'home_measurements': home_measurements,
    }


def project_physical_exam_view(ledger: dict) -> dict:
    """Project only physical-exam facts; unknown findings are normal defaults downstream."""
    _ensure_converged_ledger(ledger)
    return {
        'schema_version': 'm3_physical_exam_view.v1',
        'case_id': ledger.get('case_id'),
        'facts': [
            _project_fact(fact, include_value=True)
            for fact in ledger.get('facts') or []
            if fact.get('domain') == 'sign'
        ],
    }


def _normalize_request_text(text: str) -> str:
    return re.sub(r'[\s\-_/：:；;，,、（）()\[\]【】]+', '', str(text or '')).upper()


def _extract_request_section(text: str, marker: str) -> str:
    pattern = re.compile(
        rf'(?:\[|【){re.escape(marker)}(?:\]|】)\s*[:：]\s*(.*?)'
        rf'(?=(?:\r?\n)?\s*(?:\[|【)[^\]\】\r\n]+(?:\]|】)\s*[:：]|$)',
        flags=re.S,
    )
    match = pattern.search(str(text or ''))
    return match.group(1).strip() if match else ''


def _split_request_items(text: str, max_items: int | None = None) -> list[str]:
    body = str(text or '').strip()
    if not body:
        return []
    instruction = re.match(r'^(?:请)?(?:检测|检查)\s*[:：]\s*', body)
    if instruction:
        body = body[instruction.end():]
    items = _split_top_level_items(body)
    if max_items is not None:
        items = items[:max_items]
    return items


def _split_top_level_items(text: str) -> list[str]:
    bracket_pairs = {'(': ')', '（': '）', '[': ']', '［': '］', '【': '】', '{': '}', '｛': '｝'}
    closing_brackets = set(bracket_pairs.values())
    numbered_item = re.compile(r'(?:\(\d+\)|（\d+）|\d+\.(?!\d)|\d+\)|\d+、)\s*')
    separators = frozenset('、，,；;\n\r')
    items = []
    current = []
    stack = []

    def flush():
        item = re.sub(r'^(?:[-*•]\s*)+', '', ''.join(current).strip())
        if item:
            items.append(item)
        current.clear()

    index = 0
    while index < len(text):
        if (
            not stack
            and (index == 0 or text[index - 1].isspace() or text[index - 1] in separators)
            and (match := numbered_item.match(text, index)) is not None
        ):
            flush()
            index = match.end()
            continue
        char = text[index]
        if char in bracket_pairs:
            stack.append(bracket_pairs[char])
            current.append(char)
        elif char in closing_brackets:
            if not stack or char != stack[-1]:
                return [text.strip()] if text.strip() else []
            stack.pop()
            current.append(char)
        elif not stack and char in separators:
            flush()
        else:
            current.append(char)
        index += 1
    if stack:
        return [text.strip()] if text.strip() else []
    flush()
    return items


_LAB_REQUEST_KEYWORDS = (
    '血常规', '白细胞', '红细胞', '血红蛋白', '血小板', '肝功能', '肾功能', '电解质',
    '凝血', 'D-二聚体', '肌钙蛋白', 'BNP', 'NT-PROBNP', 'CRP', 'PCT', 'ESR', '尿常规',
    '血糖', '糖化血红蛋白', 'HBA1C', '血脂', '胆红素', '白蛋白', '肌酐', '尿素',
    '钠', '钾', '氯', '血气', 'PH', 'PACO2', 'HCO3', '乳酸', '降钙素原',
)
_IMAGING_REQUEST_KEYWORDS = (
    'CT', 'MRI', 'X线', 'X光', '胸片', '片', '超声', '彩超', '影像', '造影', 'CTA', 'MRA', 'PET',
)
_FUNCTIONAL_REQUEST_KEYWORDS = (
    '心电图', 'ECG', 'HOLTER', '动态心电', '肺功能', 'FEV1', 'FVC', '6MWT', '六分钟步行',
    '运动试验', '负荷试验', '脑电图', '肌电图', '听力', '视野',
)
_PANEL_KEYWORDS = {
    '血常规': (
        'WBC', '白细胞', 'RBC', '红细胞', 'HGB', 'HB', '血红蛋白', 'HCT', '红细胞压积',
        'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', '血小板', 'MPV', '中性粒', '淋巴', '单核', '嗜酸', '嗜碱',
    ),
    '肝功能': ('ALT', 'AST', 'ALP', 'GGT', '胆红素', '总蛋白', '白蛋白', '球蛋白', 'A/G'),
    '肾功能': ('肌酐', '尿素', '尿酸', 'EGFR', '胱抑素'),
    '电解质': ('NA', '钠', 'K', '钾', 'CL', '氯', '钙', 'CA', '镁', 'MG', '磷', 'P', 'HCO3', '碳酸氢'),
    '凝血': ('PT', 'INR', 'APTT', 'TT', '纤维蛋白原', 'D-二聚体'),
    '心肌标志物': ('肌钙蛋白', 'CK', 'CK-MB', '肌红蛋白', 'BNP', 'NT-PROBNP'),
    '炎症': ('CRP', 'PCT', 'ESR', '降钙素原'),
    '尿常规': ('尿蛋白', '尿糖', '尿酮体', '尿隐血', '尿白细胞', '尿红细胞', '管型', '尿比重'),
    '肺功能': ('肺功能', 'FEV1', 'FVC', 'FEV1/FVC', 'DLCO', 'PEF', '肺活量', '用力肺活量', '弥散'),
}

_PANEL_EXCLUDE_KEYWORDS = {
    '血常规': ('糖化血红蛋白', 'HBA1C', '糖化HB'),
}


def _classify_request_item(item: str) -> str:
    normalized = _normalize_request_text(item)
    if any(_normalize_request_text(keyword) in normalized for keyword in _FUNCTIONAL_REQUEST_KEYWORDS):
        return 'functional'
    if any(_normalize_request_text(keyword) in normalized for keyword in _IMAGING_REQUEST_KEYWORDS):
        return 'imaging'
    if any(_normalize_request_text(keyword) in normalized for keyword in _LAB_REQUEST_KEYWORDS):
        return 'lab'
    return 'lab'


def route_clinical_requests(text: str) -> dict[str, list[str]]:
    exam_body = _extract_request_section(text, '申请查体')
    lab_body = _extract_request_section(text, '申请化验')
    physical_items = _split_request_items(exam_body)
    diagnostic_items = _split_request_items(lab_body, max_items=8)
    routed = {'physical_exam': physical_items, 'lab': [], 'imaging': [], 'functional': []}
    for item in diagnostic_items:
        routed[_classify_request_item(item)].append(item)
    return routed


def _ascii_tokens(text: str) -> set[str]:
    return set(re.findall(r'[A-Z0-9]+', str(text or '').upper()))


def _panel_alias_matches_fact(alias: str, fact_norm: str) -> bool:
    alias_norm = _normalize_request_text(alias)
    if not alias_norm:
        return False
    if re.fullmatch(r'[A-Z0-9]+', alias_norm):
        return alias_norm in _ascii_tokens(fact_norm)
    return alias_norm in fact_norm


def _panel_excludes_fact(panel: str, fact: dict) -> bool:
    fact_norm = _normalize_request_text(_fact_text(fact))
    return any(
        _normalize_request_text(keyword) in fact_norm
        for keyword in _PANEL_EXCLUDE_KEYWORDS.get(panel, ())
    )


def _request_matches_panel(item: str, fact: dict) -> bool:
    item_norm = _normalize_request_text(item)
    fact_norm = _normalize_request_text(_fact_text(fact))
    for panel, aliases in _PANEL_KEYWORDS.items():
        if _normalize_request_text(panel) not in item_norm:
            continue
        if _panel_excludes_fact(panel, fact):
            return False
        if any(_panel_alias_matches_fact(alias, fact_norm) for alias in aliases):
            return True
    return False


def _normalized_contains_concept(needle_norm: str, haystack_norm: str) -> bool:
    if not needle_norm or not haystack_norm:
        return False
    if re.fullmatch(r'[A-Z0-9]+', needle_norm):
        return needle_norm in _ascii_tokens(haystack_norm)
    return needle_norm in haystack_norm


def _is_hba1c_fact(fact: dict) -> bool:
    text = _normalize_request_text(_fact_text(fact))
    return '糖化血红蛋白' in _fact_text(fact) or 'HBA1C' in text


def _request_matches_fact(item: str, fact: dict) -> bool:
    item_norm = _normalize_request_text(item)
    concept = fact.get('concept') if isinstance(fact.get('concept'), dict) else {}
    raw_norm = _normalize_request_text(concept.get('raw'))
    canonical_norm = _normalize_request_text(concept.get('canonical'))
    if not item_norm:
        return False
    hemoglobin_request = item_norm in {'血红蛋白', 'HB', 'HGB'}
    if hemoglobin_request and _is_hba1c_fact(fact):
        return False
    if _normalized_contains_concept(item_norm, raw_norm) or _normalized_contains_concept(item_norm, canonical_norm):
        return True
    if canonical_norm and _normalized_contains_concept(canonical_norm, item_norm):
        return True
    return _request_matches_panel(item, fact)


def _diagnostic_request_items(text: str) -> list[str]:
    lab_body = _extract_request_section(text, '申请化验')
    if lab_body:
        return _split_request_items(lab_body, max_items=8)
    return _split_request_items(text, max_items=8)


def _append_unique_fact(target: list, fact: dict) -> None:
    fact_id = fact.get('fact_id')
    if any(existing.get('fact_id') == fact_id for existing in target):
        return
    target.append(_project_fact(fact, include_value=True))


def project_diagnostic_view(ledger: dict, request_text: str) -> dict:
    """Project only requested diagnostic facts from the ledger."""
    _ensure_converged_ledger(ledger)
    diagnostic_facts = [
        fact for fact in ledger.get('facts') or []
        if fact.get('domain') in {'lab', 'imaging', 'functional'}
    ]
    routed = route_clinical_requests(request_text)
    facts_by_domain = {'lab': [], 'imaging': [], 'functional': []}
    for item in _diagnostic_request_items(request_text):
        matches = [fact for fact in diagnostic_facts if _request_matches_fact(item, fact)]
        if matches:
            for fact in matches:
                _append_unique_fact(facts_by_domain[fact.get('domain')], fact)
            continue
        routed_domain = _classify_request_item(item)
        if routed_domain not in facts_by_domain:
            continue
    return {
        'schema_version': 'm3_diagnostic_view.v1',
        'case_id': ledger.get('case_id'),
        'request_text': str(request_text or ''),
        'routed_requests': routed,
        'facts': facts_by_domain,
    }


def _ledger_summary_from_sidecar(csv_path: str, row_index: int) -> dict:
    row_path = utils._module_io_abs(csv_path, row_index)
    row_payload = utils._read_json_file(row_path, default=None)
    if not isinstance(row_payload, dict):
        raise ValueError(f'fact ledger sidecar row_{row_index}.json is missing')
    summary = row_payload.get(COL_M281_OUTPUT)
    if not isinstance(summary, dict):
        raise ValueError(f'fact ledger summary missing for row {row_index}')
    return summary


def build_m3_ledger_metadata(csv_path: str, row_index: int, ledger: dict) -> dict:
    summary = _ledger_summary_from_sidecar(csv_path, row_index)
    return {
        'schema_version': 'm3_fact_ledger_reference.v1',
        'case_id': ledger.get('case_id'),
        'row_index': row_index,
        'path': summary.get('path'),
        'absolute_path': os.path.join(utils._module_io_dir(csv_path), summary.get('path') or ''),
        'input_hash': summary.get('input_hash'),
        'ledger_hash': summary.get('ledger_hash'),
    }


def load_verified_fact_ledger_for_m3(csv_path: str, row_index: int,
                                     row: dict, staging_system: dict) -> dict:
    expected_case_id = _clean((row or {}).get(utils.COL_CASE_ID, ''))
    ledger = load_fact_ledger(csv_path, row_index, expected_case_id=expected_case_id)
    expected_input_hash = ledger_input_hash(row or {}, staging_system, _time_model_from_row(row or {}))
    actual_input_hash = ledger.get('provenance', {}).get('source_row_hash')
    if actual_input_hash != expected_input_hash:
        raise ValueError(
            f'fact ledger input_hash stale for row {row_index}: expected {expected_input_hash}, got {actual_input_hash}'
        )
    return {
        'ledger': ledger,
        'metadata': build_m3_ledger_metadata(csv_path, row_index, ledger),
    }


def load_m3_ledger_metadata(csv_path: str, row_index: int) -> dict:
    row_payload = utils._read_json_file(utils._module_io_abs(csv_path, row_index), default={}) or {}
    metadata = row_payload.get(M3_LEDGER_METADATA_KEY)
    return metadata if isinstance(metadata, dict) else {}


def write_m3_ledger_metadata(csv_path: str, row_index: int, metadata: dict) -> None:
    row_path = utils._module_io_abs(csv_path, row_index)
    row_payload = utils._read_json_file(row_path, default={}) or {}
    row_payload[M3_LEDGER_METADATA_KEY] = copy.deepcopy(metadata)
    utils._atomic_write_json_file(row_path, row_payload)


def m3_ledger_metadata_current(csv_path: str, row_index: int, expected_metadata: dict) -> bool:
    metadata = load_m3_ledger_metadata(csv_path, row_index)
    required_keys = ('path', 'input_hash', 'ledger_hash', 'case_id')
    return all(
        metadata.get(key) == expected_metadata.get(key)
        for key in required_keys
    )

def write_fact_ledger(csv_path: str, row_index: int, ledger: dict) -> dict:
    _validate_ledger_for_sidecar(ledger, require_final_hash=False)
    ledger_to_write = copy.deepcopy(ledger)
    io_dir = utils._module_io_dir(csv_path)
    ledger_name = f'row_{row_index}.fact_ledger.json'
    ledger_path = os.path.join(io_dir, ledger_name)
    ledger_hash = canonical_ledger_hash(ledger_to_write)
    ledger_to_write['audit']['final_hash'] = ledger_hash
    input_hash = ledger_to_write.get('provenance', {}).get('source_row_hash') or ''
    round_count = len(ledger_to_write.get('audit', {}).get('rounds') or [])
    summary = {
        'status': ledger.get('status'),
        'path': ledger_name,
        'input_hash': input_hash,
        'ledger_hash': ledger_hash,
        'round_count': round_count,
    }

    utils._atomic_write_json_file(ledger_path, ledger_to_write)
    row_path = utils._module_io_abs(csv_path, row_index)
    row_payload = utils._read_json_file(row_path, default={}) or {}
    row_payload[COL_M281_OUTPUT] = summary
    utils._atomic_write_json_file(row_path, row_payload)
    _update_manifest(csv_path, row_index, ledger_to_write, summary)
    return summary


def load_fact_ledger(csv_path: str, row_index: int, expected_case_id: str = '') -> dict:
    row_path = utils._module_io_abs(csv_path, row_index)
    row_payload = utils._read_json_file(row_path, default=None)
    if not isinstance(row_payload, dict):
        raise ValueError(f'fact ledger sidecar row_{row_index}.json is missing')
    summary = row_payload.get(COL_M281_OUTPUT)
    if not isinstance(summary, dict):
        raise ValueError(f'fact ledger summary missing for row {row_index}')
    if summary.get('status') != 'converged':
        raise ValueError(f'fact ledger status is not converged for row {row_index}')
    rel_path = summary.get('path')
    if not isinstance(rel_path, str) or os.path.basename(rel_path) != rel_path:
        raise ValueError(f'fact ledger path invalid for row {row_index}')
    ledger_path = os.path.join(utils._module_io_dir(csv_path), rel_path)
    ledger = utils._read_json_file(ledger_path, default=None)
    _validate_ledger_for_sidecar(ledger, require_final_hash=True)
    if expected_case_id and ledger.get('case_id') != expected_case_id:
        raise ValueError(
            f"fact ledger case_id mismatch: expected {expected_case_id}, got {ledger.get('case_id')}"
        )
    actual_hash = canonical_ledger_hash(ledger)
    if summary.get('ledger_hash') != actual_hash:
        raise ValueError('fact ledger ledger_hash mismatch')
    if ledger.get('audit', {}).get('final_hash') != actual_hash:
        raise ValueError('fact ledger audit final_hash mismatch')
    input_hash = ledger.get('provenance', {}).get('source_row_hash')
    if summary.get('input_hash') != input_hash:
        raise ValueError('fact ledger input_hash mismatch')
    return ledger


def _selected_input_fields(row):
    columns = [
        utils.COL_CASE_ID, utils.COL_AGE, utils.COL_GENDER, utils.COL_DIAGNOSIS,
        utils.COL_STAGE, utils.COL_PATIENT_LATERALITY, utils.COL_ACUITY,
        utils.COL_DURATION_TOTAL, utils.COL_PRIOR_VISITED,
        utils.COL_PRIOR_VISIT_COUNT, utils.COL_PRIOR_VISIT_HISTORY,
        utils.COL_SYMPTOMS, utils.COL_SIGNS,
        utils.COL_LAB_TESTS, utils.COL_IMAGING, utils.COL_FUNCTIONAL_TESTS,
        utils.COL_ABSENT_SYMPTOMS, utils.COL_ABSENT_SIGNS,
        utils.COL_ABSENT_LAB_TESTS, utils.COL_ABSENT_IMAGING,
        utils.COL_ABSENT_FUNCTIONAL, utils.COL_TIME_ORDER, utils.COL_QUANTIFIED,
        utils.COL_SPECIFIC,
    ]
    list_columns = {
        utils.COL_SYMPTOMS, utils.COL_SIGNS, utils.COL_LAB_TESTS,
        utils.COL_IMAGING, utils.COL_FUNCTIONAL_TESTS,
        utils.COL_ABSENT_SYMPTOMS, utils.COL_ABSENT_SIGNS,
        utils.COL_ABSENT_LAB_TESTS, utils.COL_ABSENT_IMAGING,
        utils.COL_ABSENT_FUNCTIONAL, utils.COL_TIME_ORDER,
        utils.COL_QUANTIFIED, utils.COL_SPECIFIC,
    }
    selected = {}
    for column in columns:
        value = row.get(column, '')
        if column in list_columns:
            selected[column] = _canonical_jsonable(_parse_items(value))
        else:
            selected[column] = value
    return selected


def _canonical_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _canonical_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(item) for item in value]
    return value


def _canonical_ledger_payload(ledger):
    payload = copy.deepcopy(ledger)
    if isinstance(payload.get('audit'), dict):
        payload['audit'].pop('final_hash', None)
    if isinstance(payload.get('facts'), list):
        payload['facts'] = sorted(payload['facts'], key=lambda item: item.get('fact_id', ''))
    return _canonical_jsonable(payload)


def _sha256_json(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _validate_time_model(time_model):
    if not isinstance(time_model, dict) or time_model.get('schema_version') != 2:
        raise ValueError('time_model must use schema_version=2')
    timeline = time_model.get('symptom_timeline')
    if not isinstance(timeline, list):
        raise ValueError('time_model schema_version=2 requires symptom_timeline list')
    normalized_timeline = []
    for item in timeline:
        if isinstance(item, dict):
            name = _clean(item.get('name'))
            category = _clean(item.get('category'))
            phase = _clean(item.get('phase'))
            time_label = _clean(item.get('time_label')).upper()
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, category, phase, time_label = (_clean(v) for v in item)
            time_label = time_label.upper()
        else:
            raise ValueError('time_model schema_version=2 symptom_timeline entries must be dicts or 4-tuples')
        if not name or category != '症状' or not phase or not time_label:
            raise ValueError('time_model schema_version=2 symptom_timeline entry is incomplete')
        normalized_timeline.append({
            'name': name,
            'category': category,
            'phase': phase,
            'time_label': time_label,
        })
    normalized = dict(time_model)
    normalized['symptom_timeline'] = normalized_timeline
    return normalized


def _build_time_index(row, time_model):
    index = {}
    for item in time_model.get('symptom_timeline', []):
        index[('症状', _canonical_concept(item['name']))] = {
            'phase': item['phase'],
            'time_label': item['time_label'],
        }
    for item in _parse_items(row.get(utils.COL_TIME_ORDER, '')):
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        name = _clean(item[0])
        category = _clean(item[1])
        phase = _clean(item[2])
        time_label = _clean(item[3]).upper()
        if name and category and phase and time_label:
            index[(category, _canonical_concept(name))] = {
                'phase': phase,
                'time_label': time_label,
            }
    return index


def _build_value_index(row):
    specific = {}
    quantified = {}
    specific_by_canonical = {}
    quantified_by_canonical = {}
    for item in _parse_items(row.get(utils.COL_SPECIFIC, '')):
        key, value = _timeline_value(item)
        if key and value:
            specific[key] = value
            specific_by_canonical.setdefault(key[:2], []).append((key, value))
    for item in _parse_items(row.get(utils.COL_QUANTIFIED, '')):
        key, value = _timeline_value(item)
        if key and value:
            quantified[key] = value
            quantified_by_canonical.setdefault(key[:2], []).append((key, value))
    value_index = {}
    for key, raw_value in specific.items():
        canonical_key = key[:2]
        raw_range = quantified.get(key)
        if raw_range is None and len(quantified_by_canonical.get(canonical_key, [])) == 1:
            raw_range = quantified_by_canonical[canonical_key][0][1]
        value_index[key] = _parse_value(raw_value, raw_range)
    for canonical_key, entries in specific_by_canonical.items():
        if len(entries) != 1:
            continue
        key, raw_value = entries[0]
        raw_range = quantified.get(key)
        if raw_range is None and len(quantified_by_canonical.get(canonical_key, [])) == 1:
            raw_range = quantified_by_canonical[canonical_key][0][1]
        value_index[canonical_key] = _parse_value(raw_value, raw_range)
    return value_index


def _timeline_value(item):
    if not isinstance(item, (list, tuple)) or len(item) < 4:
        return None, None
    name, encoded_value = _split_value_encoded_name(item[0])
    category = _clean(item[1])
    value = encoded_value
    if not value and len(item) >= 5:
        value = _clean(item[4])
    if not name or not category or not value:
        return None, None
    return (category, _canonical_concept(name), name), value


def _split_value_encoded_name(value):
    text = _clean(value)
    if not text:
        return '', ''
    actual_match = re.search(r'[；;]?\s*实际值\s*=\s*(.+)$', text)
    if actual_match:
        name = text[:actual_match.start()]
        return _strip_metric_encoding(name), _clean(actual_match.group(1))
    brace_match = re.search(r'\{[^{}]+\}', text)
    if brace_match:
        end = brace_match.end()
        unit_match = re.match(r'[^；;，,\s]*', text[end:])
        unit = unit_match.group(0) if unit_match else ''
        raw_range = text[brace_match.start():end] + unit
        return _strip_metric_encoding(text[:brace_match.start()]), raw_range
    return text, ''


def _strip_metric_encoding(value):
    return re.sub(r'[；;，,\s]+$', '', _clean(value))


def _build_patient(row, staging_system):
    diagnosis = _clean(row.get(utils.COL_DIAGNOSIS, ''))
    fixed_laterality = utils.fixed_anatomic_laterality(diagnosis)
    return {
        'age': _parse_age(row.get(utils.COL_AGE, '')),
        'sex': _clean(row.get(utils.COL_GENDER, '')),
        'diagnosis': diagnosis,
        'severity_scheme': _severity_scheme(staging_system),
        'severity_level': _clean(row.get(utils.COL_STAGE, '')),
        'laterality': fixed_laterality or _normalize_laterality(row.get(utils.COL_PATIENT_LATERALITY, '')),
        'acuity': _clean(row.get(utils.COL_ACUITY, '')),
    }


def _parse_int_or_zero(value):
    text = _clean(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _symptom_attributes_from_row(row):
    attributes = []
    for source_index, item in enumerate(_parse_items(row.get(utils.COL_SYMPTOMS, ''))):
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        name = _clean(item[0])
        payload = {
            'name': name,
            'duration': _clean(item[2]),
            'trigger': _clean(item[3]),
            'nature': _clean(item[4]),
            'source_index': source_index,
        }
        if name and any(payload.get(key) for key in ('duration', 'trigger', 'nature')):
            attributes.append(payload)
    return attributes


def _build_patient_context(row):
    return {
        'symptom_attributes': _symptom_attributes_from_row(row),
        'prior_visit': {
            'visited': _clean(row.get(utils.COL_PRIOR_VISITED, '')),
            'count': _parse_int_or_zero(row.get(utils.COL_PRIOR_VISIT_COUNT, '')),
            'history': _clean(row.get(utils.COL_PRIOR_VISIT_HISTORY, '')),
        },
    }


def _build_fact(raw_name, category, domain, clinical_state, source_column,
                source_index, source_level, time_index, value_index):
    canonical = _canonical_concept(raw_name)
    time = _fact_time(category, domain, canonical, time_index)
    context = 'current_visit'
    identity = {
        'domain': domain,
        'canonical': canonical,
        'specimen': None,
        'body_site': None,
        'laterality': None,
        'time': time,
        'context': context,
    }
    fact = {
        'fact_id': _sha256_json(identity),
        'domain': domain,
        'concept': {'raw': raw_name, 'canonical': canonical},
        'clinical_state': clinical_state,
        'interpretation': _interpretation(raw_name, clinical_state),
        'acquisition_state': _acquisition_state(domain),
        'value': None if clinical_state != 'present' else (
            value_index.get((category, canonical, raw_name))
            or value_index.get((category, canonical))
        ),
        'time': time,
        'specimen': None,
        'body_site': None,
        'laterality': None,
        'context': context,
        'visibility': _visibility(domain),
        'feasibility': 'routine',
        'provenance': {
            'source': source_column,
            'source_index': source_index,
            'source_level': _clean(source_level),
        },
        'derived_from': [],
    }
    return fact


def _add_fact(facts, identity_facts, fact):
    fact_id = fact['fact_id']
    prior = identity_facts.get(fact_id)
    if prior is None:
        identity_facts[fact_id] = fact
        facts.append(fact)
        return

    prior_state = prior['clinical_state']
    state = fact['clinical_state']
    if {prior_state, state} == {'present', 'absent'}:
        if _present_absent_assertions_are_compatible(prior, fact):
            present_fact = prior if prior_state == 'present' else fact
            absent_fact = fact if state == 'absent' else prior
            _record_compatible_absent_assertion(present_fact, absent_fact)
            if present_fact is not prior:
                _replace_fact(facts, identity_facts, fact_id, present_fact)
            return
        raise ValueError(
            f"present/absent conflict for fact identity {fact['concept']['canonical']}"
        )
    if state == 'present' and prior_state == 'present':
        _merge_present_fact(facts, identity_facts, prior, fact)
        return
    if state == 'absent' and prior_state == 'absent':
        if _absent_absent_assertions_are_compatible(prior, fact):
            _record_compatible_absent_assertion(
                prior, fact, compatibility='absent_absent'
            )
        _merge_fact_provenance(prior, prior, fact)
        return


def _merge_present_fact(facts, identity_facts, prior, fact):
    prior_interp = prior.get('interpretation')
    new_interp = fact.get('interpretation')
    if prior_interp and new_interp and prior_interp != new_interp:
        if not _present_interpretations_are_compatible(prior, fact):
            raise ValueError(
                f"interpretation conflict for fact identity {fact['concept']['canonical']}"
            )
    value_conflict = _present_value_assertions_conflict(prior, fact)
    merged = _preferred_present_fact(prior, fact)
    if merged is not prior:
        _replace_fact(facts, identity_facts, fact['fact_id'], merged)
    _merge_fact_provenance(merged, prior, fact)
    if value_conflict:
        _record_present_value_conflict(merged, prior, fact, value_conflict)


def _preferred_present_fact(prior, fact):
    positive_directional = _positive_high_preferred_fact(prior, fact)
    if positive_directional is not None:
        return positive_directional
    if _interpretation_rank(fact.get('interpretation')) > _interpretation_rank(prior.get('interpretation')):
        return fact
    return prior


def _present_value_assertions_conflict(left, right):
    interpretations = {left.get('interpretation'), right.get('interpretation')}
    if 'abnormal' not in interpretations:
        return None
    if not (interpretations & {'high', 'low', 'marked_high', 'marked_low'}):
        return None
    left_value = left.get('value') if isinstance(left.get('value'), dict) else None
    right_value = right.get('value') if isinstance(right.get('value'), dict) else None
    if not left_value or not right_value:
        return None
    if left_value.get('number') is None or right_value.get('number') is None:
        return None
    if not _fact_value_units_safely_comparable(left, right):
        return {
            'left': _fact_value_conflict_record(left),
            'right': _fact_value_conflict_record(right),
            'unit_conflict': True,
        }
    if _numbers_close(left_value.get('number'), right_value.get('number'), 1e-6):
        return None
    if _value_number_within_value_range(left_value.get('number'), right_value):
        return None
    if _value_number_within_value_range(right_value.get('number'), left_value):
        return None
    return {
        'left': _fact_value_conflict_record(left),
        'right': _fact_value_conflict_record(right),
    }


def _fact_value_units_safely_comparable(left, right):
    left_key = _fact_value_unit_key(left)
    right_key = _fact_value_unit_key(right)
    if left_key == right_key:
        return True
    if not left_key or not right_key:
        return False
    if {left_key, right_key} <= {'mmol/l', 'meq/l'}:
        text = f"{_fact_text(left)} {_fact_text(right)}"
        if _has_any(text, ('钾', 'K+', 'K＋', '钠', 'NA', '氯', 'CL', '碳酸氢', 'HCO3')):
            return True
    return False


def _fact_value_unit_key(fact):
    value = fact.get('value') if isinstance(fact.get('value'), dict) else {}
    reference = value.get('reference_range') or {}
    return _unit_key(value.get('unit') or reference.get('unit'))


def _value_number_within_value_range(number, value):
    reference = value.get('reference_range') or {}
    lower = reference.get('lower')
    upper = reference.get('upper')
    if lower is None and upper is None:
        return _numbers_close(number, value.get('number'), 1e-6)
    if lower is not None and float(number) < float(lower) - 1e-9:
        return False
    if upper is not None and float(number) > float(upper) + 1e-9:
        return False
    return True


def _fact_value_conflict_record(fact):
    value = fact.get('value') or {}
    concept = fact.get('concept') or {}
    return {
        'raw': concept.get('raw'),
        'interpretation': fact.get('interpretation'),
        'number': value.get('number'),
        'unit': value.get('unit'),
        'reference_range': value.get('reference_range'),
    }


def _record_present_value_conflict(target, left, right, conflict):
    provenance = target.setdefault('provenance', {})
    conflicts = provenance.setdefault('value_conflicts', [])
    if conflict not in conflicts:
        conflicts.append(conflict)

def _replace_fact(facts, identity_facts, fact_id, fact):
    for index, existing in enumerate(facts):
        if existing['fact_id'] == fact_id:
            facts[index] = fact
            break
    identity_facts[fact_id] = fact


def _present_absent_assertions_are_compatible(left, right):
    present_fact = left if left.get('clinical_state') == 'present' else right
    absent_fact = right if present_fact is left else left
    present_qualifier = _directional_qualifier((present_fact.get('concept') or {}).get('raw'))
    absent_qualifier = _directional_qualifier((absent_fact.get('concept') or {}).get('raw'))
    if not present_qualifier or not absent_qualifier:
        return False
    present_family = _direction_family(present_qualifier)
    absent_family = _direction_family(absent_qualifier)
    if present_family and absent_family and present_family != absent_family:
        return True
    if present_qualifier in {'high', 'low'} and absent_qualifier in {'marked_high', 'marked_low'}:
        return present_family == absent_family
    if present_qualifier == 'abnormal' and absent_qualifier in {'high', 'low', 'marked_high', 'marked_low'}:
        return True
    return False


def _present_interpretations_are_compatible(left, right):
    left_interp = left.get('interpretation')
    right_interp = right.get('interpretation')
    if left_interp == right_interp:
        return True
    if _direction_family(left_interp) and _direction_family(left_interp) == _direction_family(right_interp):
        return True
    canonical = (left.get('concept') or {}).get('canonical') or (right.get('concept') or {}).get('canonical')
    if 'abnormal' in {left_interp, right_interp} and (
            {left_interp, right_interp} & {'high', 'low', 'marked_high', 'marked_low'}):
        return True
    if canonical == '尿蛋白' and {left_interp, right_interp} <= {'positive', 'high', 'marked_high'}:
        return True
    if _lab_positive_high_interpretations_are_compatible(left, right):
        return True
    return False


def _lab_positive_high_interpretations_are_compatible(left, right):
    interpretations = {left.get('interpretation'), right.get('interpretation')}
    if 'positive' not in interpretations:
        return False
    if not interpretations <= {'positive', 'high', 'marked_high'}:
        return False
    if left.get('domain') != 'lab' or right.get('domain') != 'lab':
        return False
    canonical = (left.get('concept') or {}).get('canonical') or (right.get('concept') or {}).get('canonical')
    if canonical == '尿蛋白':
        return False
    return _has_quantified_directional_fact(left, right) or _looks_like_serology_fact(left, right)


def _positive_high_preferred_fact(left, right):
    canonical = (left.get('concept') or {}).get('canonical') or (right.get('concept') or {}).get('canonical')
    interpretations = {left.get('interpretation'), right.get('interpretation')}
    if canonical == '尿蛋白' and 'positive' in interpretations and interpretations <= {'positive', 'high', 'marked_high'}:
        return left if left.get('interpretation') == 'positive' else right
    if not _lab_positive_high_interpretations_are_compatible(left, right):
        return None
    directional = [
        item for item in (left, right)
        if item.get('interpretation') in {'high', 'marked_high'}
    ]
    if not directional:
        return None
    return max(directional, key=lambda item: _interpretation_rank(item.get('interpretation')))


def _has_quantified_directional_fact(left, right):
    return any(
        item.get('interpretation') in {'high', 'marked_high'} and item.get('value') is not None
        for item in (left, right)
    )


def _looks_like_serology_fact(left, right):
    text = ' '.join(
        _fact_text(item) for item in (left, right)
    ).upper()
    return any(token in text for token in (
        '抗体', '抗原', '因子', '免疫球蛋白', '补体', 'RF', 'ANA', 'ANCA', 'IGG', 'IGA', 'IGM',
    ))

def _absent_absent_assertions_are_compatible(left, right):
    left_qualifier = _directional_qualifier((left.get('concept') or {}).get('raw'))
    right_qualifier = _directional_qualifier((right.get('concept') or {}).get('raw'))
    if not left_qualifier or not right_qualifier:
        return False
    if left_qualifier == right_qualifier:
        return True
    if _direction_family(left_qualifier) in {'high', 'low'} and \
            _direction_family(right_qualifier) in {'high', 'low'}:
        return True
    if 'abnormal' in {left_qualifier, right_qualifier}:
        return True
    return False


def _interpretation_rank(interpretation):
    return {
        None: 0,
        'negative': 1,
        'positive': 2,
        'abnormal': 1,
        'high': 2,
        'low': 2,
        'marked_high': 3,
        'marked_low': 3,
    }.get(interpretation, 2)


def _direction_family(qualifier):
    if qualifier in {'high', 'marked_high'}:
        return 'high'
    if qualifier in {'low', 'marked_low'}:
        return 'low'
    if qualifier in {'positive', 'negative'}:
        return qualifier
    return None


def _merge_fact_provenance(target, *facts_to_merge):
    provenance = target.setdefault('provenance', {})
    sources = provenance.setdefault('sources', [])
    compatible_absent = provenance.setdefault('compatible_absent_assertions', [])
    for item in facts_to_merge:
        source = _fact_source_record(item)
        if source not in sources:
            sources.append(source)
        for assertion in (item.get('provenance') or {}).get('compatible_absent_assertions') or []:
            if assertion not in compatible_absent:
                compatible_absent.append(assertion)
    if not compatible_absent:
        provenance.pop('compatible_absent_assertions', None)


def _fact_source_record(fact):
    provenance = fact.get('provenance') or {}
    concept = fact.get('concept') or {}
    record = {
        'source': provenance.get('source'),
        'source_index': provenance.get('source_index'),
        'raw': concept.get('raw'),
    }
    if provenance.get('source_level'):
        record['source_level'] = provenance['source_level']
    return record


def _record_compatible_absent_assertion(present_fact, absent_fact, compatibility=None):
    absent_concept = absent_fact.get('concept') or {}
    absent_provenance = absent_fact.get('provenance') or {}
    assertion = {
        'raw': absent_concept.get('raw'),
        'qualifier': _directional_qualifier(absent_concept.get('raw')),
        'source': absent_provenance.get('source'),
        'source_index': absent_provenance.get('source_index'),
    }
    if compatibility:
        assertion['compatibility'] = compatibility
    provenance = present_fact.setdefault('provenance', {})
    assertions = provenance.setdefault('compatible_absent_assertions', [])
    if assertion not in assertions:
        assertions.append(assertion)


def _directional_qualifier(raw_name):
    text = re.sub(r'[\s；;，,。]+$', '', _clean(raw_name))
    suffix_groups = (
        ('marked_high', ('明显升高', '显著升高')),
        ('marked_low', ('明显降低', '显著降低')),
        ('high', ('升高', '增高', '偏高')),
        ('low', ('降低', '减低', '偏低')),
        ('positive', ('阳性',)),
        ('negative', ('阴性',)),
        ('abnormal', ('异常',)),
    )
    for qualifier, suffixes in suffix_groups:
        if any(text.endswith(suffix) and len(text) > len(suffix) for suffix in suffixes):
            return qualifier
    return None


def _fact_time(category, domain, canonical, time_index):
    entry = time_index.get((category, canonical))
    if domain == 'symptom':
        return {
            'clinical_onset': entry.get('time_label') if entry else None,
            'observed_at': None,
            'phase': entry.get('phase') if entry else None,
        }
    if entry:
        return {
            'clinical_onset': None,
            'observed_at': entry.get('time_label'),
            'phase': entry.get('phase'),
        }
    return {'clinical_onset': None, 'observed_at': 'D0', 'phase': '就诊'}


def _parse_items(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    return utils.parse_list_from_response(value)


def _phenotype_name(item):
    if isinstance(item, dict):
        return _clean(item.get('name') or item.get('raw') or item.get('concept'))
    if isinstance(item, str):
        return _clean(item)
    if isinstance(item, (list, tuple)) and item:
        if len(item) >= 4 and isinstance(item[-1], (int, float)) and _looks_like_symptom_prefix(item[0]):
            return _clean(item[2])
        return _clean(item[0])
    return ''

def _phenotype_source_level(item):
    if isinstance(item, dict):
        return _clean(item.get('source_level') or item.get('level') or item.get('stage'))
    if isinstance(item, (list, tuple)) and len(item) > 1:
        return _clean(item[1])
    return ''



def _looks_like_symptom_prefix(value):
    text = _clean(value)
    return any(token in text for token in ('天', '周', '月', '年', '小时', '数'))


def _canonical_concept(name):
    text = _clean(name)
    text = re.sub(r'[\s；;，,。]+$', '', text)
    for suffix in ('明显升高', '显著升高', '明显降低', '显著降低', '升高', '增高', '偏高', '降低', '减低', '偏低', '阳性', '阴性', '异常'):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[:-len(suffix)]
    return text


def _interpretation(raw_name, clinical_state):
    if clinical_state == 'absent':
        return 'negative'
    qualifier = _directional_qualifier(raw_name)
    if qualifier in {'marked_high', 'marked_low', 'high', 'low', 'positive', 'negative', 'abnormal'}:
        return qualifier
    return None


def _acquisition_state(domain):
    if domain == 'symptom':
        return 'historical'
    return 'latent'


def _visibility(domain):
    if domain == 'symptom':
        return ['patient', 'diagnostic_oracle']
    if domain == 'sign':
        return ['physical_exam']
    return ['diagnostic_oracle', 'doctor_after_order']


def _parse_value(raw_value, raw_range=None):
    number, unit = _parse_number_unit(raw_value)
    reference_range = _parse_range(raw_range)
    if unit is None and isinstance(reference_range, dict) and reference_range.get('unit'):
        unit = reference_range.get('unit')
    return {
        'number': number,
        'unit': unit,
        'reference_range': reference_range,
    }


def _parse_number_unit(text):
    text = _clean(text)
    actual_match = re.search(r'实际值\s*=\s*(.+)$', text)
    if actual_match:
        text = _clean(actual_match.group(1))
    match = _NUMERIC_RE.search(text)
    if not match:
        return None, text or None
    number = _normalize_number(float(match.group(0)))
    unit = text[match.end():].strip() or None
    return number, unit


def _parse_range(text):
    text = _clean(text)
    if not text:
        return None
    brace_match = re.search(r'\{([^{}]+)\}', text)
    if brace_match:
        parts = [part.strip() for part in brace_match.group(1).split(',')]
        if len(parts) != 4:
            raise ValueError(f'invalid quantified range: {text}')
        lower = _range_bound(parts[2])
        upper = _range_bound(parts[3])
        unit = text[brace_match.end():].strip() or None
        return {
            'raw': text,
            'lower': _normalize_number(lower) if lower is not None else None,
            'upper': _normalize_number(upper) if upper is not None else None,
            'unit': unit,
        }
    range_match = re.search(
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*[-~至到]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',
        text,
    )
    numbers = _NUMERIC_RE.findall(text)
    if not numbers:
        return {'raw': text, 'lower': None, 'upper': None, 'unit': None}
    lower = upper = None
    if text.lstrip().startswith(('>', '≥')):
        lower = float(numbers[0])
        last_token = numbers[0]
    elif text.lstrip().startswith(('<', '≤')):
        upper = float(numbers[0])
        last_token = numbers[0]
    elif range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        last_token = range_match.group(2)
    else:
        lower = upper = float(numbers[0])
        last_token = numbers[0]
    unit_pos = text.rfind(last_token) + len(last_token)
    unit = text[unit_pos:].strip() or None
    return {
        'raw': text,
        'lower': _normalize_number(lower) if lower is not None else None,
        'upper': _normalize_number(upper) if upper is not None else None,
        'unit': unit,
    }


def _range_bound(value):
    text = _clean(value).lower()
    if text in {'nan', 'none', 'null', ''}:
        return None
    return float(text)


def _parse_age(value):
    text = _clean(value)
    if not text:
        return None
    number = float(text)
    return int(number) if number.is_integer() else number


def _normalize_number(value):
    return int(value) if float(value).is_integer() else value


def _severity_scheme(staging_system):
    if not isinstance(staging_system, dict):
        return ''
    return _clean(
        staging_system.get('scheme')
        or staging_system.get('name')
        or staging_system.get('system')
        or staging_system.get('staging_system')
    )


def _normalize_laterality(value):
    text = _clean(value)
    if not text or text in {'不适用', '无', '无侧别', 'NA', 'N/A', 'none', 'None'}:
        return 'not_applicable'
    return text


def _validate_ledger_for_sidecar(ledger, require_final_hash=False):
    if not isinstance(ledger, dict):
        raise ValueError('fact ledger must be a dict')
    if ledger.get('schema_version') != LEDGER_SCHEMA_VERSION:
        raise ValueError('fact ledger schema_version mismatch')
    if ledger.get('module') != LEDGER_MODULE:
        raise ValueError('fact ledger module mismatch')
    if ledger.get('status') not in {'converged', 'quarantined'}:
        raise ValueError('fact ledger status must be converged or quarantined')
    if not ledger.get('case_id'):
        raise ValueError('fact ledger case_id missing')
    audit = ledger.get('audit')
    if not isinstance(audit, dict):
        raise ValueError('fact ledger audit missing or invalid')
    if not isinstance(audit.get('rounds'), list):
        raise ValueError('fact ledger audit.rounds missing or invalid')
    if require_final_hash and not audit.get('final_hash'):
        raise ValueError('fact ledger audit.final_hash missing')
    provenance = ledger.get('provenance')
    if not isinstance(provenance, dict) or not provenance.get('source_row_hash'):
        raise ValueError('fact ledger provenance.source_row_hash missing')
    if not isinstance(ledger.get('facts'), list):
        raise ValueError('fact ledger facts missing or invalid')


def _update_manifest(csv_path, row_index, ledger, summary):
    manifest_path = os.path.join(utils._module_io_dir(csv_path), MANIFEST_FILENAME)
    lock = _manifest_lock(manifest_path)
    with lock:
        lock_path = f'{manifest_path}.lock'
        with open(lock_path, 'a+', encoding='utf-8') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                records = []
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            records.append(json.loads(line))
                record = {
                    'row_index': row_index,
                    'case_id': ledger.get('case_id'),
                    'path': summary['path'],
                    'input_hash': summary['input_hash'],
                    'ledger_hash': summary['ledger_hash'],
                    'status': summary['status'],
                    'round_count': summary['round_count'],
                }
                replaced = False
                for i, existing in enumerate(records):
                    if existing.get('row_index') == row_index:
                        records[i] = record
                        replaced = True
                        break
                if not replaced:
                    records.append(record)
                payload = ''.join(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n' for item in records)
                tmp_path = f'{manifest_path}.tmp.{os.getpid()}.{threading.get_ident()}'
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(payload)
                os.replace(tmp_path, manifest_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _manifest_lock(path):
    abs_path = os.path.abspath(path)
    with _MANIFEST_LOCKS_GUARD:
        lock = _MANIFEST_LOCKS.get(abs_path)
        if lock is None:
            lock = threading.Lock()
            _MANIFEST_LOCKS[abs_path] = lock
        return lock


def _clean(value):
    if value is None:
        return ''
    return str(value).strip()
