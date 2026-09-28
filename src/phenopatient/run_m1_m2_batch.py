#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the 50-disease DemoDx input through PhenoPatient M1 and M2 only.

The default first pass stops after one department has produced at least 100
complete M2 patients, leaving a stable audit window.  Re-run the same command
with ``--continue_after_audit`` to finish every prepared M2 CSV.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from atlas_based_patient_generation import (
    _is_m2_row_complete,
    _validate_m1_row_for_m2,
    run_module2,
)
from phenotypic_atlas import process_seed_module1
from patient_fact_ledger import (
    COL_M281_OUTPUT, canonical_ledger_hash, ledger_input_hash, _time_model_from_row,
)
from run_phenopatient import (
    _sync_csv_template_and_sidecar,
    _inject_patient_rows,
    _normalize_rows,
    _read_rows,
    _safe_name,
    _seed_text_for_cases,
)


DEMODX_VISIT_ACUITY_POLICY_VERSION = 'demodx_visit_acuity_v1_20260913'
DEMODX_VISIT_ACUITY_BY_DISEASE_ID = {
    'acute_coronary_syndrome_mi': '急性',
    'heart_failure': '慢性急性加重',
    'atrial_fibrillation': '慢性急性加重',
    'pulmonary_embolism': '急性',
    'coronary_artery_disease_unstable_angina': '急性',
    'aortic_valve_stenosis': '慢性',
    'abdominal_aortic_aneurysm': '慢性',
    'aortic_dissection': '急性',
    'infective_endocarditis': '急性',
    'cardiomyopathy': '慢性',
    'pneumonia': '急性',
    'aspiration_pneumonia': '急性',
    'copd_exacerbation': '慢性急性加重',
    'asthma_exacerbation': '慢性急性加重',
    'acute_respiratory_failure': '急性',
    'pneumothorax': '急性',
    'pleural_effusion_empyema': '急性',
    'interstitial_lung_disease': '慢性',
    'sarcoidosis': '慢性',
    'bronchiectasis': '慢性急性加重',
    'acute_pancreatitis': '急性',
    'acute_appendicitis': '急性',
    'diverticulitis': '急性',
    'acute_cholecystitis': '急性',
    'gastrointestinal_bleeding': '急性',
    'bowel_obstruction': '急性',
    'cirrhosis_decompensated_liver_disease': '慢性急性加重',
    'cholangitis': '急性',
    'crohn_disease': '慢性急性加重',
    'ulcerative_colitis': '慢性急性加重',
    'acute_kidney_injury': '急性',
    'urinary_tract_infection': '急性',
    'pyelonephritis': '急性',
    'nephrolithiasis_ureteral_stone': '急性',
    'hydronephrosis_obstructive_uropathy': '急性',
    'hematuria': '急性',
    'end_stage_renal_disease': '慢性',
    'glomerulonephritis': '急性',
    'renal_transplant_complication': '急性',
    'acute_tubular_necrosis': '急性',
    'ischemic_stroke': '急性',
    'intracerebral_hemorrhage': '急性',
    'transient_ischemic_attack': '急性',
    'seizure_epilepsy': '急性',
    'subdural_hemorrhage': '急性',
    'subarachnoid_hemorrhage': '急性',
    'encephalopathy': '急性',
    'multiple_sclerosis': '慢性急性加重',
    'myasthenia_gravis': '慢性急性加重',
    'guillain_barre': '急性',
}
_VALID_VISIT_ACUITIES = {'急性', '慢性', '慢性急性加重'}


def _apply_demodx_acuity_policy(rows, explicit_default=''):
    """Resolve missing visit-scenario acuity without silently calling all cases acute."""
    default = str(explicit_default or '').strip()
    if default and default not in _VALID_VISIT_ACUITIES:
        raise ValueError(f'无法识别显式默认 acuity: {explicit_default!r}')
    resolved = []
    for index, source in enumerate(rows, start=1):
        if not isinstance(source, dict):
            raise ValueError(f'第 {index} 行不是对象/dict')
        row = dict(source)
        explicit = str(row.get('acuity') or row.get('急慢性') or '').strip()
        if explicit:
            if explicit not in _VALID_VISIT_ACUITIES:
                raise ValueError(f'第 {index} 行 acuity 非法: {explicit!r}')
            row['acuity'] = explicit
        else:
            disease_id = str(row.get('disease_id') or '').strip()
            inferred = DEMODX_VISIT_ACUITY_BY_DISEASE_ID.get(disease_id) or default
            if not inferred:
                raise ValueError(
                    f'第 {index} 行缺少 acuity，且 disease_id={disease_id!r} '
                    f'未出现在 {DEMODX_VISIT_ACUITY_POLICY_VERSION} 映射中'
                )
            if inferred not in _VALID_VISIT_ACUITIES:
                raise ValueError(
                    f'{DEMODX_VISIT_ACUITY_POLICY_VERSION} 中 disease_id={disease_id!r} '
                    f'的 acuity 非法: {inferred!r}'
                )
            row['acuity'] = inferred
        resolved.append(row)
    return resolved


def _validate_demodx_disease_ids(rows, expected_per_disease=1):
    """Require this dedicated runner to receive exactly its versioned disease set."""
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('DemoDx 输入必须全部为对象/dict')
    disease_ids = [str(row.get('disease_id') or '').strip() for row in rows]
    actual = set(disease_ids)
    expected = set(DEMODX_VISIT_ACUITY_BY_DISEASE_ID)
    if actual != expected:
        raise ValueError(
            'DemoDx disease_id 集合与版本化映射不一致: '
            f'missing={sorted(expected - actual)!r}, '
            f'unexpected={sorted(actual - expected)!r}'
        )
    counts = Counter(disease_ids)
    bad_counts = {
        disease_id: count for disease_id, count in counts.items()
        if count != expected_per_disease
    }
    if bad_counts:
        raise ValueError(
            f'DemoDx 每病种病例数应为 {expected_per_disease}: {bad_counts!r}'
        )
    identities = {}
    for row in rows:
        disease_id = str(row.get('disease_id') or '').strip()
        identity = (
            str(row.get('diagnosis') or row.get('diagnosis_zh') or '').strip(),
            str(row.get('organ_system') or row.get('department') or '').strip(),
        )
        identities.setdefault(disease_id, set()).add(identity)
    inconsistent = {
        disease_id: sorted(values)
        for disease_id, values in identities.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f'DemoDx disease_id 的诊断/科室不唯一: {inconsistent!r}')


def _group_cases(cases):
    groups = OrderedDict()
    for case in cases:
        key = (case['department'], case['diagnosis'], case['acuity'])
        groups.setdefault(key, []).append(case)
    return groups


def _validate_case_set(cases, groups, expected_cases, expected_groups, per_group):
    case_ids = [case['case_id'] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('输入包含重复 case_id')
    if len(cases) != expected_cases:
        raise ValueError(f'病例数应为 {expected_cases}，实际为 {len(cases)}')
    if len(groups) != expected_groups:
        raise ValueError(f'疾病组数应为 {expected_groups}，实际为 {len(groups)}')
    bad = {key: len(value) for key, value in groups.items() if len(value) != per_group}
    if bad:
        raise ValueError(f'每组病例数应为 {per_group}，异常组: {bad}')


def _pilot_departments(groups, per_group, minimum_cases=100):
    departments = []
    seen = set()
    group_count = 0
    for department, _diagnosis, _acuity in groups:
        if department in seen:
            continue
        seen.add(department)
        departments.append(department)
        group_count += sum(1 for key in groups if key[0] == department)
        if group_count * per_group >= minimum_cases:
            return departments
    raise ValueError(f'pilot 全部科室合计不足 {minimum_cases} 例')


def _validate_resume_status(status_path, input_sha256, expected_cases,
                            expected_groups, per_group, default_acuity,
                            guarded_dirs=(),
                            acuity_policy_version=DEMODX_VISIT_ACUITY_POLICY_VERSION):
    path = Path(status_path)
    if not path.exists():
        leftovers = [
            str(root) for root in guarded_dirs
            if Path(root).exists() and any(Path(root).iterdir())
        ]
        if leftovers:
            raise ValueError(
                f'输出目录已有产物但没有 status.json，拒绝接管: {leftovers}'
            )
        return None
    try:
        previous = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'无法验证旧运行状态: {exc}') from exc
    expected = {
        'input_sha256': input_sha256,
        'total_cases': expected_cases,
        'total_groups': expected_groups,
        'patients_per_group': per_group,
        'default_acuity_for_missing_input': default_acuity,
        'acuity_policy_version': acuity_policy_version,
    }
    actual = {key: previous.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f'输入或分组配置已变更，拒绝覆盖旧 M2 进度: '
            f'expected={expected}, previous={actual}'
        )
    return previous


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}.{threading.get_ident()}')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def _m1_csv_path(m1_dir, group_key):
    department, diagnosis, acuity = group_key
    return os.path.join(
        m1_dir, _safe_name(department), _safe_name(acuity),
        f'{_safe_name(diagnosis)}.csv',
    )


def _m2_csv_path(m2_dir, group_key):
    department, diagnosis, acuity = group_key
    return os.path.join(
        m2_dir, _safe_name(department), _safe_name(acuity),
        f'{_safe_name(diagnosis)}.csv',
    )


def _sync_m1_csv_to_m2(src, dst):
    return _sync_csv_template_and_sidecar(src, dst)


def _validate_output_manifest(m2_dir, groups, allow_missing=False):
    expected = {
        Path(_m2_csv_path(m2_dir, group_key)).resolve()
        for group_key in groups
    }
    actual = {
        path.resolve() for path in Path(m2_dir).rglob('*.csv')
    } if Path(m2_dir).exists() else set()
    missing = expected - actual
    unexpected = actual - expected
    if unexpected or (missing and not allow_missing):
        raise ValueError(
            'M2 CSV manifest 与当前输入不一致: '
            f'missing={sorted(map(str, missing))!r}, '
            f'unexpected={sorted(map(str, unexpected))!r}'
        )


def _m281_status_for_row(csv_path, row_index, row, staging_system):
    try:
        row_payload = utils._read_json_file(
            utils._module_io_abs(str(csv_path), row_index), default=None
        )
        if not isinstance(row_payload, dict):
            return 'failed'
        summary = row_payload.get(COL_M281_OUTPUT)
        if not isinstance(summary, dict):
            return 'failed'
        status = summary.get('status')
        if status not in {'converged', 'quarantined'}:
            return 'failed'
        rel_path = summary.get('path')
        if not isinstance(rel_path, str) or os.path.basename(rel_path) != rel_path:
            return 'failed'
        ledger_path = os.path.join(utils._module_io_dir(str(csv_path)), rel_path)
        ledger = utils._read_json_file(ledger_path, default=None)
        if not isinstance(ledger, dict):
            return 'failed'
        if ledger.get('schema_version') != 'fact_ledger.v1' or ledger.get('module') != '2.81':
            return 'failed'
        if ledger.get('status') != status:
            return 'failed'
        case_id = str(row.get(utils.COL_CASE_ID, '') or '').strip()
        if case_id and ledger.get('case_id') != case_id:
            return 'failed'
        actual_hash = canonical_ledger_hash(ledger)
        if summary.get('ledger_hash') != actual_hash:
            return 'failed'
        if ledger.get('audit', {}).get('final_hash') != actual_hash:
            return 'failed'
        input_hash = ledger.get('provenance', {}).get('source_row_hash')
        if summary.get('input_hash') != input_hash:
            return 'failed'
        expected_input_hash = ledger_input_hash(
            row, staging_system, _time_model_from_row(row)
        )
        if input_hash != expected_input_hash:
            return 'failed'
        return 'completed' if status == 'converged' else 'quarantined'
    except Exception:
        return 'failed'


def _count_complete_m2(m2_dir, per_group):
    files = sorted(Path(m2_dir).rglob('*.csv')) if Path(m2_dir).exists() else []
    complete = 0
    total_rows = 0
    case_ids = []
    m281_counts = {'completed': 0, 'failed': 0, 'quarantined': 0}
    for csv_path in files:
        row_1, patients = utils._load_existing_csv(str(csv_path))
        staging_system = utils.parse_staging_system(
            (row_1 or {}).get(utils.COL_STAGING_SYSTEM, '')
        )
        for patient_idx in range(1, per_group + 1):
            row = patients.get(patient_idx)
            if not row:
                continue
            total_rows += 1
            case_id = str(row.get(utils.COL_CASE_ID, '') or '').strip()
            if case_id:
                case_ids.append(case_id)
            if _is_m2_row_complete(
                    row, staging_system=staging_system,
                    csv_path=str(csv_path), row_index=patient_idx):
                complete += 1
            m281_status = _m281_status_for_row(
                csv_path, patient_idx, row, staging_system
            )
            m281_counts[m281_status] += 1
    return {
        'm2_csv_files': len(files),
        'm2_patient_rows': total_rows,
        'm2_completed': complete,
        'm2_unique_case_ids': len(set(case_ids)),
        'm281_completed': m281_counts['completed'],
        'm281_failed': m281_counts['failed'],
        'm281_quarantined': m281_counts['quarantined'],
    }


def _write_seed_file(path, groups):
    with Path(path).open('w', encoding='utf-8') as handle:
        current_department = None
        for (department, diagnosis, _acuity), group_cases in groups.items():
            if department != current_department:
                handle.write(f'# {department}\n')
                current_department = department
            handle.write(_seed_text_for_cases(diagnosis, group_cases) + '\n')


def _run_m1_group(group_key, group_cases, m1_dir):
    department, diagnosis, acuity = group_key
    output_dir = os.path.dirname(_m1_csv_path(m1_dir, group_key))
    os.makedirs(output_dir, exist_ok=True)
    seed_text = _seed_text_for_cases(diagnosis, group_cases)
    result = process_seed_module1(seed_text, output_dir, enable_evidence=False)
    if not result or result.get('status') != 'success':
        raise RuntimeError(f'M1 失败: {group_key}: {result}')
    csv_path = _m1_csv_path(m1_dir, group_key)
    row_1, _patients = utils._load_existing_csv(csv_path)
    if row_1 is None:
        raise RuntimeError(f'M1 CSV 缺失: {csv_path}')
    _validate_m1_row_for_m2(row_1)
    return {
        'department': department,
        'diagnosis': diagnosis,
        'acuity': acuity,
        'status': 'success',
        'file': csv_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='DemoDx 1000 例的 PhenoPatient M1/M2 专用编排器')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--expected_cases', type=int, default=1000)
    parser.add_argument('--expected_groups', type=int, default=50)
    parser.add_argument('--patients_per_group', type=int, default=20)
    parser.add_argument(
        '--default_acuity', default='', choices=['', *_VALID_VISIT_ACUITIES],
        help='仅在 disease_id 无映射且输入缺少 acuity 时显式启用',
    )
    parser.add_argument('--continue_after_audit', action='store_true')
    args = parser.parse_args(argv)

    if utils.DEFAULT_MODEL_NAME != 'GPT-5.5' or utils._GPT54_MODEL_NAME != 'GPT-5.5':
        raise SystemExit(
            f'拒绝启动：当前模型为 {utils.DEFAULT_MODEL_NAME!r}/'
            f'{utils._GPT54_MODEL_NAME!r}，必须同时为 GPT-5.5'
        )
    if not utils.DEFAULT_API_KEY:
        raise SystemExit('拒绝启动：PHENOPATIENT_API_KEY 未设置')
    if args.workers < 1:
        raise SystemExit('--workers 必须 >= 1')

    run_root = os.path.abspath(args.run_root)
    m1_dir = os.path.join(run_root, 'M1')
    m2_dir = os.path.join(run_root, 'M2')
    status_path = os.path.join(run_root, 'status.json')
    os.makedirs(m1_dir, exist_ok=True)
    os.makedirs(m2_dir, exist_ok=True)

    input_rows = _read_rows(args.input_file)
    _validate_demodx_disease_ids(
        input_rows, expected_per_disease=args.patients_per_group
    )
    raw_rows = _apply_demodx_acuity_policy(
        input_rows, explicit_default=args.default_acuity
    )
    cases = _normalize_rows(raw_rows, '完全虚拟患者', '')
    groups = _group_cases(cases)
    _validate_case_set(
        cases, groups, args.expected_cases, args.expected_groups,
        args.patients_per_group,
    )

    input_sha256 = hashlib.sha256(Path(args.input_file).read_bytes()).hexdigest()
    previous_status = _validate_resume_status(
        status_path,
        input_sha256=input_sha256,
        expected_cases=args.expected_cases,
        expected_groups=args.expected_groups,
        per_group=args.patients_per_group,
        default_acuity=args.default_acuity,
        guarded_dirs=(m1_dir, m2_dir),
    )
    _validate_output_manifest(m2_dir, groups, allow_missing=True)
    normalized_path = os.path.join(run_root, 'normalized_cases.jsonl')
    seed_path = os.path.join(run_root, 'demodx_m1_m2.seeds.txt')
    _write_jsonl(normalized_path, cases)
    _write_seed_file(seed_path, groups)

    status = {
        'status': 'running',
        'phase': 'M1',
        'started_at': ((previous_status or {}).get('started_at')
                       or time.strftime('%Y-%m-%d %H:%M:%S')),
        'input_file': os.path.abspath(args.input_file),
        'input_sha256': input_sha256,
        'workers': args.workers,
        'model': utils.DEFAULT_MODEL_NAME,
        'atlas_model': utils._GPT54_MODEL_NAME,
        'evidence_enabled': False,
        'default_acuity_for_missing_input': args.default_acuity,
        'acuity_policy_version': DEMODX_VISIT_ACUITY_POLICY_VERSION,
        'acuity_distribution': {
            acuity: sum(case['acuity'] == acuity for case in cases)
            for acuity in sorted(_VALID_VISIT_ACUITIES)
        },
        'total_cases': len(cases),
        'total_groups': len(groups),
        'patients_per_group': args.patients_per_group,
        'm1_completed': 0,
        'm1_failed': 0,
        'm281_completed': 0,
        'm281_failed': 0,
        'm281_quarantined': 0,
        'm1_dir': m1_dir,
        'm2_dir': m2_dir,
        'normalized_input': normalized_path,
        'generated_seed_file': seed_path,
    }
    _atomic_write_json(status_path, status)
    print(f'RUN_ROOT={run_root}')
    print(f'INPUT_ROWS={len(cases)} GROUPS={len(groups)} WORKERS={args.workers}')
    print(f'EVIDENCE=False MODEL={utils.DEFAULT_MODEL_NAME}')

    m1_results = []
    m1_errors = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(groups))) as pool:
        future_to_key = {
            pool.submit(_run_m1_group, key, group_cases, m1_dir): key
            for key, group_cases in groups.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                m1_results.append(result)
                status['m1_completed'] = len(m1_results)
                print(f'[M1 {len(m1_results)}/{len(groups)}] {key[1]} 完成')
            except Exception as exc:
                error = {'group': list(key), 'error': str(exc)}
                m1_errors.append(error)
                status['m1_failed'] = len(m1_errors)
                print(f'[M1 ERROR] {key}: {exc}')
            status['m1_results'] = m1_results
            status['m1_errors'] = m1_errors
            _atomic_write_json(status_path, status)

    if m1_errors or len(m1_results) != len(groups):
        status.update({'status': 'error', 'phase': 'M1_failed'})
        _atomic_write_json(status_path, status)
        raise SystemExit(1)

    status['phase'] = 'M2_prepare'
    _atomic_write_json(status_path, status)
    for key, group_cases in groups.items():
        src = _m1_csv_path(m1_dir, key)
        dst = _m2_csv_path(m2_dir, key)
        _sync_m1_csv_to_m2(src, dst)
        _inject_patient_rows(dst, _seed_text_for_cases(key[1], group_cases), group_cases)
    try:
        _validate_output_manifest(m2_dir, groups)
    except ValueError as exc:
        status.update({
            'status': 'error',
            'phase': 'M2_manifest_failed',
            'error': str(exc),
        })
        _atomic_write_json(status_path, status)
        raise

    if args.continue_after_audit:
        m2_target = m2_dir
        status['phase'] = 'M2_full'
        _atomic_write_json(status_path, status)
        result = run_module2(
            csv_dir=m2_target,
            num_patients=args.patients_per_group,
            num_workers=args.workers,
        )
    else:
        status['phase'] = 'M2_pilot'
        pilot_departments = _pilot_departments(
            groups, args.patients_per_group, minimum_cases=100
        )
        status['pilot_departments'] = pilot_departments
        _atomic_write_json(status_path, status)
        pilot_results = []
        for department in pilot_departments:
            result_one = run_module2(
                csv_dir=os.path.join(m2_dir, _safe_name(department)),
                num_patients=args.patients_per_group,
                num_workers=args.workers,
            )
            pilot_results.append({'department': department, 'result': result_one})
            status.update(_count_complete_m2(m2_dir, args.patients_per_group))
            status['m2_pilot_results'] = pilot_results
            _atomic_write_json(status_path, status)
            if not result_one or result_one.get('status') != 'success':
                break
            if status['m2_completed'] >= 100:
                break
        result = {
            'status': ('success' if pilot_results and all(
                item['result'] and item['result'].get('status') == 'success'
                for item in pilot_results
            ) else 'error'),
            'departments': pilot_results,
        }
    status['m2_result'] = result
    status.update(_count_complete_m2(m2_dir, args.patients_per_group))
    status['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    if not result or result.get('status') != 'success':
        status.update({'status': 'error', 'phase': 'M2_failed'})
        _atomic_write_json(status_path, status)
        raise SystemExit(1)

    if args.continue_after_audit:
        expected = args.expected_cases
        if (status['m2_completed'] != expected
                or status['m2_unique_case_ids'] != expected):
            status.update({'status': 'error', 'phase': 'M2_incomplete'})
            _atomic_write_json(status_path, status)
            raise SystemExit(1)
        status.update({'status': 'completed', 'phase': 'completed'})
    else:
        if status['m2_completed'] < 100:
            status.update({'status': 'error', 'phase': 'pilot_below_100'})
            _atomic_write_json(status_path, status)
            raise SystemExit(1)
        status.update({'status': 'audit_ready', 'phase': 'awaiting_audit'})

    _atomic_write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
