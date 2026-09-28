#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run PhenoPatient from the same minimal fully-synthetic patient input used by
DemoDx-only style baselines: age, gender/sex, and diagnosis/disease.

Input rows may be CSV, JSON list, or JSONL. Supported aliases:
  - case_id: case_id, id, patient_id, hadm_id
  - age: age, 年龄
  - gender/sex: gender, sex, 性别
  - diagnosis: diagnosis, disease, target_diagnosis, primary_diagnosis, 诊断, 疾病
  - optional: organ_system/system/department/科室, icd_code/icd/ICD, acuity/急慢性, stage/时期
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COL_CASE_ID = 'case_id'
COL_SEED = 'seed'
COL_AGE = '年龄'
COL_GENDER = '性别'
COL_DIAGNOSIS = '诊断'
COL_STAGING_SYSTEM = '分级系统'
COL_STAGE = '时期'
COL_ACUITY = '急慢性'

GENDER_MAP = {
    'm': '男', 'male': '男', 'man': '男', '男': '男', '男性': '男', '1': '男',
    'f': '女', 'female': '女', 'woman': '女', '女': '女', '女性': '女', '0': '女',
}
VALID_ACUITY = {'急性', '慢性', '慢性急性加重'}


def _first(row, names, default=''):
    for name in names:
        if name in row and row[name] not in (None, ''):
            return row[name]
    return default


def _safe_name(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '_', str(text).strip()) or 'unknown'


def _normalize_gender(value):
    key = str(value).strip().lower()
    if key not in GENDER_MAP:
        raise ValueError(f'无法识别性别: {value!r}；请使用 男/女、male/female、M/F')
    return GENDER_MAP[key]


def _normalize_age(value):
    try:
        age = int(float(str(value).strip()))
    except Exception as exc:
        raise ValueError(f'无法识别年龄: {value!r}') from exc
    if age <= 0 or age > 120:
        raise ValueError(f'年龄超出合理范围: {age}')
    return age


def _normalize_acuity(value, default):
    text = str(value or '').strip()
    if not text:
        text = str(default or '').strip()
    if text in VALID_ACUITY:
        return text
    raise ValueError(f'无法识别急慢性类型: {value!r}')


def _read_rows(input_file):
    path = Path(input_file)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if suffix == '.jsonl':
        rows = []
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{input_file}:{line_no} 不是合法 JSONL') from exc
        return rows
    if suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ('patients', 'cases', 'data'):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError('JSON 输入需要是 list，或包含 patients/cases/data list')
    raise ValueError('输入文件后缀只支持 .csv / .jsonl / .json')


def _normalize_rows(rows, default_department, default_acuity):
    normalized = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f'第 {idx} 行不是对象/dict')
        diagnosis = str(_first(row, ['diagnosis', 'disease', 'target_diagnosis', 'primary_diagnosis', '诊断', '疾病'])).strip()
        if not diagnosis:
            raise ValueError(f'第 {idx} 行缺少 diagnosis/disease/诊断')
        age = _normalize_age(_first(row, ['age', '年龄']))
        gender = _normalize_gender(_first(row, ['gender', 'sex', '性别']))
        case_id = str(_first(row, ['case_id', 'id', 'patient_id', 'hadm_id'], f'case_{idx:05d}')).strip()
        department = str(_first(row, ['organ_system', 'system', 'department', '科室'], default_department)).strip() or default_department
        icd_code = str(_first(row, ['icd_code', 'icd', 'ICD'], '')).strip()
        acuity = _normalize_acuity(_first(row, ['acuity', '急慢性'], ''), default_acuity)
        stage = str(_first(row, ['stage', 'severity', '时期'], '')).strip()
        normalized.append({
            'case_id': case_id,
            'age': age,
            'gender': gender,
            'diagnosis': diagnosis,
            'department': department,
            'icd_code': icd_code,
            'acuity': acuity,
            'stage': stage,
        })
    return normalized


def _seed_text_for_cases(diagnosis, cases):
    ages = [case['age'] for case in cases]
    male_ratio = sum(1 for case in cases if case['gender'] == '男') / max(len(cases), 1)
    acuities = [case['acuity'] for case in cases if case.get('acuity')]
    if not acuities or len(set(acuities)) != 1:
        raise ValueError('同一 PhenoPatient 种子组内的急慢性类型必须一致')
    acuity = acuities[0]
    icd = next((case['icd_code'] for case in cases if case.get('icd_code')), '')
    suffix = f' {icd}' if icd else ''
    return f'{diagnosis} # {min(ages)}-{max(ages)} {male_ratio:.2f} {acuity}{suffix}'


def _copy_csv_and_sidecar(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return 'noop'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    src_io = src + '.module_io'
    dst_io = dst + '.module_io'
    if os.path.isdir(dst_io):
        shutil.rmtree(dst_io)
    if os.path.isdir(src_io):
        shutil.copytree(src_io, dst_io)
    return 'copied'


def _canonical_template_value(column, value):
    import ast
    from utils import parse_staging_system

    if column == COL_STAGING_SYSTEM:
        parsed = parse_staging_system(value)
        if parsed:
            return _canonical_jsonable(parsed)
    if isinstance(value, (dict, list, tuple)):
        return _canonical_jsonable(value)
    text = str(value if value is not None else '').strip()
    if not text:
        return ''
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (dict, list, tuple)):
            return _canonical_jsonable(parsed)
    return text


def _canonical_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _canonical_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(item) for item in value]
    return value


def _template_signature(csv_path):
    from utils import CSV_COLUMNS, _load_existing_csv, _read_json_file

    row_1, _patients = _load_existing_csv(csv_path)
    if row_1 is None:
        return None
    sidecar_path = os.path.join(f'{csv_path}.module_io', 'row_0.json')
    sidecar = _read_json_file(sidecar_path, default={})
    if not isinstance(sidecar, dict):
        sidecar = {}
    csv_columns = set(CSV_COLUMNS)
    row_1_semantic = {
        column: _canonical_template_value(column, row_1.get(column, ''))
        for column in CSV_COLUMNS
    }
    extra_sidecar = {
        key: _canonical_template_value(key, sidecar[key])
        for key in sorted(sidecar)
        if key not in csv_columns
    }
    return {
        'row_1': row_1_semantic,
        'row_0_extra_sidecar': extra_sidecar,
    }


def _sync_csv_template_and_sidecar(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return 'noop'
    if not os.path.exists(dst):
        _copy_csv_and_sidecar(src, dst)
        return 'copied'
    if _template_signature(src) == _template_signature(dst):
        return 'unchanged'
    _copy_csv_and_sidecar(src, dst)
    return 'refreshed'


def _preflight_m3_fact_ledgers(csv_path):
    from patient_fact_ledger import load_verified_fact_ledger_for_m3
    from utils import _load_existing_csv, parse_staging_system

    row_1, patients = _load_existing_csv(csv_path)
    if row_1 is None:
        raise RuntimeError(f'M3 fact ledger preflight failed: CSV missing or empty: {csv_path}')
    staging_system = parse_staging_system(row_1.get(COL_STAGING_SYSTEM, ''))
    if not staging_system:
        raise RuntimeError(f'M3 fact ledger preflight failed: staging system missing: {csv_path}')
    if not patients:
        raise RuntimeError(f'M3 fact ledger preflight failed: no patient rows: {csv_path}')

    errors = []
    for row_index, row in sorted(patients.items()):
        case_id = str(row.get(COL_CASE_ID, '') or '').strip() or '<missing case_id>'
        try:
            load_verified_fact_ledger_for_m3(csv_path, row_index, row, staging_system)
        except Exception as exc:
            errors.append(f'row {row_index} case_id={case_id}: {exc}')
    if errors:
        raise RuntimeError(
            f'M3 fact ledger preflight failed for {csv_path}: ' + '; '.join(errors)
        )
    return len(patients)


def _validate_injected_stage(stage, staging_system):
    text = str(stage or '').strip()
    if not text:
        return ''
    levels = staging_system.get('levels', []) if isinstance(staging_system, dict) else []
    level_names = [
        str(level.get('name', '') or '').strip()
        for level in levels
        if isinstance(level, dict) and str(level.get('name', '') or '').strip()
    ]
    if level_names.count(text) != 1:
        raise ValueError(f'外部时期不属于M1分级系统: {text}')
    return text


def _inject_patient_rows(csv_path, seed_text, cases):
    from utils import _empty_row, _load_existing_csv, _save_csv, parse_staging_system

    row_1, existing_patients = _load_existing_csv(csv_path)
    if row_1 is None:
        raise RuntimeError(f'M1 CSV 不存在或为空: {csv_path}')
    staging_system = parse_staging_system(row_1.get(COL_STAGING_SYSTEM, ''))
    if not staging_system:
        raise RuntimeError(f'M1 CSV 缺少有效分级系统: {csv_path}')
    patient_rows = []
    for idx, case in enumerate(cases, start=1):
        row = dict(existing_patients.get(idx) or _empty_row())
        row[COL_CASE_ID] = case['case_id']
        row[COL_SEED] = seed_text
        row[COL_AGE] = str(case['age'])
        row[COL_GENDER] = case['gender']
        row[COL_DIAGNOSIS] = case['diagnosis']
        row[COL_ACUITY] = case['acuity']
        if case.get('stage'):
            row[COL_STAGE] = _validate_injected_stage(case['stage'], staging_system)
        patient_rows.append(row)
    _save_csv(csv_path, [row_1] + patient_rows)


def _write_manifest(output_root, cases, groups, paths):
    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_format': 'DemoDx-style minimal virtual patient input: age, gender, diagnosis',
        'model': os.environ.get('PHENOPATIENT_MODEL_NAME', 'GPT-5.5'),
        'atlas_model': os.environ.get('PHENOPATIENT_ATLAS_MODEL_NAME', os.environ.get('PHENOPATIENT_MODEL_NAME', 'GPT-5.5')),
        'num_cases': len(cases),
        'num_groups': len(groups),
        'paths': paths,
        'cases': cases,
    }
    out = Path(output_root) / 'phenopatient_run_manifest.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(out)


def main():
    parser = argparse.ArgumentParser(description='Run PhenoPatient from age/gender/diagnosis input rows.')
    parser.add_argument('--input_file', required=True, help='CSV/JSON/JSONL with age, gender/sex, diagnosis/disease')
    parser.add_argument('--output_root', required=True, help='Output root directory')
    parser.add_argument('--total_workers', type=int, default=50)
    parser.add_argument('--seed_parallel', type=int, default=5)
    parser.add_argument('--default_department', default='完全虚拟患者')
    parser.add_argument('--default_acuity', default='急性', choices=sorted(VALID_ACUITY))
    parser.add_argument('--enable_evidence', action='store_true', default=False)
    parser.add_argument('--evidence_env', default='yufa')
    parser.add_argument('--max_interaction_turns', type=int, default=None)
    parser.add_argument('--max_followup_turns', type=int, default=None)
    args = parser.parse_args()

    from phenotypic_atlas import process_seed_module1
    from atlas_based_patient_generation import run_module2
    from virtual_clinical_interaction import run_module345
    from check_and_reset_seed import check_and_reset

    rows = _read_rows(args.input_file)
    cases = _normalize_rows(rows, args.default_department, args.default_acuity)
    if not cases:
        raise SystemExit('输入为空')

    run_name = f'PhenoPatient_{time.strftime("%Y%m%d_%H%M%S")}'
    m1_dir = os.path.join(args.output_root, run_name)
    step2_dir = os.path.join(args.output_root, f'{run_name}_step2')
    step3_dir = os.path.join(args.output_root, f'{run_name}_step3')
    step4_dir = os.path.join(args.output_root, f'{run_name}_step4')
    for directory in (m1_dir, step2_dir, step3_dir, step4_dir):
        os.makedirs(directory, exist_ok=True)

    groups = OrderedDict()
    for case in cases:
        key = (case['department'], case['diagnosis'], case['acuity'])
        groups.setdefault(key, []).append(case)

    w_m2 = max(1, args.total_workers // max(1, args.seed_parallel))
    w_method = max(1, args.total_workers // max(1, args.seed_parallel * 2))

    generated_seed_path = os.path.join(args.output_root, f'{run_name}_demodx_input.seeds.txt')
    with open(generated_seed_path, 'w', encoding='utf-8') as seed_file:
        for (department, diagnosis, acuity), group_cases in groups.items():
            seed_file.write(f'# {department}_{acuity}\n')
            seed_file.write(_seed_text_for_cases(diagnosis, group_cases) + '\n')

    print(f'读取病例: {len(cases)}；疾病组: {len(groups)}')
    print(f'输出目录: {args.output_root}')
    print(f'模型: {os.environ.get("PHENOPATIENT_MODEL_NAME", "GPT-5.5")}')

    results = []
    for (department, diagnosis, acuity), group_cases in groups.items():
        seed_text = _seed_text_for_cases(diagnosis, group_cases)
        safe = _safe_name(diagnosis)
        dept_safe = os.path.join(_safe_name(department), _safe_name(acuity))
        print(f'\n=== {department} / {diagnosis} / {acuity}: {len(group_cases)} cases ===')

        dept_m1 = os.path.join(m1_dir, dept_safe)
        dept_step2 = os.path.join(step2_dir, dept_safe)
        dept_step3 = os.path.join(step3_dir, dept_safe)
        dept_step4 = os.path.join(step4_dir, dept_safe)
        for directory in (dept_m1, dept_step2, dept_step3, dept_step4):
            os.makedirs(directory, exist_ok=True)

        m1_csv = os.path.join(dept_m1, f'{safe}.csv')
        step2_csv = os.path.join(dept_step2, f'{safe}.csv')
        step3_csv = os.path.join(dept_step3, f'{safe}.csv')
        step4_csv = os.path.join(dept_step4, f'{safe}.csv')

        result = process_seed_module1(
            seed_text,
            dept_m1,
            enable_evidence=args.enable_evidence,
            evidence_env=args.evidence_env,
        )
        if result.get('status') != 'success':
            results.append({'diagnosis': diagnosis, 'status': 'M1_failed'})
            continue

        _copy_csv_and_sidecar(m1_csv, step2_csv)
        _inject_patient_rows(step2_csv, seed_text, group_cases)
        m2_result = run_module2(
            csv_file=step2_csv,
            num_patients=len(group_cases),
            num_workers=w_m2,
        )
        if not m2_result or m2_result.get('status') != 'success':
            results.append({
                'department': department,
                'diagnosis': diagnosis,
                'acuity': acuity,
                'status': 'M2_failed',
                'detail': m2_result,
            })
            continue

        _copy_csv_and_sidecar(step2_csv, step3_csv)
        check_and_reset(step3_csv)
        _preflight_m3_fact_ledgers(step3_csv)
        run_module345(
            csv_file=step3_csv,
            max_interaction_turns=args.max_interaction_turns,
            max_followup_turns=args.max_followup_turns,
            num_workers=w_method,
        )

        _copy_csv_and_sidecar(step3_csv, step4_csv)
        results.append({
            'department': department,
            'diagnosis': diagnosis,
            'acuity': acuity,
            'cases': len(group_cases),
            'status': 'success',
            'step3_csv': step3_csv,
            'step4_csv': step4_csv,
        })

    manifest_path = _write_manifest(args.output_root, cases, groups, {
        'm1_dir': m1_dir,
        'step2_dir': step2_dir,
        'step3_dir': step3_dir,
        'step4_dir': step4_dir,
        'generated_seed_file': generated_seed_path,
    })
    failed = [item for item in results if item.get('status') != 'success']
    print('\n完成。' if not failed else '\n运行结束，但存在失败分组。')
    print(json.dumps({'manifest': manifest_path, 'results': results}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
