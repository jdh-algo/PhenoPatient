# -*- coding: utf-8 -*-
"""
atlas_based_patient_generation.py
模块2：基于表型概率库生成模拟患者

功能：读取 phenotypic_atlas.py 输出的 CSV（含 row_1 概率库），
为每个种子生成 N 个模拟患者行（row_2..N+1），写入同一 CSV。
患者行包含：采样表型、关联校正、时间排序、伴随疾病、数值量化、具体数值化、主诉。

输出：更新同一 {csv_dir}/{seed_name}.csv（在 row_1 之后追加患者行）

上游：phenotypic_atlas.py（须先运行，生成含 row_1 的 CSV）
下游：virtual_clinical_interaction.py 读取本脚本填充的患者行
"""

import sys
import os
import ast
import hashlib
import json
import math
import re
import time
import random
import argparse
import traceback
import queue as _queue_module
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DEFAULT_NUM_PATIENTS_PER_SEED, DEFAULT_NUM_SEED_WORKERS,
    DEFAULT_CORRELATION_CO_OCCUR_PROB,
    call_gpt54 as call_gpt5, _retry_module_call,
    parse_list_from_response, parse_staging_system,
    _get_num_levels, _get_level_names,
    _parse_seed, _quantize_values, _derive_chief_complaint,
    COL_CASE_ID, COL_SEED, COL_STAGING_SYSTEM, COL_STAGE,
    COL_SYMPTOMS, COL_SIGNS, COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS,
    COL_TIME_ORDER, COL_COMORBIDITIES,
    COL_QUANTIFIED, COL_SPECIFIC, COL_CHIEF_COMPLAINT,
    COL_AGE, COL_GENDER, COL_DIAGNOSIS,
    COL_M24_INPUT, COL_M24_OUTPUT, COL_M25_INPUT, COL_M25_OUTPUT,
    COL_M27_INPUT, COL_M27_OUTPUT,
    COL_LATERALITY_TYPE, COL_PATIENT_LATERALITY,
    COL_COMPLICATION_PHENOTYPES,
    COL_ACUITY, COL_PRIOR_VISITED, COL_PRIOR_VISIT_COUNT,
    COL_DURATION_TOTAL, COL_PRIOR_VISIT_HISTORY,
    COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS, COL_ABSENT_LAB_TESTS,
    COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL,
    COL_M23_INPUT, COL_M23_OUTPUT, COL_M26_INPUT, COL_M26_OUTPUT,
    ACUITY_ACUTE, ACUITY_CHRONIC, ACUITY_CHRONIC_EXACERBATION,
    _empty_row, _load_existing_csv, _save_csv,
    fixed_anatomic_laterality,
)
from phenotypic_atlas import (
    validate_m1_flat_probability_schema,
    validate_m1_symptom_schema,
    validate_m1_staging_system,
)
from patient_fact_ledger import (
    COL_M281_OUTPUT, PENDING_LEDGER_ROW_KEY,
    ledger_input_hash, load_fact_ledger,
    module_2_81_finalize_fact_ledger, write_fact_ledger,
)


M2_QUANTIFICATION_BATCH_SIZE = 20
M2_QUANTIFICATION_MAX_RETRIES = 5
M2_CORRELATION_MAX_RETRIES = 5
M2_TIMELINE_MAX_RETRIES = 5
M2_PRIOR_HISTORY_MAX_RETRIES = 3


_MALE_ONLY_PHENOTYPE_TERMS = (
    '男性', '男童', '前列腺', 'PSA', '前列腺特异抗原',
    '阴茎', '阴囊', '睾丸', '精索', '精液', '射精', '勃起',
)
_FEMALE_ONLY_PHENOTYPE_TERMS = (
    '女性', '女童', '妊娠', '孕妇', '产后', '月经', '绝经',
    '子宫', '宫颈', '卵巢', '阴道', '外阴', '妇科', '宫内孕', '胎心',
)
_MALE_THEN_FEMALE_THRESHOLD_RE = re.compile(
    r'男性\s*([^（）()]+?)\s*(?:或|和|及|、|，|,|/|／)\s*'
    r'女性\s*([^（）()]+)'
)
_FEMALE_THEN_MALE_THRESHOLD_RE = re.compile(
    r'女性\s*([^（）()]+?)\s*(?:或|和|及|、|，|,|/|／)\s*'
    r'男性\s*([^（）()]+)'
)


def _personalize_phenotype_name_for_gender(name, patient_gender):
    """Drop clearly inapplicable sex-specific phenotypes and localize dual cutoffs."""
    text = str(name or '').strip()
    gender = str(patient_gender or '').strip()
    if not text or gender not in ('男', '女'):
        return text

    def _male_then_female(match):
        if gender == '男':
            return f'男性{match.group(1).strip()}'
        return f'女性{match.group(2).strip()}'

    def _female_then_male(match):
        if gender == '女':
            return f'女性{match.group(1).strip()}'
        return f'男性{match.group(2).strip()}'

    text = _MALE_THEN_FEMALE_THRESHOLD_RE.sub(_male_then_female, text)
    text = _FEMALE_THEN_MALE_THRESHOLD_RE.sub(_female_then_male, text)
    applicability_text = text.replace('女性化', '').replace('男性化', '')
    folded = applicability_text.upper()
    has_male = any(term.upper() in folded for term in _MALE_ONLY_PHENOTYPE_TERMS)
    has_female = any(term.upper() in folded for term in _FEMALE_ONLY_PHENOTYPE_TERMS)
    if gender == '男' and has_female and not has_male:
        return None
    if gender == '女' and has_male and not has_female:
        return None
    return text


def _personalize_phenotype_items_for_gender(items, patient_gender):
    result = []
    for item in items or []:
        if isinstance(item, (list, tuple)) and item:
            name = _personalize_phenotype_name_for_gender(item[0], patient_gender)
            if name is not None:
                result.append((name, *item[1:]))
        else:
            name = _personalize_phenotype_name_for_gender(item, patient_gender)
            if name is not None:
                result.append(name)
    return result


def _validate_m1_row_for_m2(row):
    """Validate and canonicalize all M1 phenotype libraries before patient sampling."""
    try:
        staging_system = validate_m1_staging_system(row.get(COL_STAGING_SYSTEM, ''))
    except ValueError as exc:
        raise ValueError(f'模块1.0 严重程度分级系统无效: {exc}') from exc
    row[COL_SYMPTOMS] = validate_m1_symptom_schema(
        row.get(COL_SYMPTOMS, ''), staging_system
    )
    if not parse_list_from_response(row[COL_SYMPTOMS]):
        raise ValueError('模块1.11 症状库为空')
    for column, module_name in (
        (COL_SIGNS, '模块1.12'),
        (COL_LAB_TESTS, '模块1.21'),
        (COL_IMAGING, '模块1.22'),
        (COL_FUNCTIONAL_TESTS, '模块1.23'),
    ):
        row[column] = validate_m1_flat_probability_schema(
            row.get(column, ''), staging_system, module_name=module_name
        )
    return staging_system


def _validate_patient_stage_name(stage, level_names):
    text = str(stage or '').strip()
    if not text:
        raise ValueError('患者时期为空')
    matches = [name for name in level_names if name == text]
    if len(matches) != 1:
        raise ValueError(f'患者时期不属于M1分级系统: {text}')
    return text


# ============================================================
# 关联性校正规则表
# ============================================================

CORRELATION_RULES = [
    ('体温升高', '心动过速', 'co_occur', 0.85),
    ('体温升高', '寒战', 'co_occur', 0.70),
    ('呼吸困难', '血氧饱和度下降', 'co_occur', 0.80),
    ('呼吸困难', '呼吸频率增快', 'co_occur', 0.85),
    ('胸痛', '心动过速', 'co_occur', 0.60),
    ('咳嗽', '咳痰', 'co_occur', 0.70),
    ('腹泻', '脱水', 'co_occur', 0.65),
    ('发热', '体温升高', 'co_occur', 0.95),
    ('SpO2下降', '发绀', 'co_occur', 0.70),
    ('心动过速', '心动过缓', 'exclusive', 0.0),
    ('体温升高', '体温降低', 'exclusive', 0.0),
    ('高血压', '低血压', 'exclusive', 0.0),
]

# 以下文本将规则注入到 GPT 提示词供 LLM 参考（不再编程应用）
_CORRELATION_RULES_TEXT = """已知的临床关联规则（供审查参考）：
共同出现规则（若A出现，则B有较高概率同时存在）：
- 体温升高 → 心动过速（概率0.85）；体温升高 → 寒战（概率0.70）
- 呼吸困难 → 血氧饱和度下降（概率0.80）；呼吸困难 → 呼吸频率增快（概率0.85）
- 胸痛 → 心动过速（概率0.60）；咳嗽 → 咳痰（概率0.70）
- 腹泻 → 脱水（概率0.65）；发热 → 体温升高（概率0.95）；SpO2下降 → 发绀（概率0.70）
互斥规则（以下对组不得同时存在，若存在请保留更符合当前分期的一个）：
- 心动过速 与 心动过缓；体温升高 与 体温降低；高血压 与 低血压
反向排除规则（若 A 在阳性列表且 B 在已排除列表，以下组合生理上不可共存，须从阳性列表删除 A）：
- 发绀/口唇发绀 阳性 + SpO2正常/SpO2无下降 排除 → 发绀不可信，删除发绀
- 颈静脉怒张 阳性 + 中心静脉压正常 排除 → 矛盾，删除颈静脉怒张
- 大量胸腔积液（影像）阳性 + 叩诊清音（体征）排除 → 矛盾，删除大量胸腔积液
- 严重贫血貌 阳性 + 血红蛋白正常 排除 → 矛盾，删除严重贫血貌
- 高热（>39℃）阳性 + WBC正常且CRP正常 排除 → 严重度矛盾，需降低体温数值或删除高热
- 库斯莫尔呼吸必须有直接代谢性酸中毒证据（血pH降低、碳酸氢根降低或明确酸血症）；仅有深大呼吸、呼吸频率增快、阴离子间隙升高、乳酸升高或酮体阳性均不足以推出库斯莫尔呼吸。若血pH降低与碳酸氢根降低均在阴性列表，库斯莫尔呼吸必须保持阴性。
- Kussmaul征（吸气时JVP升高）是不同的心血管体征，不受上述库斯莫尔呼吸规则影响。
"""


# ============================================================
# 模块2.0 / 2.1
# ============================================================

def module_2_0_assign_stages(num_patients, staging_system):
    """
    模块2.0 时期分配（无GPT调用，纯本地计算）
    根据 staging_system 中各级别的 proportion 比例分配患者。
    """
    levels = staging_system.get('levels', [])
    if not levels:
        return ['早期'] * num_patients

    proportions = [lvl.get('proportion', 0) for lvl in levels]
    total_prop = sum(proportions)
    if total_prop <= 0:
        proportions = [1.0 / len(levels)] * len(levels)
        total_prop = 1.0

    counts = []
    remaining = num_patients
    for i, prop in enumerate(proportions):
        if i == len(proportions) - 1:
            counts.append(remaining)
        else:
            n = round(num_patients * prop / total_prop)
            n = min(n, remaining)
            counts.append(n)
            remaining -= n

    stages = []
    for lvl, cnt in zip(levels, counts):
        stages.extend([lvl['name']] * cnt)

    random.shuffle(stages)
    return stages


def module_2_1_sample_phenotypes(symptoms_text, signs_text, lab_tests_text,
                                  imaging_text, functional_tests_text,
                                  patient_level_index, num_levels,
                                  patient_stage='早期', patient_gender=''):
    """
    模块2.1 五库伯努利采样（无GPT调用）。

    v4 双轨改造：在 selected_* 之外，同时返回 absent_*（伯努利未中的表型名列表，
    仅名称字符串，供下游 prompt 使用）。
    返回 10 元组：
        (selected_symptoms, selected_signs, selected_lab_tests,
         selected_imaging, selected_functional,
         absent_symptoms, absent_signs, absent_lab_tests,
         absent_imaging, absent_functional)
    """
    prob_idx = patient_level_index + 1

    def _sample_from_library(text, prob_index, min_tuple_len, nested=False):
        items = parse_list_from_response(text)
        selected = []
        absent = []
        for item in items:
            if len(item) < min_tuple_len:
                continue
            name = _personalize_phenotype_name_for_gender(item[0], patient_gender)
            if name is None:
                continue
            try:
                if nested and prob_index < len(item) and isinstance(item[prob_index], (list, tuple)):
                    sub = item[prob_index]
                    duration = sub[0] if len(sub) >= 1 else ''
                    trigger = sub[1] if len(sub) >= 2 else ''
                    nature = sub[2] if len(sub) >= 3 else ''
                    prob = float(sub[3]) if len(sub) >= 4 else 0.5
                    payload = (name, patient_stage, duration, trigger, nature)
                    if random.random() < prob:
                        selected.append(payload)
                    else:
                        absent.append(str(name))
                else:
                    prob = float(item[prob_index]) if prob_index < len(item) else 0.5
                    payload = (name, patient_stage)
                    if random.random() < prob:
                        selected.append(payload)
                    else:
                        absent.append(str(name))
            except (ValueError, TypeError):
                pass
        return selected, absent

    selected_symptoms,   absent_symptoms   = _sample_from_library(
        symptoms_text, prob_idx, prob_idx + 1, nested=True
    )
    selected_signs,      absent_signs      = _sample_from_library(signs_text,           prob_idx, prob_idx + 1)
    selected_lab_tests,  absent_lab_tests  = _sample_from_library(lab_tests_text,       prob_idx, prob_idx + 1)
    selected_imaging,    absent_imaging    = _sample_from_library(imaging_text,         prob_idx, prob_idx + 1)
    selected_functional, absent_functional = _sample_from_library(functional_tests_text, prob_idx, prob_idx + 1)

    return (selected_symptoms, selected_signs, selected_lab_tests,
            selected_imaging, selected_functional,
            absent_symptoms, absent_signs, absent_lab_tests,
            absent_imaging, absent_functional)


# ============================================================
# 模块2.4
# ============================================================

def _category_key_of(cat: str) -> str:
    """把中文类型标签映射为 symptoms/signs/lab_tests/imaging/functional 之一。"""
    cat = (cat or '').strip()
    mapping = {
        '症状': 'symptoms',
        '体征': 'signs',
        '实验室检查': 'lab_tests',
        '化验检查': 'lab_tests',
        '影像检查': 'imaging',
        '影像': 'imaging',
        '功能检查': 'functional',
    }
    return mapping.get(cat, '')


_CATEGORY_LABELS = ('症状', '体征', '实验室检查', '影像检查', '功能检查')

_CATEGORY_KEYS = {
    '症状': 'symptoms',
    '体征': 'signs',
    '实验室检查': 'lab_tests',
    '化验检查': 'lab_tests',
    '影像检查': 'imaging',
    '影像': 'imaging',
    '功能检查': 'functional',
}
_CATEGORY_LABEL_BY_KEY = {
    'symptoms': '症状',
    'signs': '体征',
    'lab_tests': '实验室检查',
    'imaging': '影像检查',
    'functional': '功能检查',
}


def _phenotype_name(item):
    if isinstance(item, (list, tuple)) and item:
        return str(item[0]).strip()
    return str(item).strip()


def _parse_correlation_response(response):
    if not response or not str(response).strip():
        raise ValueError('模块2.4返回为空')
    text = str(response).strip()
    fenced = re.search(r'```(?:json|python)?\s*(.*?)```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, TypeError):
            pass
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        fragment = text[start:end + 1]
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(fragment)
                if isinstance(value, dict):
                    return value
            except (ValueError, SyntaxError, TypeError):
                pass
    raise ValueError('模块2.4返回不是合法JSON对象')


def _candidate_payloads(library_text, category_key, patient_level_index,
                        patient_stage, patient_gender=''):
    payloads = {}
    prob_idx = patient_level_index + 1
    for item in parse_list_from_response(library_text):
        if not isinstance(item, (list, tuple)) or not item:
            continue
        name = _personalize_phenotype_name_for_gender(item[0], patient_gender)
        if not name:
            continue
        if category_key == 'symptoms' and prob_idx < len(item) and isinstance(item[prob_idx], (list, tuple)):
            staged = item[prob_idx]
            payloads[name] = (
                name,
                patient_stage,
                staged[0] if len(staged) >= 1 else '',
                staged[1] if len(staged) >= 2 else '',
                staged[2] if len(staged) >= 3 else '',
            )
        else:
            payloads[name] = (name, patient_stage)
    return payloads


def _candidate_probabilities(library_text, category_key, patient_level_index,
                             patient_gender=''):
    """Return the stage-specific sampling probability for each frozen candidate."""
    probabilities = {}
    prob_idx = patient_level_index + 1
    for item in parse_list_from_response(library_text):
        if not isinstance(item, (list, tuple)) or not item:
            continue
        name = _personalize_phenotype_name_for_gender(item[0], patient_gender)
        if not name:
            continue
        try:
            if category_key == 'symptoms' and prob_idx < len(item) and \
                    isinstance(item[prob_idx], (list, tuple)):
                staged = item[prob_idx]
                probability = float(staged[3]) if len(staged) >= 4 else 0.5
            else:
                probability = float(item[prob_idx]) if prob_idx < len(item) else 0.5
        except (TypeError, ValueError):
            probability = 0.5
        probabilities[name] = max(probabilities.get(name, 0.0), probability)
    return probabilities


def _build_phenotype_state(selected_by_key, absent_by_key, payloads_by_key,
                           probabilities_by_key=None):
    state = {}
    probabilities_by_key = probabilities_by_key or {}
    for key in _CATEGORY_LABEL_BY_KEY:
        positive = {}
        for item in selected_by_key.get(key, []):
            name = _phenotype_name(item)
            if name:
                positive[name] = tuple(item) if isinstance(item, (list, tuple)) else (name, '')
        negative = []
        seen_negative = set()
        for item in absent_by_key.get(key, []):
            name = _phenotype_name(item)
            if name and name not in positive and name not in seen_negative:
                negative.append(name)
                seen_negative.add(name)
        payloads = dict(payloads_by_key.get(key, {}))
        payloads.update(positive)
        state[key] = {
            'positive': positive,
            'negative': negative,
            'payloads': payloads,
            'probabilities': dict(probabilities_by_key.get(key, {})),
        }
    return state


def _state_hash(state):
    canonical = []
    for key in sorted(state):
        canonical.extend((key, name, 'positive') for name in sorted(state[key]['positive']))
        canonical.extend((key, name, 'negative') for name in sorted(state[key]['negative']))
    raw = json.dumps(canonical, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _state_for_prompt(state):
    result = {}
    for key, label in _CATEGORY_LABEL_BY_KEY.items():
        result[label] = {
            'positive': list(state[key]['positive']),
            'negative': list(state[key]['negative']),
        }
    return result


def _apply_correlation_changes(state, changes, patient_stage):
    applied = []
    rejected = []
    for change in changes if isinstance(changes, list) else []:
        if not isinstance(change, dict):
            rejected.append({'change': change, 'reason': 'change不是对象'})
            continue
        category = str(change.get('category', '')).strip()
        key = _CATEGORY_KEYS.get(category, '')
        name = str(change.get('name', '')).strip()
        source = str(change.get('from', '')).strip().lower()
        target = str(change.get('to', '')).strip().lower()
        if (not key or not name or source not in ('positive', 'negative')
                or target not in ('positive', 'negative') or source == target):
            rejected.append({'change': change, 'reason': '类别、名称或目标状态非法'})
            continue
        bucket = state[key]
        universe = set(bucket['positive']) | set(bucket['negative']) | set(bucket['payloads'])
        if name not in universe:
            rejected.append({'change': change, 'reason': '表型不在冻结候选集合中'})
            continue
        actual = 'positive' if name in bucket['positive'] else 'negative'
        if source != actual:
            rejected.append({'change': change, 'reason': f'来源状态不匹配，当前为{actual}'})
            continue
        if source == 'positive' and target == 'negative' and \
                str(change.get('basis', '')).strip().lower() not in {
                    'mutual_exclusion', 'reverse_contradiction', 'hard_dependency'
                }:
            rejected.append({
                'change': change,
                'reason': '阳性表型仅允许基于明确硬冲突转为阴性',
            })
            continue
        if source == 'positive' and target == 'negative':
            conflict = change.get('conflicts_with')
            conflict_category = str(
                conflict.get('category', '') if isinstance(conflict, dict) else ''
            ).strip()
            conflict_key = _CATEGORY_KEYS.get(conflict_category, '')
            conflict_name = str(
                conflict.get('name', '') if isinstance(conflict, dict) else ''
            ).strip()
            conflict_state = str(
                conflict.get('state', '') if isinstance(conflict, dict) else ''
            ).strip().lower()
            conflict_bucket = state.get(conflict_key, {})
            actual_conflict_state = (
                'positive' if conflict_name in conflict_bucket.get('positive', {})
                else 'negative' if conflict_name in conflict_bucket.get('negative', [])
                else ''
            )
            if not conflict_key or not conflict_name or \
                    (conflict_key == key and conflict_name == name) or \
                    conflict_state not in ('positive', 'negative') or \
                    actual_conflict_state != conflict_state:
                rejected.append({
                    'change': change,
                    'reason': '阳性转阴性必须引用冻结状态中的明确冲突表型',
                })
                continue
        if target == 'positive':
            bucket['negative'] = [n for n in bucket['negative'] if n != name]
            if key == 'symptoms':
                fallback = (name, patient_stage, '', '', '')
            else:
                fallback = (name, patient_stage) if name in bucket['payloads'] else (name, '伴随疾病')
            bucket['positive'][name] = bucket['payloads'].get(name, fallback)
        else:
            bucket['positive'].pop(name, None)
            bucket['negative'].append(name)
        applied.append(change)
    return applied, rejected


def _state_to_lists(state):
    selected = [list(state[key]['positive'].values()) for key in _CATEGORY_LABEL_BY_KEY]
    absent = [list(state[key]['negative']) for key in _CATEGORY_LABEL_BY_KEY]
    return selected + absent


def _is_exertional_functional_test(name):
    """Conservatively identify tests that require active exercise or stress."""
    text = re.sub(r'\s+', '', str(name or ''))
    folded = text.upper()
    if any(term in folded for term in ('CPET', '6MWT')):
        return True
    return any(term in text for term in (
        '心肺运动', '运动心肺', '运动负荷试验', '运动负荷检查',
        '负荷心电图', '平板运动', '运动平板', '活动平板',
        '跑台试验', '踏车试验', 'Bruce试验', '运动试验',
        '运动耐量试验', '运动耐力试验',
        '6分钟步行', '六分钟步行', '穿梭步行',
    ))


def _critical_threshold_in_name(name):
    text = re.sub(r'\s+', '', str(name or ''))
    folded = text.upper().replace('₂', '2')
    predicates = _threshold_predicates(text)
    if any(term in folded for term in ('SPO2', '血氧饱和度')):
        if any(op in ('<', '<=') and value <= 90 for op, value, _ in predicates):
            return True
    respiratory_rate_mentioned = (
        any(term in folded for term in ('呼吸频率', '呼吸次数'))
        or ('RR' in folded and 'RR间期' not in folded)
    )
    if respiratory_rate_mentioned:
        if any(op in ('>', '>=') and value >= 30 for op, value, _ in predicates):
            return True
    if any(term in folded for term in ('收缩压', 'SBP')):
        if any(op in ('<', '<=') and value <= 90 for op, value, _ in predicates):
            return True
    if any(term in folded for term in ('平均动脉压', 'MAP')):
        if any(op in ('<', '<=') and value <= 65 for op, value, _ in predicates):
            return True
    if any(term in folded for term in ('PAO2', '动脉血氧分压')):
        if any(op in ('<', '<=') and value <= 60 for op, value, _ in predicates):
            return True
    return False


def _has_acute_vital_instability(acuity, symptoms=None, signs=None, lab_tests=None,
                                  quantified=None, specific=None):
    names = [
        _phenotype_name(item)
        for items in (symptoms or [], signs or [], lab_tests or [])
        for item in items
    ]
    for name in names:
        text = re.sub(r'\s+', '', str(name or ''))
        if any(negated in text for negated in (
                '无意识障碍', '意识清楚', '未见意识障碍', '否认晕厥')):
            continue
        if any(term in text for term in (
                '休克', '低血压', '血流动力学不稳定',
                '严重低氧', '重度低氧',
                '严重呼吸急促', '呼吸窘迫', '呼吸频率显著增快',
                '意识丧失', '意识障碍', '意识不清', '晕厥发作', '突发晕厥',
                '昏迷', '谵妄', '昏睡', '嗜睡',
        )) or _critical_threshold_in_name(text):
            return True

    quantified_items = quantified or []
    specific_items = specific or []
    if len(quantified_items) != len(specific_items):
        return False
    for index, quantified_item in enumerate(quantified_items):
        if not isinstance(quantified_item, (list, tuple)) or len(quantified_item) != 4:
            continue
        match = _STRICT_QUANTIFIED_NAME_RE.fullmatch(str(quantified_item[0]).strip())
        if not match:
            continue
        source_name, _, suffix = match.groups()
        try:
            value = _specific_value_from_name(
                source_name, suffix, specific_items[index][0]
            )
        except (IndexError, TypeError, ValueError):
            continue
        metric_key = _objective_metric_key(source_name, quantified_item[1])
        metric = str(metric_key or '').split(':')[-1]
        if metric == 'spo2' and value < 90:
            return True
        if metric == 'respiratory_rate' and value >= 30:
            return True
        folded = re.sub(r'\s+', '', source_name).upper().replace('₂', '2')
        if re.match(r'^(?:收缩压|SBP)(?:降低|偏低|$)', folded) and value < 90:
            return True
        if re.match(r'^(?:平均动脉压|MAP)(?:降低|偏低|$)', folded) and value < 65:
            return True
        if any(term in folded for term in ('PAO2', '动脉血氧分压')) and value < 60:
            return True
    return False


def _is_kussmaul_breathing(name):
    text = re.sub(r'\s+', '', str(name or ''))
    return '呼吸' in text and ('库斯莫尔' in text or 'KUSSMAUL' in text.upper())


def _has_kussmaul_without_acidosis(signs=None, lab_tests=None, absent_lab_tests=None):
    sign_names = [_phenotype_name(item) for item in (signs or [])]
    if not any(_is_kussmaul_breathing(name) for name in sign_names):
        return False
    positive_names = [_phenotype_name(item) for item in (lab_tests or [])]
    acid_terms = ('碳酸氢根降低', '血碳酸氢根降低', 'pH降低',
                  '酸碱度降低', '酸血症')
    if any(any(term.lower() in name.lower() for term in acid_terms)
           for name in positive_names):
        return False
    negative_names = [_phenotype_name(item) for item in (absent_lab_tests or [])]
    return (
        any('碳酸氢' in name and '降低' in name for name in negative_names)
        and any((('ph' in name.lower() or '酸碱度' in name) and '降低' in name)
                or '酸血症' in name
                for name in negative_names))

def _has_exercise_functional_contraindication(
        acuity, symptoms=None, signs=None, lab_tests=None, functional_tests=None,
        quantified=None, specific=None, diagnosis=''):
    diagnosis_text = re.sub(r'\s+', '', str(diagnosis or ''))
    acuity_text = str(acuity or '').strip()
    acute_episode_prohibits_exercise = (
        acuity_text == ACUITY_CHRONIC_EXACERBATION
        or (
            acuity_text == ACUITY_ACUTE
            and any(term in diagnosis_text for term in (
                '主动脉夹层', '不稳定型心绞痛', '急性冠脉综合征',
                '心肌梗死', '肺栓塞', '感染性心内膜炎',
            ))
        )
    )
    clinical_items = list(symptoms or []) + list(signs or [])
    severe_stage = any(
        isinstance(item, (list, tuple)) and len(item) >= 2
        and '重度' in str(item[1] or '')
        for item in clinical_items
    )
    symptomatic_severe_aortic_stenosis = (
        '主动脉瓣狭窄' in diagnosis_text
        and severe_stage
        and any(any(term in _phenotype_name(item) for term in (
            '静息性呼吸困难', '端坐呼吸', '不能平卧',
            '晕厥', '近晕厥', '濒倒',
        )) for item in clinical_items)
    )
    return (
        any(_is_exertional_functional_test(_phenotype_name(item))
            for item in (functional_tests or []))
        and (
            acute_episode_prohibits_exercise
            or _has_acute_vital_instability(
                acuity, symptoms, signs, lab_tests, quantified, specific
            )
            or symptomatic_severe_aortic_stenosis
        )
    )


def _repair_kussmaul_without_acidosis(state):
    selected = state['signs']['positive']
    if not _has_kussmaul_without_acidosis(
            selected.values(),
            state['lab_tests']['positive'].values(),
            state['lab_tests']['negative']):
        return []
    changes = []
    for name in list(selected):
        if not _is_kussmaul_breathing(name):
            continue
        selected.pop(name, None)
        if name not in state['signs']['negative']:
            state['signs']['negative'].append(name)
        changes.append({'category': '体征', 'name': name, 'from': 'positive',
                        'to': 'negative', 'basis': 'hard_dependency',
                        'reason': '缺少代谢性酸中毒证据'})
    return changes

def _build_correlation_prompt(seed_text, patient_stage, state, round_index,
                              previous_feedback=''):
    feedback_section = ''
    if previous_feedback:
        feedback_section = f"""

上一轮执行反馈：
{previous_feedback}
请根据当前状态重新判断。若问题仍成立，changes 必须使用上方候选中的精确类别、名称和当前状态，给出可执行的最小修正；若上一轮问题是误报，本轮应返回 clean。"""
    return f"""你是一位资深临床医学专家。请审查患者五类表型的完整阳性/阴性状态，进行最小化一致性修正。

患者信息：{seed_text}
疾病分期：{patient_stage}
当前为第 {round_index} 轮审查。

当前冻结候选状态：
{json.dumps(_state_for_prompt(state), ensure_ascii=False, indent=2)}

{_CORRELATION_RULES_TEXT}
规则：
1. 同时审查症状、体征、实验室检查、影像检查和功能检查，阳性与阴性都必须考虑。
2. 只能修改上面已经出现的表型，禁止创造新名称、改名或改变类别。
3. 默认保留全部已采样表型。仅修正明确互斥、反向矛盾或高度确定的临床依赖；优先用最少的状态翻转消除冲突，概率性共现不能机械强制，也不能因罕见或不典型而删除表型。
4. 允许真实无主观症状；无症状分期或客观检查驱动的就诊不得为了填充列表而将阴性症状翻为阳性。
5. changes 只列出需要翻转状态的项目；status 为 issues 时 changes 不得为空。无问题时 status 必须为 clean 且 changes 为空。
6. 只输出严格 JSON：
{{"status":"issues或clean","issues":["问题"],"changes":[{{"category":"体征","name":"心动过速","from":"positive","to":"negative","basis":"mutual_exclusion或reverse_contradiction或hard_dependency","conflicts_with":{{"category":"体征","name":"心动过缓","state":"positive"}},"reason":"原因"}}]}}
其中 positive→negative 必须提供上述三种明确硬冲突 basis 之一，并用 conflicts_with 精确引用当前冻结状态中的另一表型；不得因低概率、少见或不典型删除阳性表型。{feedback_section}"""


def module_2_4_correlation_correction(selected_symptoms, selected_signs,
                                       selected_lab_tests, selected_imaging,
                                       selected_functional,
                                       symptoms_text, signs_text,
                                       lab_tests_text, imaging_text,
                                       functional_tests_text,
                                       seed_text, patient_stage,
                                       absent_symptoms=None,
                                       absent_signs=None,
                                       absent_lab_tests=None,
                                       absent_imaging=None,
                                       absent_functional=None,
                                       patient_level_index=0,
                                       patient_gender='',
                                       max_rounds=5,
                                       acuity='',
                                       diagnosis='',
                                       force_exercise_contraindication=False,
                                       tag='模块2.4'):
    """
    模块2.4 全模态多轮相关性校正。

    五类阳性/阴性状态共同进入审查；模型只返回状态翻转操作，程序负责确定性
    应用并同步十个列表。达到 clean 且状态不再变化时收敛。
    """
    t0 = time.time()
    selected_by_key = {
        'symptoms': _personalize_phenotype_items_for_gender(selected_symptoms, patient_gender),
        'signs': _personalize_phenotype_items_for_gender(selected_signs, patient_gender),
        'lab_tests': _personalize_phenotype_items_for_gender(selected_lab_tests, patient_gender),
        'imaging': _personalize_phenotype_items_for_gender(selected_imaging, patient_gender),
        'functional': _personalize_phenotype_items_for_gender(selected_functional, patient_gender),
    }
    absent_by_key = {
        'symptoms': _personalize_phenotype_items_for_gender(absent_symptoms, patient_gender),
        'signs': _personalize_phenotype_items_for_gender(absent_signs, patient_gender),
        'lab_tests': _personalize_phenotype_items_for_gender(absent_lab_tests, patient_gender),
        'imaging': _personalize_phenotype_items_for_gender(absent_imaging, patient_gender),
        'functional': _personalize_phenotype_items_for_gender(absent_functional, patient_gender),
    }
    libraries = {
        'symptoms': symptoms_text, 'signs': signs_text,
        'lab_tests': lab_tests_text, 'imaging': imaging_text,
        'functional': functional_tests_text,
    }
    payloads_by_key = {
        key: _candidate_payloads(
            text, key, patient_level_index, patient_stage, patient_gender
        )
        for key, text in libraries.items()
    }
    probabilities_by_key = {
        key: _candidate_probabilities(
            text, key, patient_level_index, patient_gender
        )
        for key, text in libraries.items()
    }
    state = _build_phenotype_state(
        selected_by_key, absent_by_key, payloads_by_key, probabilities_by_key
    )
    acuity = str(acuity or '').strip()
    if not acuity:
        seed_tail = str(seed_text or '').strip().split()
        candidate = seed_tail[-1] if seed_tail else ''
        if candidate in {ACUITY_ACUTE, ACUITY_CHRONIC, ACUITY_CHRONIC_EXACERBATION}:
            acuity = candidate
    diagnosis = str(diagnosis or '').strip() or str(seed_text or '').split('#', 1)[0].strip()
    seen_hashes = {_state_hash(state)}
    prompts = []
    rounds = []
    previous_feedback = ''
    pending_issues = []

    for round_index in range(1, max(1, int(max_rounds)) + 1):
        prompt = _build_correlation_prompt(
            seed_text, patient_stage, state, round_index, previous_feedback
        )
        prompts.append(prompt)
        print(f"    [{tag}] 第 {round_index} 轮全模态相关性审查...")
        response = call_gpt5(prompt, tag=f"{tag}-R{round_index}")
        try:
            parsed = _parse_correlation_response(response)
        except ValueError as exc:
            rounds.append({'round': round_index, 'status': 'invalid', 'error': str(exc), 'raw': response or ''})
            pending_issues = [str(exc)]
            previous_feedback = json.dumps({
                'issues': pending_issues,
                'instruction': '上一轮输出无效；请严格按JSON格式重新审查。',
            }, ensure_ascii=False)
            continue

        status = str(parsed.get('status', '')).strip().lower()
        issues = parsed.get('issues', [])
        requested_changes = parsed.get('changes', [])
        before_hash = _state_hash(state)
        applied, rejected = _apply_correlation_changes(
            state, requested_changes, patient_stage
        )
        deterministic_changes = _repair_objective_positive_conflicts(
            state, patient_stage
        )
        deterministic_changes.extend(
            _repair_objective_threshold_state(state, patient_stage)
        )
        deterministic_changes.extend(
            _repair_kussmaul_without_acidosis(state)
        )
        after_hash = _state_hash(state)
        rounds.append({
            'round': round_index,
            'status': status,
            'issues': issues,
            'applied_changes': applied,
            'deterministic_changes': deterministic_changes,
            'rejected_changes': rejected,
            'state_hash': after_hash,
            'raw': response or '',
        })

        if (status == 'clean' and not issues and not requested_changes
                and not applied and not deterministic_changes
                and not rejected and before_hash == after_hash):
            audit = {
                'schema_version': 2,
                'status': 'converged',
                'rounds': rounds,
                'final_hash': after_hash,
            }
            print(f"    [{tag}] {round_index} 轮后收敛，总用时 {round(time.time()-t0,1)}s")
            return (json.dumps(prompts, ensure_ascii=False),
                    json.dumps(audit, ensure_ascii=False),
                    *_state_to_lists(state))

        if after_hash == before_hash:
            if issues:
                pending_issues = issues if isinstance(issues, list) else [str(issues)]
            elif rejected and not pending_issues:
                pending_issues = ['上一轮请求的修正无法执行']
            feedback_payload = {
                'issues': pending_issues,
                'rejected_changes': rejected,
                'deterministic_overrides': deterministic_changes,
                'instruction': '问题未改变当前状态；请给出精确可执行的最小状态翻转，或确认其为误报并返回clean。',
            }
            previous_feedback = json.dumps(feedback_payload, ensure_ascii=False)
            print(f"    [{tag}] 第 {round_index} 轮报告问题但无有效修正，继续重新审查")
            continue
        previous_feedback = ''
        pending_issues = []
        if after_hash in seen_hashes:
            print(f"    [{tag}] 第 {round_index} 轮检测到状态振荡，重启本模块")
            return None
        seen_hashes.add(after_hash)

    print(f"    [{tag}] ⚠️ 在 {max_rounds} 轮内未收敛，重启本模块")
    return None



def _correlation_audit_is_converged(audit_text, expected_hash=None):
    try:
        audit = json.loads(str(audit_text or ''))
    except (ValueError, TypeError):
        return False
    if audit.get('schema_version') != 2 or audit.get('status') != 'converged':
        return False
    return expected_hash is None or audit.get('final_hash') == expected_hash


def _phenotype_state_hash_from_lists(selected_lists, absent_lists):
    selected_by_key = dict(zip(_CATEGORY_LABEL_BY_KEY, selected_lists))
    absent_by_key = dict(zip(_CATEGORY_LABEL_BY_KEY, absent_lists))
    state = _build_phenotype_state(selected_by_key, absent_by_key, {})
    return _state_hash(state)


def _build_final_time_order(symptom_timeline, selected_signs, selected_lab_tests,
                            selected_imaging, selected_functional):
    result = list(symptom_timeline)
    seen = {('症状', _phenotype_name(item)) for item in result}
    groups = (
        ('体征', selected_signs),
        ('实验室检查', selected_lab_tests),
        ('影像检查', selected_imaging),
        ('功能检查', selected_functional),
    )
    for category, items in groups:
        for item in items:
            name = _phenotype_name(item)
            key = (category, name)
            if name and key not in seen:
                result.append((name, category, '就诊', 'D0'))
                seen.add(key)
    return result


def _symptom_projection_from_time_order(items):
    projection = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        name = str(item[0]).strip()
        category = str(item[1]).strip()
        if category != '症状':
            continue
        projection.append((
            name,
            category,
            str(item[2]).strip(),
            str(item[3]).strip().upper(),
        ))
    return projection


def _validated_existing_time_model(row, selected_symptoms):
    if not row.get(COL_M25_OUTPUT, '').strip():
        return None
    expected_names = [_phenotype_name(item) for item in selected_symptoms]
    acuity = (row.get(COL_ACUITY, '') or ACUITY_ACUTE).strip()
    try:
        return _parse_m25_time_model(
            row.get(COL_M25_OUTPUT, '') or '', expected_names, acuity
        )
    except ValueError:
        return None


def _validated_existing_symptom_timeline(row, selected_symptoms):
    time_model = _validated_existing_time_model(row, selected_symptoms)
    return None if time_model is None else time_model['symptom_timeline']



def _default_case_id(patient_idx):
    return f"case_{int(patient_idx):05d}"


def _case_id_for_patient(patient_idx, reserved_case_ids=None):
    reserved = set(reserved_case_ids or set())
    base = _default_case_id(patient_idx)
    if base not in reserved:
        return base
    suffix = 2
    while f"{base}_{suffix}" in reserved:
        suffix += 1
    return f"{base}_{suffix}"


def _is_generated_case_id_for_patient(case_id, patient_idx):
    try:
        base = _default_case_id(patient_idx)
    except (TypeError, ValueError):
        return False
    return case_id == base or case_id.startswith(f"{base}_")


def _existing_case_ids(existing_patients):
    ids = set()
    for row in (existing_patients or {}).values():
        case_id = str((row or {}).get(COL_CASE_ID, '') or '').strip()
        if case_id:
            ids.add(case_id)
    return ids


def _initialize_patient_row_identity(patient_idx, existing_row=None, reserved_case_ids=None):
    row = dict(existing_row) if existing_row else _empty_row()
    case_id = str(row.get(COL_CASE_ID, '') or '').strip()
    if not case_id:
        case_id = _case_id_for_patient(patient_idx, reserved_case_ids)
        row[COL_CASE_ID] = case_id
    else:
        row[COL_CASE_ID] = case_id
    return row

def _parse_integral_text(value, minimum=None, maximum=None):
    """Parse an integer while tolerating the ``N.0`` spelling produced by old CSV loads."""
    text = str(value if value is not None else '').strip()
    if not re.fullmatch(r'[+-]?\d+(?:\.0+)?', text):
        raise ValueError(f'不是整数: {value!r}')
    number = int(float(text))
    if minimum is not None and number < minimum:
        raise ValueError(f'整数小于下限 {minimum}: {value!r}')
    if maximum is not None and number > maximum:
        raise ValueError(f'整数大于上限 {maximum}: {value!r}')
    return number


def _resolve_patient_demographics(row, seed_text):
    """Resolve demographics once; externally injected patient facts remain immutable."""
    case_id = str(row.get(COL_CASE_ID, '') or '').strip()
    patient_idx = row.get('_patient_idx')
    generated_case_id = _is_generated_case_id_for_patient(case_id, patient_idx)
    age_text = str(row.get(COL_AGE, '') or '').strip()
    gender_text = str(row.get(COL_GENDER, '') or '').strip()
    try:
        age = _parse_integral_text(age_text, minimum=1, maximum=120)
    except ValueError:
        if case_id and not generated_case_id:
            raise ValueError(f'输入病例 {case_id} 的年龄无效: {age_text!r}')
        age_min, age_max, _, _, _ = _parse_seed(seed_text)
        age = random.randint(age_min, age_max)

    if gender_text in ('男', '女'):
        gender = gender_text
    elif case_id and not generated_case_id:
        raise ValueError(f'输入病例 {case_id} 的性别无效: {gender_text!r}')
    else:
        _, _, gender, _, _ = _parse_seed(seed_text)

    row[COL_AGE] = str(age)
    row[COL_GENDER] = gender
    return age, gender


def _is_m2_row_complete(row, staging_system=None, csv_path=None, row_index=None):
    required = (
        COL_GENDER, COL_QUANTIFIED, COL_SPECIFIC,
        COL_CHIEF_COMPLAINT, COL_DURATION_TOTAL,
    )
    if any(not str(row.get(col, '') or '').strip() for col in required):
        return False
    selected_lists = [
        parse_list_from_response(row.get(col, '') or '')
        for col in (COL_SYMPTOMS, COL_SIGNS, COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS)
    ]
    absent_lists = [
        parse_list_from_response(row.get(col, '') or '')
        for col in (COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS, COL_ABSENT_LAB_TESTS,
                    COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL)
    ]
    patient_gender = str(row.get(COL_GENDER, '') or '').strip()
    if patient_gender not in ('男', '女'):
        return False
    fixed_laterality = fixed_anatomic_laterality(
        f"{row.get(COL_DIAGNOSIS) or ''} {row.get(COL_SEED) or ''}"
    )
    current_laterality = str(row.get(COL_PATIENT_LATERALITY, '') or '').strip()
    if fixed_laterality and current_laterality != fixed_laterality:
        return False
    final_time_order = parse_list_from_response(row.get(COL_TIME_ORDER, '') or '')
    quantified = parse_list_from_response(row.get(COL_QUANTIFIED, '') or '')
    specific = parse_list_from_response(row.get(COL_SPECIFIC, '') or '')
    comorbidities = parse_list_from_response(row.get(COL_COMORBIDITIES, '') or '')
    acuity = str(row.get(COL_ACUITY, '') or '').strip()
    if not acuity:
        try:
            _, _, _, _, acuity = _parse_seed(row.get(COL_SEED, '') or '')
        except (TypeError, ValueError):
            acuity = ''
    if _has_kussmaul_without_acidosis(
            selected_lists[1], selected_lists[2], absent_lists[2]):
        return False
    for items in selected_lists + absent_lists + [
            comorbidities, final_time_order, quantified, specific]:
        for item in items:
            name = _phenotype_name(item)
            if _personalize_phenotype_name_for_gender(name, patient_gender) != name:
                return False
    expected_hash = _phenotype_state_hash_from_lists(selected_lists, absent_lists)
    if not _correlation_audit_is_converged(row.get(COL_M24_OUTPUT, ''), expected_hash):
        return False
    existing_time_model = _validated_existing_time_model(row, selected_lists[0])
    if existing_time_model is None:
        return False
    if str(row.get(COL_DURATION_TOTAL, '') or '').strip() != str(
            existing_time_model.get('current_episode_duration', '') or '').strip():
        return False
    if _symptom_projection_from_time_order(final_time_order) != list(
            existing_time_model.get('symptom_timeline') or []):
        return False

    if not _symptom_durations_match_timeline(selected_lists[0], final_time_order):
        return False
    try:
        _validate_quantified_timeline(final_time_order, quantified)
        _validate_objective_value_consistency(
            quantified, specific,
            {
                '体征': absent_lists[1],
                '实验室检查': absent_lists[2],
                '影像检查': absent_lists[3],
                '功能检查': absent_lists[4],
            },
        )
    except ValueError:
        return False

    prior_visited = (row.get(COL_PRIOR_VISITED, '') or '').strip()
    if prior_visited == '就诊过':
        try:
            prior_visit_count = _parse_integral_text(
                row.get(COL_PRIOR_VISIT_COUNT, '') or '0', minimum=1
            )
        except (TypeError, ValueError):
            return False
        if _validate_prior_visit_history(
            row.get(COL_M26_OUTPUT, '') or '',
            prior_visit_count,
            _time_model_prior_history_duration(existing_time_model),
            selected_lists[0],
            symptom_timeline=[
                item for item in final_time_order
                if isinstance(item, (list, tuple)) and len(item) >= 2
                and str(item[1]).strip() == '症状'
            ],
        ) is None:
            return False
    elif prior_visited == '未就诊':
        try:
            prior_visit_count = _parse_integral_text(
                row.get(COL_PRIOR_VISIT_COUNT, '') or '0', minimum=0
            )
        except ValueError:
            return False
        if prior_visit_count != 0:
            return False
        if any(str(row.get(col, '') or '').strip() for col in (
            COL_M26_INPUT, COL_M26_OUTPUT, COL_PRIOR_VISIT_HISTORY,
        )):
            return False
    else:
        return False
    if staging_system is not None and csv_path is not None and row_index is not None:
        try:
            ledger = load_fact_ledger(
                csv_path, row_index,
                expected_case_id=str(row.get(COL_CASE_ID, '') or '').strip(),
            )
            expected_input_hash = ledger_input_hash(row, staging_system, existing_time_model)
        except Exception:
            return False
        if ledger.get('provenance', {}).get('source_row_hash') != expected_input_hash:
            return False

    return True


def _row_is_m2_complete(row, staging_system=None, csv_path=None, row_index=None):
    try:
        return _is_m2_row_complete(
            row, staging_system=staging_system, csv_path=csv_path, row_index=row_index
        )
    except TypeError:
        return _is_m2_row_complete(row)


def _save_csv_with_fact_ledgers(csv_path, all_rows):
    _save_csv(csv_path, all_rows)
    for row_index, row in enumerate(all_rows):
        ledger = row.get(PENDING_LEDGER_ROW_KEY) if isinstance(row, dict) else None
        if not isinstance(ledger, dict):
            continue
        summary = write_fact_ledger(csv_path, row_index, ledger)
        row[COL_M281_OUTPUT] = summary


# ============================================================
# 模块2.3 / 2.6（既往就诊状态与长期病史）
# ============================================================

def module_2_3_prior_visit_roll(acuity: str, tag: str = '模块2.3'):
    """
    模块2.3 既往就诊状态 roll（v4 新增）。

    - 急性病：直接 visited='未就诊', count=0，不调 LLM。
    - 慢性 / 慢性急性加重：5:5 概率 roll 是否就诊过；就诊过的再 random.randint(1, 3) 决定次数。
      不调 LLM，但构造伪 prompt/response 字符串记录决策过程，便于审计。

    返回：(prompt_str, response_str, dict{visited, count})
    """
    acuity = (acuity or '').strip()
    if acuity == ACUITY_ACUTE or acuity not in (ACUITY_CHRONIC, ACUITY_CHRONIC_EXACERBATION):
        return '', '', {'visited': '未就诊', 'count': 0}

    roll = random.random()
    if roll < 0.5:
        visited = '未就诊'
        count = 0
    else:
        visited = '就诊过'
        count = random.randint(1, 3)

    prompt = (f"[模块2.3 本地决策] acuity={acuity}, "
              f"5:5 伯努利 roll={roll:.3f}, 阈值=0.5；"
              f"若就诊过则在 [1,3] 区间随机一次次数。")
    response = (f"visited={visited}; count={count}")
    print(f"    [{tag}] {response}")
    return prompt, response, {'visited': visited, 'count': count}


def module_2_6_prior_visit_history(diagnosis, acuity, prior_visit_count, duration_total,
                                     selected_symptoms, selected_signs, selected_labs,
                                     symptom_timeline=None, tag='模块2.6'):
    """
    模块2.6 既往多次就诊经历（本地确定性生成）。

    既往就诊史受次数、总病程、症状起病时间等硬约束影响，交给 LLM 生成容易
    在校验器处反复失败。这里改为本地构造严格合法的 JSON，再复用
    _validate_prior_visit_history 做最终验收；检查/治疗文本仍综合诊断、体征和
    实验室检查，使表型表达尽量丰富。

    返回：(prompt, response, dict{history_text, suggested_chief_complaint})
    """
    def _names_of(items):
        out = []
        for it in (items or []):
            name = _phenotype_name(it)
            if name:
                out.append(name)
        return out

    sym_names = _names_of(selected_symptoms)
    sign_names = _names_of(selected_signs)
    lab_names = _names_of(selected_labs)
    timeline_items = [
        item for item in (symptom_timeline or [])
        if isinstance(item, (list, tuple)) and len(item) >= 4
        and str(item[1]).strip() == '症状'
    ]
    if sym_names and not timeline_items:
        print(f"    [{tag}] ⚠️ 缺少已验证的症状起病时间轴")
        return None

    n = max(1, int(prior_visit_count or 1))
    prompt = (
        f"[模块2.6 本地生成] diagnosis={diagnosis}; acuity={acuity}; "
        f"duration_total={duration_total}; prior_visit_count={n}; "
        f"symptoms={json.dumps(sym_names, ensure_ascii=False)}; "
        f"symptom_timeline={json.dumps(timeline_items, ensure_ascii=False)}; "
        f"signs={json.dumps(sign_names, ensure_ascii=False)}; "
        f"labs={json.dumps(lab_names, ensure_ascii=False)}"
    )

    print(f"    [{tag}] 本地生成既往 {n} 次就诊经历...")
    try:
        payload = _build_deterministic_prior_visit_payload(
            diagnosis=diagnosis,
            acuity=acuity,
            prior_visit_count=n,
            duration_total=duration_total,
            selected_symptoms=selected_symptoms,
            selected_signs=selected_signs,
            selected_labs=selected_labs,
            symptom_timeline=timeline_items,
        )
    except (TypeError, ValueError) as exc:
        print(f"    [{tag}] ⚠️ 既往就诊史本地生成失败: {exc}")
        return None

    response = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    info = _validate_prior_visit_history(
        response, n, duration_total, selected_symptoms,
        symptom_timeline=timeline_items,
    )
    if info is None:
        print(f"    [{tag}] ⚠️ 既往就诊史本地生成未通过校验")
        return None
    return prompt, response, info


# ============================================================
# 模块2.5 / 2.6 / 2.7 / 2.8
# ============================================================

_DURATION_RE = re.compile(r'^(\d+(?:\.\d+)?)\s*(小时|天|个月|月|年)$')
_TIME_LABEL_RE = re.compile(r'^([HDMY])(?:-([0-9]+(?:\.[0-9]+)?)|0)$', re.IGNORECASE)
_UNIT_HOURS = {'H': 1.0, 'D': 24.0, 'M': 30.0 * 24.0, 'Y': 365.0 * 24.0}
_DURATION_UNIT_HOURS = {
    '小时': 1.0,
    '天': 24.0,
    '月': 30.0 * 24.0,
    '个月': 30.0 * 24.0,
    '年': 365.0 * 24.0,
}

_VALID_ACUITIES = {ACUITY_ACUTE, ACUITY_CHRONIC, ACUITY_CHRONIC_EXACERBATION}


def _validate_prior_visit_history(response, expected_count, duration_total,
                                  selected_symptoms, symptom_timeline=None):
    """校验 2.6 的次数、时间边界、时序与主诉归属；失败返回 None。"""
    text = str(response or '').strip()
    try:
        count = int(expected_count)
        duration_hours = _duration_to_hours(duration_total)
    except (TypeError, ValueError):
        return None
    if count < 1 or not text:
        return None

    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        'schema_version', 'visits', 'current_chief'
    }:
        return None
    if type(payload.get('schema_version')) is not int or payload['schema_version'] != 1:
        return None
    visits = payload.get('visits')
    if not isinstance(visits, list) or len(visits) != count:
        return None
    symptom_names = {_phenotype_name(item) for item in (selected_symptoms or [])}
    symptom_names.discard('')
    allowed_symptoms = symptom_names or {'无'}
    required_visit_fields = {
        'visit_no', 'time_label', 'chief_symptom', 'examination', 'treatment', 'outcome'
    }
    allowed_outcomes = {'缓解', '部分缓解', '无改善', '加重', '复发', '稳定'}
    if any(not isinstance(visit, dict) or set(visit) != required_visit_fields
           for visit in visits):
        return None
    try:
        indices = [visit.get('visit_no') for visit in visits]
        labels = [str(visit.get('time_label', '')).strip().upper() for visit in visits]
        offsets = [_time_label_to_hours(label) for label in labels]
    except (AttributeError, TypeError, ValueError):
        return None
    if any(type(index) is not int for index in indices) or indices != list(range(1, count + 1)):
        return None
    if any(label[0] not in {'D', 'M', 'Y'} for label in labels):
        return None
    if any(offset <= 0 or offset > duration_hours for offset in offsets):
        return None
    if any(left <= right for left, right in zip(offsets, offsets[1:])):
        return None
    timeline = symptom_timeline if symptom_timeline is not None else selected_symptoms
    onset_by_name = {}
    for item in timeline or []:
        if isinstance(item, (list, tuple)) and len(item) >= 4 and str(item[1]).strip() == '症状':
            try:
                onset_by_name[_phenotype_name(item)] = _time_label_to_hours(item[3])
            except ValueError:
                return None

    for visit, visit_offset in zip(visits, offsets):
        for field in ('chief_symptom', 'examination', 'treatment', 'outcome'):
            if not isinstance(visit.get(field), str) or not visit[field].strip():
                return None
        if visit['outcome'].strip() not in allowed_outcomes:
            return None
        visit_chief = str(visit['chief_symptom']).strip()
        if visit_chief != '无' and visit_chief not in allowed_symptoms:
            return None
        onset = onset_by_name.get(visit_chief)
        if visit_chief != '无' and onset is not None and onset < visit_offset:
            return None

    current_chief = payload.get('current_chief')
    if not isinstance(current_chief, dict) or set(current_chief) != {
        'symptom', 'change', 'change_time'
    }:
        return None
    if any(not isinstance(current_chief.get(field), str) or not current_chief[field].strip()
           for field in ('symptom', 'change', 'change_time')):
        return None
    suggested = current_chief['symptom'].strip()
    if suggested not in allowed_symptoms:
        return None
    if symptom_names and current_chief['change'].strip() not in {'新发', '加重', '复发'}:
        return None
    if not symptom_names and current_chief['change'].strip() != '无':
        return None
    try:
        change_offset = _time_label_to_hours(current_chief['change_time'])
    except ValueError:
        return None
    if change_offset > duration_hours or change_offset >= offsets[-1]:
        return None
    current_onset = onset_by_name.get(suggested)
    if current_onset is not None and current_onset < change_offset:
        return None
    if current_chief['change'].strip() == '新发' and current_onset != change_offset:
        return None
    if not symptom_names and current_chief['change_time'].strip().upper() != 'D0':
        return None
    history_parts = []
    for visit in visits:
        chief_text = str(visit['chief_symptom']).strip()
        reason_text = '因常规随访/复查就诊' if chief_text == '无' else f'因{chief_text}就诊'
        history_parts.append(
            f"第{visit['visit_no']}次就诊（{str(visit['time_label']).upper()}）："
            f"{reason_text}；"
            f"检查：{str(visit['examination']).strip()}；"
            f"治疗：{str(visit['treatment']).strip()}；"
            f"转归：{str(visit['outcome']).strip()}。"
        )
    return {
        'history_text': '\n'.join(history_parts),
        'suggested_chief_complaint': suggested,
    }


def _duration_to_hours(duration_text):
    match = _DURATION_RE.fullmatch(str(duration_text or '').strip())
    if not match:
        raise ValueError(f'非法患病总时长: {duration_text!r}')
    value = float(match.group(1))
    if value <= 0:
        raise ValueError('患病总时长必须大于0')
    return value * _DURATION_UNIT_HOURS[match.group(2)]


def _prior_visit_time_labels(duration_total, count):
    """Generate deterministic prior-visit labels from early to recent within total duration."""
    duration_days = max(1.0, _duration_to_hours(duration_total) / 24.0)
    labels = []
    last_days = None
    for i in range(count, 0, -1):
        days = max(1, int(round(duration_days * i / (count + 1))))
        if last_days is not None and days >= last_days:
            days = max(1, last_days - 1)
        labels.append(f'D-{days}')
        last_days = days
    return labels


def _format_compact_number(value, precision=12):
    number = float(value)
    if not math.isfinite(number):
        return 'nan'
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f'{number:.{precision}g}'


def _time_label_to_hours(time_label):
    label = str(time_label or '').strip().upper()
    match = _TIME_LABEL_RE.fullmatch(label)
    if not match:
        raise ValueError(f'非法时间标签: {time_label!r}')
    amount = float(match.group(2) or 0)
    return amount * _UNIT_HOURS[match.group(1)]


_REALIZED_DURATION_UNITS = {
    'H': ('小时', '不足1小时'),
    'D': ('天', '不足1天'),
    'M': ('个月', '不足1个月'),
    'Y': ('年', '不足1年'),
}



def _prior_visit_time_label_candidates(duration_hours):
    max_days = int(math.floor(duration_hours / 24.0))
    candidates = []
    for days in range(1, max_days + 1):
        candidates.append((f'D-{days}', days * 24.0))
    for months in range(1, int(math.floor(duration_hours / _UNIT_HOURS['M'])) + 1):
        candidates.append((f'M-{months}', months * _UNIT_HOURS['M']))
    for years in range(1, int(math.floor(duration_hours / _UNIT_HOURS['Y'])) + 1):
        candidates.append((f'Y-{years}', years * _UNIT_HOURS['Y']))
    if not candidates:
        raise ValueError('总病程不足以生成 D/M/Y 既往就诊时间')
    return candidates


def _select_prior_visit_time_labels(duration_hours, count):
    candidates = _prior_visit_time_label_candidates(duration_hours)
    unit_penalty = {'Y': 0, 'M': 1, 'D': 2}
    selected = []
    used_offsets = set()
    previous_offset = duration_hours + 1e-9
    for index in range(count):
        target = duration_hours * (count - index) / (count + 1)
        usable = [
            (
                abs(offset - target) / max(target, 1.0)
                + unit_penalty.get(label[0], 9) * 0.02,
                abs(offset - target),
                -offset,
                label,
                offset,
            )
            for label, offset in candidates
            if 0 < offset <= duration_hours and offset < previous_offset and offset not in used_offsets
        ]
        if not usable:
            raise ValueError('无法生成严格从早到晚且不重复的既往就诊时间')
        _, _, _, label, offset = min(usable)
        selected.append((label, offset))
        used_offsets.add(offset)
        previous_offset = offset
    return selected


def _onset_by_symptom_name(symptom_timeline):
    onset_by_name = {}
    label_by_name = {}
    for item in symptom_timeline or []:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        if str(item[1]).strip() != '症状':
            continue
        name = _phenotype_name(item)
        if not name:
            continue
        label = str(item[3]).strip().upper()
        onset_by_name[name] = _time_label_to_hours(label)
        label_by_name[name] = label
    return onset_by_name, label_by_name


def _summarize_prior_examination(diagnosis, selected_signs, selected_labs):
    signs = [_phenotype_name(item) for item in (selected_signs or []) if _phenotype_name(item)]
    labs = [_phenotype_name(item) for item in (selected_labs or []) if _phenotype_name(item)]
    parts = []
    if signs:
        parts.append('查体关注' + '、'.join(signs[:3]))
    if labs:
        parts.append('复查' + '、'.join(labs[:3]))
    if not parts:
        parts.append('完成生命体征、基础实验室检查及专科评估')
    if diagnosis:
        parts.append(f'结合{diagnosis}评估病情变化')
    return '，'.join(parts)


def _summarize_prior_treatment(diagnosis, acuity, visit_no, has_symptoms=True):
    disease = str(diagnosis or '原发疾病').strip() or '原发疾病'
    if (acuity or '').strip() == ACUITY_CHRONIC_EXACERBATION:
        if not has_symptoms:
            return f'针对{disease}相关客观异常调整长期管理方案，并嘱监测与复诊'
        return f'调整{disease}长期管理方案，给予本次症状对症处理并嘱急性加重预警复诊'
    if (acuity or '').strip() == ACUITY_CHRONIC:
        return f'围绕{disease}进行长期管理、危险因素控制和随访计划调整'
    return f'针对{disease}进行初步处理并安排短期复查'


def _build_deterministic_prior_visit_payload(diagnosis, acuity, prior_visit_count,
                                             duration_total, selected_symptoms,
                                             selected_signs, selected_labs,
                                             symptom_timeline=None):
    count = max(1, int(prior_visit_count or 1))
    duration_hours = _duration_to_hours(duration_total)
    labels_offsets = _select_prior_visit_time_labels(duration_hours, count)
    symptom_names = [_phenotype_name(item) for item in (selected_symptoms or []) if _phenotype_name(item)]
    onset_by_name, label_by_name = _onset_by_symptom_name(symptom_timeline)
    examination = _summarize_prior_examination(diagnosis, selected_signs, selected_labs)
    outcomes = ['部分缓解', '稳定', '无改善', '复发']

    visits = []
    for idx, (label, offset) in enumerate(labels_offsets, start=1):
        available = [name for name in symptom_names if onset_by_name.get(name, duration_hours) >= offset]
        chief = available[-1] if available else '无'
        visits.append({
            'visit_no': idx,
            'time_label': label,
            'chief_symptom': chief,
            'examination': examination,
            'treatment': _summarize_prior_treatment(
                diagnosis, acuity, idx, has_symptoms=bool(symptom_names)
            ),
            'outcome': outcomes[(idx - 1) % len(outcomes)],
        })

    if symptom_names:
        timeline_symptoms = [
            _phenotype_name(item) for item in (symptom_timeline or [])
            if _phenotype_name(item) in symptom_names
        ]
        suggested = timeline_symptoms[-1] if timeline_symptoms else symptom_names[0]
        onset = onset_by_name.get(suggested)
        onset_label = label_by_name.get(suggested, 'D0')
        last_visit_offset = labels_offsets[-1][1]
        if onset is not None and onset < last_visit_offset:
            change = '新发'
            change_time = onset_label
        else:
            change = '加重' if (acuity or '').strip() == ACUITY_CHRONIC_EXACERBATION else '复发'
            change_time = 'D0'
        current_chief = {
            'symptom': suggested,
            'change': change,
            'change_time': change_time,
        }
    else:
        current_chief = {'symptom': '无', 'change': '无', 'change_time': 'D0'}

    return {
        'schema_version': 1,
        'visits': visits,
        'current_chief': current_chief,
    }


def _realize_symptom_durations(selected_symptoms, symptom_timeline):
    """Replace atlas-level typical durations with this patient's realized onset windows."""
    labels_by_name = {
        _phenotype_name(item): str(item[3]).strip().upper()
        for item in symptom_timeline
        if isinstance(item, (list, tuple)) and len(item) == 4
        and str(item[1]).strip() == '症状'
    }
    realized = []
    for item in selected_symptoms:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            realized.append(tuple(item) if isinstance(item, (list, tuple)) else item)
            continue
        values = list(item)
        label = labels_by_name.get(_phenotype_name(item))
        match = _TIME_LABEL_RE.fullmatch(label or '')
        if match:
            amount = float(match.group(2) or 0)
            unit, current_text = _REALIZED_DURATION_UNITS[match.group(1)]
            values[2] = f'{_format_compact_number(amount)}{unit}' if amount else current_text
        realized.append(tuple(values))
    return realized


def _symptom_durations_match_timeline(selected_symptoms, symptom_timeline):
    realized = _realize_symptom_durations(selected_symptoms, symptom_timeline)
    if len(realized) != len(selected_symptoms):
        return False
    for original, expected in zip(selected_symptoms, realized):
        if not isinstance(original, (list, tuple)) or tuple(original) != tuple(expected):
            return False
    return True


def _normalize_timeline_entry(item):
    if isinstance(item, dict):
        allowed = {'name', 'category', 'phase', 'time_label'}
        if set(item) != allowed:
            raise ValueError(f'模块2.5 JSON 时间条目字段非法: {item!r}')
        name = str(item.get('name', '')).strip()
        category = str(item.get('category', '')).strip()
        phase_raw = str(item.get('phase', '')).strip()
        time_label = str(item.get('time_label', '')).strip().upper()
    elif isinstance(item, (list, tuple)) and len(item) == 4:
        name = str(item[0]).strip()
        category = str(item[1]).strip()
        phase_raw = str(item[2]).strip()
        time_label = str(item[3]).strip().upper()
    else:
        raise ValueError(f'模块2.5时间条目必须为四元组或 JSON 对象: {item!r}')
    if not name or category != '症状':
        raise ValueError(f'模块2.5只允许症状条目: {item!r}')
    if phase_raw not in {'起病', '就诊'}:
        raise ValueError(f'模块2.5 phase 只能为起病或就诊: {item!r}')
    phase = '就诊' if _time_label_to_hours(time_label) == 0 else '起病'
    return (name, '症状', phase, time_label)


def _validate_symptom_timeline(items, expected_names, duration_total,
                               acuity=ACUITY_ACUTE):
    normalized = [_normalize_timeline_entry(item) for item in items]
    actual_names = [item[0] for item in normalized]
    expected = [str(name).strip() for name in expected_names]
    if len(actual_names) != len(set(actual_names)):
        raise ValueError('症状时间轴包含重复名称')
    if set(actual_names) != set(expected) or len(actual_names) != len(expected):
        raise ValueError(f'症状名称不守恒: expected={expected!r}, actual={actual_names!r}')

    acuity = str(acuity or '').strip()
    if acuity not in _VALID_ACUITIES:
        raise ValueError(f'非法急慢性类型: {acuity!r}')
    duration_match = _DURATION_RE.fullmatch(str(duration_total or '').strip())
    if not duration_match:
        raise ValueError(f'非法患病总时长: {duration_total!r}')
    duration_unit = duration_match.group(2)
    duration_hours = _duration_to_hours(duration_total)
    offsets = [_time_label_to_hours(item[3]) for item in normalized]
    if any(offset > duration_hours for offset in offsets):
        raise ValueError('症状时间超出患病总时长')
    if any(left < right for left, right in zip(offsets, offsets[1:])):
        raise ValueError('症状时间轴未按从早到晚排序')

    labels = [item[3].upper() for item in normalized]
    prefixes = [label[0] for label in labels]
    if any(re.fullmatch(r'[HDMY]-0+(?:\.0+)?', label, re.IGNORECASE) for label in labels):
        raise ValueError('零时点必须写为 H0 或 D0，不得写为 -0')
    if acuity == ACUITY_ACUTE:
        expected_unit = '小时' if duration_hours <= 24 else '天'
        expected_prefix = 'H' if duration_hours <= 24 else 'D'
        if duration_unit != expected_unit:
            raise ValueError('急性病程总时长单位与时长不匹配')
        if any(prefix != expected_prefix for prefix in prefixes):
            raise ValueError('急性病程的时间标签单位不一致')
        if any(offset == 0 and label != f'{expected_prefix}0'
               for offset, label in zip(offsets, labels)):
            raise ValueError('急性病程的当前时点标签不规范')
    elif acuity == ACUITY_CHRONIC:
        if duration_hours < 30 * 24:
            raise ValueError('慢性病程总时长不得少于30天')
        if duration_unit == '小时' or any(prefix not in {'D', 'M', 'Y'} for prefix in prefixes):
            raise ValueError('慢性病程不得使用小时级总病程或 H 标签')
        if any(offset == 0 and label != 'D0' for offset, label in zip(offsets, labels)):
            raise ValueError('慢性病程的当前时点统一使用 D0')
    else:
        if duration_hours < 30 * 24:
            raise ValueError('慢性急性加重的基础病程不得少于30天')
        if duration_unit == '小时' or any(prefix not in {'H', 'D', 'M', 'Y'} for prefix in prefixes):
            raise ValueError('慢性急性加重的总病程不得以小时表示')
        if normalized and not any(offset <= 14 * 24 for offset in offsets):
            raise ValueError('慢性急性加重缺少近14天内的急性事件')
        if any(offset == 0 and label != 'D0' for offset, label in zip(offsets, labels)):
            raise ValueError('慢性急性加重的当前时点统一使用 D0')
    return normalized


def _validate_symptom_timeline_shape(items, expected_names, duration_total):
    normalized = [_normalize_timeline_entry(item) for item in items]
    actual_names = [item[0] for item in normalized]
    expected = [str(name).strip() for name in expected_names]
    if len(actual_names) != len(set(actual_names)):
        raise ValueError('症状时间轴包含重复名称')
    if set(actual_names) != set(expected) or len(actual_names) != len(expected):
        raise ValueError(f'症状名称不守恒: expected={expected!r}, actual={actual_names!r}')
    duration_hours = _duration_to_hours(duration_total)
    offsets = [_time_label_to_hours(item[3]) for item in normalized]
    if any(offset > duration_hours for offset in offsets):
        raise ValueError('症状时间超出基础病程')
    if any(left < right for left, right in zip(offsets, offsets[1:])):
        raise ValueError('症状时间轴未按从早到晚排序')
    labels = [item[3].upper() for item in normalized]
    if any(re.fullmatch(r'[HDMY]-0+(?:\.0+)?', label, re.IGNORECASE) for label in labels):
        raise ValueError('零时点必须写为 H0 或 D0，不得写为 -0')
    if any(label[0] not in {'H', 'D', 'M', 'Y'} for label in labels):
        raise ValueError('慢性急性加重时间标签单位非法')
    if any(offset == 0 and label != 'D0' for offset, label in zip(offsets, labels)):
        raise ValueError('慢性急性加重的当前时点统一使用 D0')
    return normalized


def _validate_time_model(payload, expected_symptoms, acuity):
    if not isinstance(payload, dict):
        raise ValueError('模块2.5必须返回 JSON 对象')
    expected_keys = {
        'schema_version',
        'underlying_duration',
        'current_episode_duration',
        'symptom_timeline',
    }
    if set(payload) != expected_keys:
        raise ValueError(f'模块2.5 JSON 字段必须且只能为 {sorted(expected_keys)!r}')
    if payload.get('schema_version') != 2:
        raise ValueError('模块2.5 schema_version 必须为 2')

    acuity = str(acuity or '').strip()
    if acuity not in _VALID_ACUITIES:
        raise ValueError(f'非法急慢性类型: {acuity!r}')
    underlying_duration = payload.get('underlying_duration')
    current_episode_duration = payload.get('current_episode_duration')
    symptom_timeline = payload.get('symptom_timeline')
    if not isinstance(symptom_timeline, list):
        raise ValueError('symptom_timeline 必须为列表')

    if acuity == ACUITY_ACUTE:
        if underlying_duration is not None:
            raise ValueError('急性病程 underlying_duration 必须为 null')
        normalized = _validate_symptom_timeline(
            symptom_timeline, expected_symptoms,
            current_episode_duration, acuity=acuity,
        )
    elif acuity == ACUITY_CHRONIC:
        if not isinstance(underlying_duration, str) or not underlying_duration.strip():
            raise ValueError('慢性病程必须提供 underlying_duration')
        if current_episode_duration != underlying_duration:
            raise ValueError('慢性稳定病程的 current_episode_duration 必须等于 underlying_duration')
        normalized = _validate_symptom_timeline(
            symptom_timeline, expected_symptoms,
            current_episode_duration, acuity=acuity,
        )
    else:
        if not isinstance(underlying_duration, str) or not underlying_duration.strip():
            raise ValueError('慢性急性加重必须提供基础病程 underlying_duration')
        if not isinstance(current_episode_duration, str) or not current_episode_duration.strip():
            raise ValueError('慢性急性加重必须提供本次发作 current_episode_duration')
        underlying_hours = _duration_to_hours(underlying_duration)
        current_hours = _duration_to_hours(current_episode_duration)
        if underlying_hours < 30 * 24:
            raise ValueError('慢性急性加重的基础病程不得少于30天')
        if underlying_hours < current_hours:
            raise ValueError('基础病程不得短于本次发作时长')
        normalized = _validate_symptom_timeline_shape(
            symptom_timeline, expected_symptoms, underlying_duration
        )
        if normalized and not any(_time_label_to_hours(item[3]) <= current_hours for item in normalized):
            raise ValueError('慢性急性加重缺少落在本次发作窗口内的症状')

    return {
        'schema_version': 2,
        'underlying_duration': underlying_duration,
        'current_episode_duration': current_episode_duration,
        'symptom_timeline': normalized,
    }


def _parse_m25_time_model(response, expected_symptoms, acuity):
    try:
        payload = json.loads(str(response or '').strip())
    except (TypeError, ValueError) as exc:
        raise ValueError('模块2.5必须返回严格 JSON 对象') from exc
    return _validate_time_model(payload, expected_symptoms, acuity)


def _time_model_to_json(time_model):
    return json.dumps(time_model, ensure_ascii=False)


def _default_time_model_for_empty_symptoms(acuity):
    """无阳性主观症状时，给 2.5 一个合法、保守的三层时间模型。"""
    acuity = str(acuity or '').strip()
    if acuity == ACUITY_ACUTE:
        return {
            'schema_version': 2,
            'underlying_duration': None,
            'current_episode_duration': '12小时',
            'symptom_timeline': [],
        }
    if acuity == ACUITY_CHRONIC:
        return {
            'schema_version': 2,
            'underlying_duration': '6个月',
            'current_episode_duration': '6个月',
            'symptom_timeline': [],
        }
    if acuity == ACUITY_CHRONIC_EXACERBATION:
        return {
            'schema_version': 2,
            'underlying_duration': '1年',
            'current_episode_duration': '3天',
            'symptom_timeline': [],
        }
    raise ValueError(f'非法急慢性类型: {acuity!r}')


def _time_model_prior_history_duration(time_model):
    return time_model.get('underlying_duration') or time_model.get('current_episode_duration') or ''



def module_2_5_build_timeline(seed_text, patient_stage,
                                     selected_symptoms, selected_signs,
                                     selected_lab_tests, selected_imaging,
                                     selected_functional,
                                     acuity: str = ACUITY_ACUTE,
                                     prior_visited: str = '未就诊',
                                     prior_visit_count: int = 0,
                                     tag="模块2.5"):
    """
    模块2.5 时间排序（GPT调用：仅对症状做时序，输出 schema v2 三层时间模型）。

    输入：2.4 收敛后的 selected_symptoms（其余四类不再参与时序，仅 selected_signs/labs
          等参数保留以维持调用兼容性，但函数体不再使用）。
    返回：(prompt, response_json, time_model)
        - response_json: 严格 JSON 字符串，写入模块2.5 sidecar
        - time_model: schema_version / underlying_duration / current_episode_duration / symptom_timeline
    """
    def _names_of(items):
        return [str(it[0]) for it in items if isinstance(it, (list, tuple)) and it]

    symptoms_names = _names_of(selected_symptoms)
    symptoms_str = (json.dumps(selected_symptoms, ensure_ascii=False)
                    if symptoms_names else '无')

    acuity = (acuity or ACUITY_ACUTE).strip() or ACUITY_ACUTE
    prior_visited = (prior_visited or '未就诊').strip() or '未就诊'

    if not symptoms_names:
        time_model = _default_time_model_for_empty_symptoms(acuity)
        time_model = _validate_time_model(time_model, [], acuity)
        prompt = (
            f"[{tag} 本地决策] 2.4 收敛后无阳性主观症状；"
            "仅跳过症状时间轴生成，体征/实验室/影像/功能检查等客观表型仅作为 observed_at=D0 "
            "由后续最终表型视图保留，不进入 symptom_timeline。"
        )
        response_json = _time_model_to_json(time_model)
        print(f"    [{tag}] 无阳性主观症状，跳过GPT症状时序；"
              f"current_episode_duration={time_model['current_episode_duration']}")
        return prompt, response_json, time_model

    # ----- 根据 (acuity, prior_visited) 切换时间格式说明 -----
    if acuity == ACUITY_ACUTE:
        time_format_desc = (
            "本患者为【急性病程】。underlying_duration 必须为 null；current_episode_duration 表示本次起病到当前就诊为止的时长：\n"
            "  - 若本次发作在 1 天以内，时长用小时表示（如 \"6小时\"），时间标签使用 H-N 形式，H0 代表当前；\n"
            "  - 若超过 1 天，时长用天表示（如 \"3天\"），时间标签使用 D-N 形式，D0 代表当前；\n"
            "  - 所有症状的时间标签必须落在 [起病时刻, 当前(H0/D0)] 这段区间内；\n"
            "  - N 可以是整数，也可以在医学上需要时使用小数（如 H-4.5、D-0.25），但不能写 D-0/H-0；\n"
            "  - 起病最早出现的症状贴近 H-X / D-X，最近新出现/加重的症状贴近 H0/D0。"
        )
        example = '{"schema_version":2,"underlying_duration":null,"current_episode_duration":"6小时","symptom_timeline":[{"name":"胸痛","category":"症状","phase":"起病","time_label":"H-3"},{"name":"呼吸困难","category":"症状","phase":"起病","time_label":"H-1"},{"name":"大汗","category":"症状","phase":"就诊","time_label":"H0"}]}'
    elif acuity == ACUITY_CHRONIC:
        time_format_desc = (
            f"本患者为【慢性稳定】，既往就诊 {prior_visit_count} 次。请决定从首次发病到当前的基础病程：\n"
            "  - underlying_duration 与 current_episode_duration 必须相同；\n"
            "  - 两个时长只能用年 / 月 / 天，例如 \"3年\" / \"8个月\" / \"30天\"；\n"
            "  - 起病时间标签可用 Y-N / M-N / D-N，当前时点统一写 D0；\n"
            "  - N 可以是整数，也可以在医学上需要时使用小数（如 M-1.5、D-0.25），但不能写 D-0；\n"
            "  - 按症状真实出现时间从早到晚排列，稳定慢病不强制设置近期急性事件。"
        )
        example = '{"schema_version":2,"underlying_duration":"2年","current_episode_duration":"2年","symptom_timeline":[{"name":"反复咳嗽","category":"症状","phase":"起病","time_label":"Y-2"},{"name":"活动后气短","category":"症状","phase":"起病","time_label":"M-3"}]}'
    elif acuity == ACUITY_CHRONIC_EXACERBATION:
        time_format_desc = (
            f"本患者为【慢性急性加重】，既往就诊 {prior_visit_count} 次。请同时给出基础病程和本次发作时长：\n"
            "  - underlying_duration 表示基础慢性疾病自首次发病到当前的时长，只能用年 / 月 / 天；\n"
            "  - current_episode_duration 表示本次急性加重/当前发作持续时间，可用小时 / 天；\n"
            "  - underlying_duration 必须大于或等于 current_episode_duration；\n"
            "  - 慢性基线症状可用 Y-N / M-N / D-N，近期加重可用 H-N / D-N，当前时点统一写 D0；\n"
            "  - N 可以是整数，也可以在医学上需要时使用小数（如 H-6.5、D-0.25），但不能写 D-0/H-0；\n"
            "  - 至少 1 条阳性症状必须落在 current_episode_duration 窗口内，表示本次急性加重。"
        )
        example = '{"schema_version":2,"underlying_duration":"2年","current_episode_duration":"5天","symptom_timeline":[{"name":"反复咳嗽","category":"症状","phase":"起病","time_label":"Y-2"},{"name":"活动后气短","category":"症状","phase":"起病","time_label":"M-3"},{"name":"胸闷加重","category":"症状","phase":"起病","time_label":"H-3"}]}'
    else:
        raise ValueError(f'非法急慢性类型: {acuity!r}')

    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息和已选中的【症状】列表，确定所有症状出现的时间先后顺序，并为每个症状标注时间标签。

患者信息：{seed_text}
本患者所处疾病分期：{patient_stage}
急慢性类型：{acuity}
此前就诊状态：{prior_visited}（既往就诊次数：{prior_visit_count}）

已选中的症状（患者主观感受；每项依次为名称、分期、典型持续时间、诱因、性质；可能为空列表）：
{symptoms_str}

时间格式说明：
{time_format_desc}

要求：
1. **只输出一个严格 JSON 对象**，不得输出 Markdown、Python 列表、注释、解释或额外文本。
2. JSON 字段必须且只能为：schema_version、underlying_duration、current_episode_duration、symptom_timeline；schema_version 固定为 2。
3. symptom_timeline 仅包含症状对象，每个对象字段必须且只能为：name、category、phase、time_label；category 固定为“症状”，phase 只能为“起病”或“就诊”。
4. **必须保留所有症状条目**，不得合并、删除或变更名称；列表按时间先后顺序（从早到晚）排列。
5. 体征/化验/影像/功能检查等客观表型不进入 symptom_timeline；它们只在后续最终视图中作为 observed_at=D0 处理。
6. current_episode_duration 和每个症状的相对起病时间必须参考上方典型持续时间，不能短于已经持续的症状。

示例 JSON：
{example}
"""

    n_total = len(selected_symptoms)
    print(f"    [{tag}] 正在确定症状时间顺序（共{n_total}个症状，分期={patient_stage}，"
          f"acuity={acuity}，visited={prior_visited}）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag=tag)
    print(f"    [{tag}] 模块用时 {round(time.time()-t0,1)}s")
    if not response:
        # 让 _retry_module_call 触发重试
        return None

    try:
        time_model = _parse_m25_time_model(response, symptoms_names, acuity)
    except ValueError as exc:
        print(f"    [{tag}] ⚠️ 时间轴校验失败: {exc}")
        return None

    return prompt, _time_model_to_json(time_model), time_model


def _normalize_symptom_name(s):
    """将症状名拆分为原子名集合，用于跨条目语义比较。

    - 按顿号/斜杠/逗号/和/或/及等分隔符拆分；
    - 去除首尾空白；过滤长度 < 2 的碎片（避免单字符误切）；
    - 返回 frozenset[str]；空输入返回空集合。
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ''
    if not s.strip():
        return frozenset()
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE_VCI.split(s) if p and p.strip()]
    parts = [p for p in parts if len(p) >= 2]
    if not parts:
        return frozenset([s.strip()])
    return frozenset(parts)


_COMPOUND_SPLIT_RE_VCI = re.compile(r'[、/／,，]|(?:\s和\s)|(?:\s或\s)|(?:\s及\s)')


def _backfill_missing_symptoms(time_ordered_items, selected_symptoms):
    """
    若 LLM 在时间排序时丢弃了症状条目，按 selected_symptoms 的顺序把缺失症状
    以 (name, '症状', '起病', 'D-1') 的默认时相补回列表最前部。

    使用原子名集合做语义比较，避免复合名（"易疲劳、体力下降"）与独立名
    （"易疲劳" + "体力下降"）并存时造成重复回填。
    """
    existing_atoms = set()
    for it in time_ordered_items:
        if isinstance(it, (list, tuple)) and it:
            existing_atoms |= _normalize_symptom_name(it[0])

    missing = []
    appended_atoms = set()
    for it in selected_symptoms:
        if not isinstance(it, (list, tuple)) or not it:
            continue
        name = str(it[0])
        atoms = _normalize_symptom_name(name)
        if not atoms:
            continue
        if atoms.issubset(existing_atoms) or atoms.issubset(appended_atoms):
            continue
        missing.append((name, '症状', '起病', 'D-1'))
        appended_atoms |= atoms

    if missing:
        return missing + list(time_ordered_items)
    return list(time_ordered_items)


COMORBIDITY_EXCLUSIVE_PAIRS = [
    ('1型糖尿病', '2型糖尿病'),
    ('妊娠', '绝经'),
    ('妊娠', '前列腺'),
]


def _gender_applies(item_gender: str, patient_gender: str) -> bool:
    g = re.sub(r'\s+', '', str(item_gender or '')).strip()
    if not g or g.upper() in ('通用', '任意', '不限', 'N/A', '-', '全部', '均可'):
        return True
    if '男' in g and '女' in g:
        return True
    return g == (patient_gender or '').strip()


def _age_applies(age_min, age_max, patient_age: int) -> bool:
    try:
        amin = int(age_min)
    except (ValueError, TypeError):
        amin = 0
    try:
        amax = int(age_max)
    except (ValueError, TypeError):
        amax = 120
    if amin > amax:
        amin, amax = amax, amin
    if patient_age is None:
        return True
    return amin <= patient_age <= amax


def module_2_2_sample_comorbidities(comorbidities_text,
                                     patient_age=None, patient_gender=''):
    """
    模块2.2 伴随疾病选择（伯努利随机采样，无GPT调用）。
    A9：库中条目必须为 5 元组 `(name, prob, age_min, age_max, gender)`，
        按患者 age/gender 过滤后再采样，并去除定义于
        COMORBIDITY_EXCLUSIVE_PAIRS 的互斥对。
    """
    comorbidities_list = parse_list_from_response(comorbidities_text)
    selected = []
    dropped_demo = []

    for item in comorbidities_list:
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            continue
        name = str(item[0])
        try:
            prob = float(item[1])
        except (ValueError, TypeError):
            continue
        age_min, age_max, gender = item[2], item[3], str(item[4])
        if not _gender_applies(gender, patient_gender):
            dropped_demo.append(f'{name}(性别)')
            continue
        if not _age_applies(age_min, age_max, patient_age):
            dropped_demo.append(f'{name}(年龄)')
            continue
        if random.random() < prob:
            selected.append(name)

    if dropped_demo:
        print(f"    [模块2.2] 人口学过滤移除: {dropped_demo}")

    if selected:
        filtered = list(selected)
        for a, b in COMORBIDITY_EXCLUSIVE_PAIRS:
            hit_a = [n for n in filtered if a in n]
            hit_b = [n for n in filtered if b in n]
            if hit_a and hit_b:
                drop_side = random.choice([hit_a, hit_b])
                for n in drop_side:
                    if n in filtered:
                        filtered.remove(n)
                print(f"    [模块2.2] 互斥对 ({a}/{b}) 移除: {drop_side}")
        selected = filtered

    return selected


# ---- 模块2.7 辅助：修复 LLM 输出的复合多维生命体征 ----

_PARAM_OK_RE = re.compile(
    r'^(\d+\.?\d*),(\d+\.?\d*),(nan|\d+\.?\d*),(nan|\d+\.?\d*)$',
    re.IGNORECASE,
)
_QUANTIFIED_NAME_RE = re.compile(r'^(.+?)\{([^{}]+)\}(.+)$')
_PARAM_COMPOUND_RE = re.compile(
    r'([^{}\s,]+)\{([^{}]+)\}([^\s,()\[\]\'"]*)'
)

_COMPOUND_VITAL_SPLIT_MAP = {
    '血压': ('收缩压', '舒张压'),
    'BP': ('SBP', 'DBP'),
    'PaO2/PaCO2': ('PaO2', 'PaCO2'),
    'FEV1/FVC': ('FEV1', 'FVC'),
}


def _try_split_compound_param_block(label, inner, unit):
    """若 {mean,sd,min,max} 内每项都是 'a/b' 形式，按 label 拆成两条条目。

    返回 [(new_label_1, inner_1, unit), (new_label_2, inner_2, unit)] 或 None。
    """
    parts = [p.strip() for p in inner.split(',')]
    if len(parts) != 4:
        return None
    left_vals, right_vals = [], []
    for p in parts:
        if p.lower() == 'nan':
            left_vals.append('nan')
            right_vals.append('nan')
            continue
        if '/' not in p:
            return None
        segs = p.split('/')
        if len(segs) != 2:
            return None
        left_vals.append(segs[0].strip())
        right_vals.append(segs[1].strip())
    split_names = None
    for key, names in _COMPOUND_VITAL_SPLIT_MAP.items():
        if key in label:
            split_names = names
            break
    if split_names is None:
        if '血压' in label or label.upper() in ('BP', '血压'):
            split_names = ('收缩压', '舒张压')
        else:
            split_names = (f'{label}_1', f'{label}_2')
    inner_l = ','.join(left_vals)
    inner_r = ','.join(right_vals)
    if not _PARAM_OK_RE.match(inner_l) or not _PARAM_OK_RE.match(inner_r):
        return None
    return [(split_names[0], inner_l, unit), (split_names[1], inner_r, unit)]


def _postprocess_compound_vitals(items):
    """遍历 2.7 输出的 4 元组列表，拆分含复合 {mean,sd,min,max} 的条目。

    items: list of 4-tuples (content, category, phase, rel_day)
    返回: 处理后的 list；破损且不可修复时抛错，绝不静默丢弃表型。
    """
    if not isinstance(items, list):
        return items
    new_items = []
    for it in items:
        if not isinstance(it, (list, tuple)) or len(it) < 4:
            new_items.append(it)
            continue
        content, cat, phase, rel_day = it[0], it[1], it[2], it[3]
        if not isinstance(content, str) or '{' not in content:
            new_items.append(it)
            continue
        m = _PARAM_COMPOUND_RE.search(content)
        if not m:
            new_items.append(it)
            continue
        label, inner, unit = m.group(1), m.group(2), m.group(3)
        if _PARAM_OK_RE.match(inner):
            new_items.append(it)
            continue
        split_result = _try_split_compound_param_block(label, inner, unit)
        if split_result is None:
            raise ValueError(f'2.7 无法解析复合量化块: {content!r}')
        for new_label, new_inner, new_unit in split_result:
            new_content = f'{new_label}{{{new_inner}}}{new_unit}'
            new_items.append((new_content, cat, phase, rel_day))
    return new_items


def _validate_quantified_timeline(source_items, output_items):
    """2.7 只允许改写名称字段中的数值块，事件身份必须逐项守恒。"""
    if not isinstance(source_items, list) or not isinstance(output_items, list):
        raise ValueError('2.7 输入输出必须为列表')
    if len(source_items) != len(output_items):
        raise ValueError('2.7 不得增删、拆分或合并表型条目')

    validated = []
    for index, (source, output) in enumerate(zip(source_items, output_items)):
        if not isinstance(source, (list, tuple)) or len(source) != 4:
            raise ValueError(f'2.7 源条目 {index} 不是四元组')
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise ValueError(f'2.7 输出条目 {index} 不是四元组')

        source_name = str(source[0]).strip()
        output_name = str(output[0]).strip()
        source_meta = tuple(str(value).strip() for value in source[1:])
        output_meta = tuple(str(value).strip() for value in output[1:])
        if output_meta != source_meta:
            raise ValueError(f'2.7 条目 {index} 的类别/phase/时间被改写')

        if output_name == source_name:
            if _is_obviously_quantifiable(source):
                raise ValueError(f'2.7 条目 {index} 的明确数值型表型未量化')
        else:
            output_name = _validate_quantified_name(source_name, output_name, source_meta[0])
        validated.append((output_name, *output_meta))
    return validated


_QUANTIFIABLE_CATEGORIES = {
    '体征', '实验室检查', '化验检查', '影像检查', '功能检查'
}
_QUANTITATIVE_DIRECTION_HINTS = (
    '升高', '增高', '偏高', '降低', '减低', '偏低', '延长', '缩短',
    '增快', '减慢', '过速', '过缓', '增大', '减少', '缩小', '扩大',
    '扩张', '增多', '狭窄', '超标', '不足', '超过', '低于', '高于',
)
_SIGN_CONTINUOUS_MEASURES = (
    '体温', '心率', '心室率', '脉率', '呼吸频率', 'SpO2', 'SPO2',
    '血氧饱和度', '收缩压', '舒张压', '平均动脉压', '脉压差',
    '毛细血管再充盈时间', '腰围', 'BMI', '体重', '身高',
)
_TEST_CONTINUOUS_MEASURES = (
    '计数', '水平', '浓度', '滴度', '活性', '时间', '比值', '比例',
    '百分比', '分数', '速度', '流速', '压差', '压力', '面积', '直径',
    '厚度', '距离', '射血分数', '容积', '容量', '指数', '负荷', '阻力',
    '顺应性', '潜伏期', '波幅', '振幅', '频率', '密度', '硬度', '峰值',
    '平均值', '下降率', '灰度值', '摄取值', 'SUV', 'ADC', 'T值', 'Z值',
    'FEV1', 'FVC', 'DLCO', 'TLC', 'PEF', 'eGFR', 'GFR', 'PaO2', 'PaCO2',
)
_CATEGORICAL_OR_ORDINAL_TERMS = (
    '阳性', '阴性', '未检出', '可闻', '不可触及', '消失', '缺如',
    '杂音', '压痛', '反跳痛', '水肿', '搏动减弱', '脉搏减弱',
    '脉搏短绌', '奔马律', '三音律',
    '肌力', '瘫痪', '意识障碍', '反射', '评分', '分级', '等级',
    'Barthel', '巴氏指数',
)
_COMPOUND_QUANTIFICATION_CONNECTOR_RE = re.compile(
    r'(?:且|同时|并伴|合并|以及|、|或|'
    r'(?<!饱)(?<!中)(?<!亲)和(?!度|力|抗体)|'
    r'(?<!触)(?<!闻)(?<!累)及(?!以上|以下)|并)'
)
_CJK_SLASH_CONNECTOR_RE = re.compile(
    r'(?<=[\u4e00-\u9fff])\s*[/／]\s*(?=[\u4e00-\u9fff])'
)
_RATIO_TERM_RE = re.compile(r'(?:比值|比率|比例|指数)')
_CJK_UNIT_SLASH_RE = re.compile(
    r'(?:次|毫升|升|[千毫微纳]?[克米瓦]|单位|个)\s*[/／]\s*'
    r'(?:分(?:钟)?|秒|小时|天|升|分升|平方米|立方米|视野|高倍视野)'
)
_ABNORMAL_OR_THRESHOLD_RE = re.compile(
    r'(?:升高|增高|降低|减低|延长|缩短|增大|减少|[<>＜＞]|[≤≥])'
)
_NUMERIC_SLASH_RE = re.compile(
    r'(?:[<>＜＞≤≥]\s*)?[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*/\s*'
    r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
)
_EXPLICIT_THRESHOLD_RE = re.compile(
    r'(?P<op>>=|<=|>|<|＞|＜|≥|≤)\s*'
    r'(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*'
    r'(?P<unit>Agatston\s*(?:units?|单位)|毫米汞柱|次/(?:分|min)|毫秒|秒|分钟|小时|°[CcFf]?|[%％℃]|'
    r'[A-Za-zµμ×][A-Za-z0-9µμ×·^/._-²³㎡]*)?',
    re.IGNORECASE,
)
_REVERSED_THRESHOLD_RE = re.compile(
    r'(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*'
    r'(?P<unit>Agatston\s*(?:units?|单位)|毫米汞柱|次/(?:分|min)|毫秒|秒|分钟|小时|°[CcFf]?|[%％℃]|'
    r'[A-Za-zµμ×][A-Za-z0-9µμ×·^/._-²³㎡]*)?\s*'
    r'(?P<op>>=|<=|>|<|＞|＜|≥|≤)\s*(?=[A-Za-z\u4e00-\u9fff])',
    re.IGNORECASE,
)
_STRICT_QUANTIFIED_NAME_RE = re.compile(r'^(.+?)\{([^{}]+)\}([^{}]*)$')
_NUMBER_TOKEN_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def _has_multiple_quantified_findings(name):
    if _NUMERIC_SLASH_RE.search(name):
        return True
    if _CJK_SLASH_CONNECTOR_RE.search(name) and \
            not _RATIO_TERM_RE.search(name) and \
            not _CJK_UNIT_SLASH_RE.search(name):
        return True
    parts = [
        part.strip() for part in _COMPOUND_QUANTIFICATION_CONNECTOR_RE.split(name)
        if part.strip()
    ]
    return len(parts) >= 2 and any(
        _ABNORMAL_OR_THRESHOLD_RE.search(part) for part in parts
    )


def _normalize_measurement_unit(unit):
    normalized = re.sub(r'\s+', '', str(unit or '')).replace('％', '%').lower()
    normalized = normalized.replace('℃', '°c')
    if re.match(r'^10\^', normalized):
        normalized = f'×{normalized}'
    return normalized


_AGATSTON_UNIT_ALIASES = {'au', 'agatstonunit', 'agatstonunits', 'agatston单位'}
_AGATSTON_CONTEXT_RE = re.compile(
    r'(?:agatston|\bCAC\b|冠状动脉钙化|冠脉钙化|钙化积分|钙化评分)',
    re.IGNORECASE,
)


def _canonical_explicit_threshold_unit(unit, context=''):
    normalized = _normalize_measurement_unit(unit).replace('.', '')
    if normalized in _AGATSTON_UNIT_ALIASES and (
            normalized != 'au' or _AGATSTON_CONTEXT_RE.search(str(context or ''))):
        return 'agatston_unit'
    return normalized


def _is_obviously_quantifiable(item):
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return False
    name, category = str(item[0]), str(item[1]).strip()
    if category not in _QUANTIFIABLE_CATEGORIES or not name.strip():
        return False
    if _has_multiple_quantified_findings(name):
        return False
    if any(term in name for term in _CATEGORICAL_OR_ORDINAL_TERMS) or \
            re.search(r'(?:\d+(?:\.\d*)?)\s*(?:级|/6级|分)(?:\D|[^/分]|$)', name):
        return False
    has_direction = any(hint in name for hint in _QUANTITATIVE_DIRECTION_HINTS)
    has_threshold = bool(_EXPLICIT_THRESHOLD_RE.search(name))
    if category == '体征':
        return (has_direction or has_threshold) and any(
            term in name for term in _SIGN_CONTINUOUS_MEASURES
        ) or has_threshold and bool(re.search(r'(?:超高热|高热|发热|低热)', name))
    if category in ('实验室检查', '化验检查'):
        return has_direction or has_threshold
    return has_threshold or (
        has_direction and any(term in name for term in _TEST_CONTINUOUS_MEASURES)
    )



def _threshold_margin(threshold, unit):
    normalized_unit = _normalize_measurement_unit(unit)
    abs_threshold = abs(float(threshold))
    if normalized_unit == 'ph':
        return 0.01
    if normalized_unit in {'°c', '°f'} or 30 <= abs_threshold <= 45:
        return 0.1
    if abs_threshold < 1:
        return 0.01
    if normalized_unit in {'次/分', '次/min'} or abs_threshold >= 50:
        return 1.0
    return 0.1


_OBJECTIVE_CATEGORY_CANONICAL = {
    '体征': 'sign',
    '实验室检查': 'lab',
    '化验检查': 'lab',
    '影像检查': 'imaging',
    '影像': 'imaging',
    '功能检查': 'functional',
    '功能': 'functional',
}

_OBJECTIVE_METRIC_DOMAINS = {
    'spo2': (0.0, 100.0),
    'heart_rate': (20.0, 300.0),
    'systolic_bp': (40.0, 300.0),
    'diastolic_bp': (20.0, 200.0),
    'mean_arterial_pressure': (20.0, 250.0),
    'respiratory_rate': (3.0, 80.0),
    'temperature': (25.0, 45.0),
    'egfr': (0.0, 200.0),
    'fev1_fvc': (0.0, 1.0),
    'hba1c': (0.0, 25.0),
    'serum_creatinine': (0.0, 3000.0),
    'alt': (0.0, 10000.0),
    'hs_crp': (0.0, 1000.0),
    'blood_ph': (6.5, 8.0),
    'blood_pco2': (5.0, 200.0),
    'bicarbonate': (1.0, 80.0),
}

_OBJECTIVE_DEFAULT_UNITS = {
    'spo2': '%',
    'heart_rate': '次/分',
    'systolic_bp': 'mmhg',
    'diastolic_bp': 'mmhg',
    'mean_arterial_pressure': 'mmhg',
    'respiratory_rate': '次/分',
    'temperature': '°c',
    'egfr': 'ml/分/1.73m2',
    'fev1_fvc': '',
    'hba1c': '%',
    'alt': 'u/l',
    'hs_crp': 'mg/l',
    'blood_ph': '',
    'blood_pco2': 'mmhg',
    'bicarbonate': 'mmol/l',
}


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


def _objective_metric_key(name, category):
    """Return a conservative canonical key for an atomic objective measurement."""
    category_key = _OBJECTIVE_CATEGORY_CANONICAL.get(str(category or '').strip())
    if not category_key:
        return None
    text = re.sub(r'\s+', '', str(name or ''))
    folded = text.upper().replace('₂', '2').replace('₃', '3')
    matches = set()

    if category_key == 'sign':
        if any(term in text for term in ('脉搏短绌', '奔马律', '三音律', '奇脉')) \
                or _is_kussmaul_breathing(text):
            return None
        if _is_relative_blood_pressure_metric_text(text, folded):
            return None
        if '收缩压' in text or 'SBP' in folded:
            matches.add('systolic_bp')
        if '舒张压' in text or 'DBP' in folded:
            matches.add('diastolic_bp')
        if '平均动脉压' in text or 'MAP' in folded:
            matches.add('mean_arterial_pressure')
        if 'SPO2' in folded or '血氧饱和度' in text or '经皮氧饱和度' in text:
            matches.add('spo2')
        if any(term in text for term in ('心率', '心室率', '脉率', '心动过速', '心动过缓')):
            matches.add('heart_rate')
        if any(term in text for term in ('呼吸频率', '呼吸次数')):
            matches.add('respiratory_rate')
        if any(term in text for term in ('体温', '高热', '超高热', '低体温')):
            matches.add('temperature')
    elif category_key == 'lab':
        if ('PH' in folded and any(term in text for term in ('血', '动脉', '静脉'))) \
                or '血液酸碱度' in text \
                or text.startswith('酸碱度'):
            matches.add('blood_ph')
        if 'PACO2' in folded or 'PCO2' in folded or '二氧化碳分压' in text:
            matches.add('blood_pco2')
        if 'HCO3' in folded or any(term in text for term in ('碳酸氢根', '碳酸氢盐')):
            matches.add('bicarbonate')
        if 'EGFR' in folded or '估算肾小球滤过率' in text:
            matches.add('egfr')
        if 'HBA1C' in folded or '糖化血红蛋白' in text:
            matches.add('hba1c')
        if ('血清肌酐' in text or re.search(r'(?<!尿)(?<!白蛋白)肌酐(?:水平)?(?:升高|增高|降低|减低)', text)) \
                and not any(term in text for term in ('肌酐比', '肌酐比值', '肌酐清除率', '尿肌酐')):
            matches.add('serum_creatinine')
        has_alt = 'ALT' in folded or any(
            term in text for term in ('丙氨酸氨基转移酶', '谷丙转氨酶'))
        has_ast = 'AST' in folded or any(
            term in text for term in ('天冬氨酸氨基转移酶', '谷草转氨酶'))
        is_liver_enzyme_ratio = has_alt and has_ast and (
            '比值' in text or '比率' in text or '/' in text or '／' in text)
        if has_alt and not is_liver_enzyme_ratio:
            matches.add('alt')
        if 'HS-CRP' in folded or 'HSCRP' in folded or any(
                term in text for term in ('高敏C反应蛋白', '超敏C反应蛋白')):
            matches.add('hs_crp')
    elif category_key == 'functional':
        if re.search(r'FEV1\s*/\s*FVC', folded):
            matches.add('fev1_fvc')

    if len(matches) != 1:
        return None
    return f'{category_key}:{next(iter(matches))}'


def _objective_context_key(name, metric_key):
    """Separate the same analyte measured under clinically different conditions."""
    text = re.sub(r'\s+', '', str(name or ''))
    context = []
    context_groups = (
        (('静息', '安静状态', '休息状态'), 'rest'),
        (('初测', '首次测量', '首次测得', '第一次测量'), 'initial_measurement'),
        (('复测', '再测', '重复测量', '再次测量'), 'repeat_measurement'),
        (('治疗后', '处理后', '干预后'), 'post_treatment'),
        (('降压后', '降压治疗后'), 'post_antihypertensive'),
        (('运动后', '活动后', '运动负荷', '负荷后'), 'exercise'),
        (('吸氧后', '氧疗后', '给氧后'), 'on_oxygen'),
        (('未吸氧', '室内空气', '空气下'), 'room_air'),
        (('支气管舒张前', '舒张前'), 'pre_bronchodilator'),
        (('支气管舒张后', '舒张后'), 'post_bronchodilator'),
        (('仰卧位', '卧位'), 'supine'),
        (('站立位', '立位'), 'standing'),
        (('坐位',), 'sitting'),
        (('腋温',), 'axillary'),
        (('口温',), 'oral'),
        (('肛温', '直肠温'), 'rectal'),
        (('耳温',), 'tympanic'),
        (('额温',), 'forehead'),
    )
    for aliases, label in context_groups:
        if any(alias in text for alias in aliases):
            context.append(label)
    metric = str(metric_key or '').split(':')[-1]
    if metric in {'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure'}:
        if any(alias in text for alias in ('左臂', '左上肢')):
            context.append('left_arm')
        elif any(alias in text for alias in ('右臂', '右上肢')):
            context.append('right_arm')
    if metric in {'blood_ph', 'blood_pco2', 'bicarbonate'}:
        if '动脉' in text:
            context.append('arterial')
        elif '静脉' in text:
            context.append('venous')
    if metric == 'egfr':
        if '胱抑素C' in text or '胱抑素c' in text:
            context.append('cystatin_c_based')
        elif '基于肌酐' in text or '肌酐eGFR' in text:
            context.append('creatinine_based')
    return '+'.join(sorted(set(context))) or 'default'


def _objective_measurement_key(name, category):
    metric_key = _objective_metric_key(name, category)
    if not metric_key:
        return None
    return metric_key, _objective_context_key(name, metric_key)


def _objective_unit_key(source_name, suffix=''):
    unit = _normalize_measurement_unit(suffix)
    if not unit:
        match = next(_EXPLICIT_THRESHOLD_RE.finditer(str(source_name or '')), None)
        if match is not None:
            unit = _normalize_measurement_unit(match.group('unit') or '')
    unit = (unit.replace('毫米汞柱', 'mmhg')
            .replace('²', '2').replace('³', '3').replace('㎡', 'm2')
            .replace('m^2', 'm2').replace('m^3', 'm3')
            .replace('μ', 'u').replace('µ', 'u'))
    return unit.replace('次/min', '次/分').replace('/min', '/分').replace('bpm', '次/分')


def _predicate_unit_key(predicate, metric_key):
    unit = _objective_unit_key('', predicate[2])
    metric = str(metric_key or '').split(':')[-1]
    return unit or _OBJECTIVE_DEFAULT_UNITS.get(metric, '')


def _threshold_predicates(name, unit_hint=''):
    predicates = []
    seen = set()
    normalized_hint = _objective_unit_key('', unit_hint)
    for match in _EXPLICIT_THRESHOLD_RE.finditer(str(name or '')):
        unit = _objective_unit_key('', match.group('unit') or '')
        if normalized_hint and unit and normalized_hint != unit:
            continue
        operator = {'＞': '>', '＜': '<', '≥': '>=', '≤': '<='}.get(
            match.group('op'), match.group('op')
        )
        predicate = (operator, float(match.group('value')), match.group('unit') or unit_hint)
        if predicate not in seen:
            predicates.append(predicate)
            seen.add(predicate)
    inverse_operator = {
        '>': '<', '>=': '<=', '<': '>', '<=': '>=',
        '＞': '<', '≥': '<=', '＜': '>', '≤': '>=',
    }
    for match in _REVERSED_THRESHOLD_RE.finditer(str(name or '')):
        unit = _objective_unit_key('', match.group('unit') or '')
        if normalized_hint and unit and normalized_hint != unit:
            continue
        predicate = (
            inverse_operator[match.group('op')],
            float(match.group('value')),
            match.group('unit') or unit_hint,
        )
        if predicate not in seen:
            predicates.append(predicate)
            seen.add(predicate)
    return predicates


def _objective_predicates(name, metric_key, unit_hint=''):
    """Return explicit constraints plus conservative clinical defaults."""
    predicates = _threshold_predicates(name, unit_hint)
    if predicates:
        return predicates
    metric = str(metric_key or '').split(':')[-1]
    text = str(name or '')
    if metric == 'blood_ph':
        if '升高' in text or '碱血症' in text:
            return [('>=', 7.45, 'pH')]
        if '降低' in text or '酸血症' in text:
            return [('<=', 7.35, 'pH')]
    if metric == 'blood_pco2':
        unit = _normalize_measurement_unit(unit_hint)
        high, low, label = (6.0, 4.7, 'kPa') if unit == 'kpa' \
            else (45.0, 35.0, 'mmHg')
        if any(term in text for term in ('升高', '增高', '上升', '高碳酸血症')):
            return [('>=', high, label)]
        if any(term in text for term in ('降低', '减低', '下降', '低碳酸血症')):
            return [('<=', low, label)]
    if metric == 'bicarbonate':
        if any(term in text for term in ('升高', '增高', '上升')):
            return [('>=', 26.0, 'mmol/L')]
        if any(term in text for term in ('降低', '减低', '下降')):
            return [('<=', 22.0, 'mmol/L')]
    return []


def _apply_predicate_bounds(lower, upper, predicate, present=True):
    operator, threshold, unit = predicate
    margin = _threshold_margin(threshold, unit)
    if present:
        if operator == '>':
            lower = max(lower, threshold + margin)
        elif operator == '>=':
            lower = max(lower, threshold)
        elif operator == '<':
            upper = min(upper, threshold - margin)
        else:
            upper = min(upper, threshold)
    else:
        if operator == '>':
            upper = min(upper, threshold)
        elif operator == '>=':
            upper = min(upper, threshold - margin)
        elif operator == '<':
            lower = max(lower, threshold)
        else:
            lower = max(lower, threshold + margin)
    return lower, upper


def _metric_domain(metric_key):
    metric = str(metric_key or '').split(':')[-1]
    return _OBJECTIVE_METRIC_DOMAINS.get(metric, (-math.inf, math.inf))


def _interval_implies_predicate(lower, upper, predicate):
    operator, threshold, _ = predicate
    if operator == '>':
        return math.isfinite(lower) and lower > threshold
    if operator == '>=':
        return math.isfinite(lower) and lower >= threshold
    if operator == '<':
        return math.isfinite(upper) and upper < threshold
    return math.isfinite(upper) and upper <= threshold


def _repair_objective_positive_conflicts(state, patient_stage):
    """Keep the largest compatible objective phenotype subset, demoting only conflicts."""
    repairs = []
    category_by_key = {
        'signs': '体征',
        'lab_tests': '实验室检查',
        'imaging': '影像检查',
        'functional': '功能检查',
    }
    for state_key, category in category_by_key.items():
        bucket = state.get(state_key, {})
        grouped = {}
        for order, name in enumerate(bucket.get('positive', {})):
            measurement_key = _objective_measurement_key(name, category)
            if not measurement_key:
                continue
            metric_key = measurement_key[0]
            predicates = _objective_predicates(name, metric_key)
            if not predicates:
                continue
            lower, upper = _metric_domain(metric_key)
            for predicate in predicates:
                lower, upper = _apply_predicate_bounds(lower, upper, predicate, present=True)
            if lower <= upper:
                grouped.setdefault(measurement_key, []).append({
                    'name': name,
                    'lower': lower,
                    'upper': upper,
                    'probability': float(bucket.get('probabilities', {}).get(name, 0.5)),
                    'order': order,
                })

        for records in grouped.values():
            if len(records) < 2:
                continue
            if max(record['lower'] for record in records) <= \
                    min(record['upper'] for record in records):
                continue
            points = sorted({
                bound
                for record in records
                for bound in (record['lower'], record['upper'])
                if math.isfinite(bound)
            })
            candidates = []
            for point in points:
                kept = [
                    record for record in records
                    if record['lower'] <= point <= record['upper']
                ]
                score = (
                    len(kept),
                    sum(record['probability'] for record in kept),
                    -sum(record['order'] for record in kept),
                )
                candidates.append((score, kept))
            _, kept = max(candidates, key=lambda item: item[0])
            kept_names = {record['name'] for record in kept}
            for record in records:
                name = record['name']
                if name in kept_names:
                    continue
                bucket['positive'].pop(name, None)
                if name not in bucket['negative']:
                    bucket['negative'].append(name)
                repairs.append({
                    'category': category,
                    'name': name,
                    'from': 'positive',
                    'to': 'negative',
                    'reason': '同一测量条件下的客观方向互斥；以最少翻转并优先保留高概率表型',
                })
    return repairs


def _repair_objective_threshold_state(state, patient_stage):
    """Promote impossible negative threshold aliases while preserving every label."""
    repairs = []
    category_by_key = {
        'signs': '体征',
        'lab_tests': '实验室检查',
        'imaging': '影像检查',
        'functional': '功能检查',
    }
    for state_key, category in category_by_key.items():
        bucket = state.get(state_key, {})
        positives_by_measurement = {}
        for positive_name in bucket.get('positive', {}):
            measurement_key = _objective_measurement_key(positive_name, category)
            if measurement_key:
                positives_by_measurement.setdefault(measurement_key, []).append(positive_name)

        for negative_name in list(bucket.get('negative', [])):
            measurement_key = _objective_measurement_key(negative_name, category)
            if not measurement_key:
                continue
            metric_key = measurement_key[0]
            positive_names = positives_by_measurement.get(measurement_key, [])
            negative_predicates = _objective_predicates(negative_name, metric_key)
            if not positive_names or not negative_predicates:
                continue
            implied = False
            for negative_predicate in negative_predicates:
                negative_unit = _predicate_unit_key(negative_predicate, metric_key)
                lower, upper = _metric_domain(metric_key)
                positive_predicate_count = 0
                for positive_name in positive_names:
                    for predicate in _objective_predicates(positive_name, metric_key):
                        if _predicate_unit_key(predicate, metric_key) != negative_unit:
                            continue
                        lower, upper = _apply_predicate_bounds(
                            lower, upper, predicate, present=True
                        )
                        positive_predicate_count += 1
                if positive_predicate_count and lower <= upper and \
                        _interval_implies_predicate(lower, upper, negative_predicate):
                    implied = True
                else:
                    implied = False
                    break
            if not implied:
                continue
            bucket['negative'] = [name for name in bucket['negative'] if name != negative_name]
            fallback = (negative_name, patient_stage)
            bucket['positive'][negative_name] = bucket.get('payloads', {}).get(
                negative_name, fallback
            )
            repairs.append({
                'category': category,
                'name': negative_name,
                'from': 'negative',
                'to': 'positive',
                'reason': '已阳性的同指标阈值必然满足该条件',
            })
    return repairs


def _absent_predicates_by_metric(absent_by_category):
    result = {}
    for category, items in (absent_by_category or {}).items():
        for item in items or []:
            name = _phenotype_name(item)
            measurement_key = _objective_measurement_key(name, category)
            if not measurement_key:
                continue
            predicates = _objective_predicates(name, measurement_key[0])
            if predicates:
                result.setdefault(measurement_key, []).append((name, predicates))
    return result


def _quantified_objective_record(item, index):
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return None
    quantified_name = str(item[0]).strip()
    match = _STRICT_QUANTIFIED_NAME_RE.fullmatch(quantified_name)
    if not match:
        return None
    source_name, params, suffix = match.groups()
    tokens = [token.strip() for token in params.split(',')]
    if len(tokens) != 4:
        return None
    try:
        values = [None if token.lower() == 'nan' else float(token) for token in tokens]
    except ValueError:
        return None
    measurement_key = _objective_measurement_key(source_name, item[1])
    if not measurement_key:
        return None
    metric_key = measurement_key[0]
    unit_key = _objective_unit_key(source_name, suffix)
    return {
        'index': index,
        'source_name': source_name.strip(),
        'category': str(item[1]).strip(),
        'phase': str(item[2]).strip(),
        'time': str(item[3]).strip().upper(),
        'values': values,
        'suffix': suffix.strip(),
        'metric_key': metric_key,
        'measurement_key': measurement_key,
        'unit_key': unit_key,
        'group_key': (
            measurement_key, str(item[2]).strip(), str(item[3]).strip().upper(), unit_key
        ),
    }


def _harmonized_group_range(records, absent_predicates):
    metric_key = records[0]['metric_key']
    lower, upper = _metric_domain(metric_key)
    for record in records:
        for predicate in _objective_predicates(
                record['source_name'], metric_key, record['suffix']):
            lower, upper = _apply_predicate_bounds(lower, upper, predicate, present=True)
    means = sorted(record['values'][0] for record in records)
    preferred_mean = means[len(means) // 2]
    if records[0]['time'] in {'D0', 'H0'}:
        for _, predicates in absent_predicates.get(records[0]['measurement_key'], []):
            compatible = []
            incompatible_unit = False
            for predicate in predicates:
                predicate_unit = _predicate_unit_key(predicate, metric_key)
                if predicate_unit and records[0]['unit_key'] and \
                        predicate_unit != records[0]['unit_key']:
                    incompatible_unit = True
                    break
                compatible.append(predicate)
            if incompatible_unit or not compatible:
                continue
            alternatives = []
            for predicate in compatible:
                candidate_lower, candidate_upper = _apply_predicate_bounds(
                    lower, upper, predicate, present=False
                )
                if candidate_lower > candidate_upper:
                    continue
                contains_preferred = candidate_lower <= preferred_mean <= candidate_upper
                distance = 0.0 if contains_preferred else min(
                    abs(preferred_mean - candidate_lower),
                    abs(preferred_mean - candidate_upper),
                )
                alternatives.append((
                    (contains_preferred, -distance, candidate_upper - candidate_lower),
                    candidate_lower,
                    candidate_upper,
                ))
            if not alternatives:
                raise ValueError(f'同指标阳性/阴性阈值无可行交集: {metric_key}')
            _, lower, upper = max(alternatives, key=lambda item: item[0])
    if lower > upper:
        raise ValueError(f'同指标阳性/阴性阈值无可行交集: {metric_key}')

    finite_lowers = [record['values'][2] for record in records if record['values'][2] is not None]
    finite_uppers = [record['values'][3] for record in records if record['values'][3] is not None]
    preferred_lower = max([lower] + finite_lowers)
    preferred_upper = min([upper] + finite_uppers)
    if preferred_lower <= preferred_upper:
        range_lower, range_upper = preferred_lower, preferred_upper
    else:
        envelope_lower = min(finite_lowers) if finite_lowers else lower
        envelope_upper = max(finite_uppers) if finite_uppers else upper
        range_lower = max(lower, envelope_lower)
        range_upper = min(upper, envelope_upper)
        if range_lower > range_upper:
            range_lower, range_upper = lower, upper

    mean = means[len(means) // 2]
    if not math.isfinite(range_lower):
        range_lower = min([mean] + finite_lowers) - max(abs(mean) * 0.25, 1.0)
    if not math.isfinite(range_upper):
        range_upper = max([mean] + finite_uppers) + max(abs(mean) * 0.25, 1.0)
    if range_lower > range_upper:
        raise ValueError(f'同指标量化范围无可行交集: {metric_key}')
    mean = min(max(mean, range_lower), range_upper)
    sd_values = sorted(max(0.0, record['values'][1]) for record in records)
    sd = sd_values[len(sd_values) // 2]
    if range_upper == range_lower:
        sd = 0.0
    else:
        sd = min(sd, (range_upper - range_lower) / 3.0)
    return mean, sd, range_lower, range_upper


def _harmonize_objective_metric_ranges(quantified_items, absent_by_category=None):
    """Keep every phenotype label, but give aliases one feasible shared distribution."""
    items = [tuple(item) for item in (quantified_items or [])]
    records = [
        record for index, item in enumerate(items)
        if (record := _quantified_objective_record(item, index)) is not None
    ]
    groups = {}
    for record in records:
        groups.setdefault(record['group_key'], []).append(record)
    absent_predicates = _absent_predicates_by_metric(absent_by_category)

    for group in groups.values():
        has_absent_constraint = (
            group[0]['time'] in {'D0', 'H0'}
            and bool(absent_predicates.get(group[0]['measurement_key']))
        )
        if len(group) < 2 and not has_absent_constraint:
            continue
        mean, sd, lower, upper = _harmonized_group_range(group, absent_predicates)
        tokens = ','.join(_format_compact_number(value) for value in (
            mean, sd, lower, upper
        ))
        for record in group:
            candidate = f"{record['source_name']}{{{tokens}}}{record['suffix']}"
            validated_name = _validate_quantified_name(
                record['source_name'], candidate, record['category']
            )
            original = items[record['index']]
            items[record['index']] = (validated_name, *original[1:])
    return items


def _parse_quantified_name_parts(source_name, quantified_name):
    match = _STRICT_QUANTIFIED_NAME_RE.fullmatch(str(quantified_name).strip())
    if not match or match.group(1).strip() != str(source_name).strip():
        return None
    tokens = [token.strip() for token in match.group(2).split(',')]
    if len(tokens) != 4 or any(
        token.lower() != 'nan' and not _NUMBER_TOKEN_RE.fullmatch(token)
        for token in tokens
    ):
        return None
    values = [None if token.lower() == 'nan' else float(token) for token in tokens]
    return match.group(1), values, match.group(3).strip()


def _has_unitless_decimal_ratio_threshold(source_name):
    name = str(source_name or '')
    is_ratio = bool(_RATIO_TERM_RE.search(name)) or any(
        term in name.upper() for term in ('FEV1/FVC', 'FVC/FEV1')
    )
    return is_ratio and any(
        not item.group('unit') and abs(float(item.group('value'))) <= 1
        for item in _EXPLICIT_THRESHOLD_RE.finditer(name)
    )


def _is_unit_interval_ratio_name(source_name):
    name = str(source_name or '')
    return any(term in name for term in ('比例', '比率')) or any(
        term in name.upper() for term in ('FEV1/FVC', 'FVC/FEV1')
    )


def _maybe_normalize_percent_scale_for_decimal_threshold(source_name, values, suffix):
    thresholds = list(_EXPLICIT_THRESHOLD_RE.finditer(str(source_name)))
    if not thresholds or not _has_unitless_decimal_ratio_threshold(source_name) or \
            _normalize_measurement_unit(suffix) != '%':
        return values, suffix
    finite = [value for value in values if value is not None]
    if not finite or any(abs(value) > 100 for value in finite):
        return values, suffix
    if all(abs(value) <= 1 for value in finite):
        return values, ''
    return [None if value is None else value / 100.0 for value in values], ''


def _repair_quantified_name_threshold_bounds(source_name, quantified_name, category):
    parts = _parse_quantified_name_parts(source_name, quantified_name)
    if parts is None:
        return None
    prefix, values, suffix = parts
    values, suffix = _maybe_normalize_percent_scale_for_decimal_threshold(
        source_name, values, suffix
    )
    mean, sd, lower, upper = values
    if mean is None or sd is None or not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
        return None

    changed = False
    metric_key = _objective_metric_key(source_name, category)
    source_predicates = (
        _objective_predicates(source_name, metric_key, suffix)
        if metric_key else _threshold_predicates(source_name)
    )
    for operator, threshold, unit in source_predicates:
        margin = _threshold_margin(threshold, unit or suffix)
        if operator in ('>', '>='):
            bound = threshold + (margin if operator == '>' else 0.0)
            if lower is None or lower < bound:
                lower = bound
                changed = True
            if mean < lower:
                mean = lower
                changed = True
            if upper is not None and upper < mean:
                upper = mean
                changed = True
        else:
            bound = threshold - (margin if operator == '<' else 0.0)
            if upper is None or upper > bound:
                upper = bound
                changed = True
            if mean > upper:
                mean = upper
                changed = True
            if lower is not None and lower > mean:
                lower = mean
                changed = True

    if lower is not None and upper is not None and lower > upper:
        return None
    if not changed:
        return None
    repaired_tokens = [
        _format_compact_number(mean),
        _format_compact_number(sd),
        'nan' if lower is None else _format_compact_number(lower),
        'nan' if upper is None else _format_compact_number(upper),
    ]
    return f'{prefix}{{{",".join(repaired_tokens)}}}{suffix}'


def _validate_quantified_name(source_name, quantified_name, category):
    if category not in _QUANTIFIABLE_CATEGORIES:
        raise ValueError('2.7 只允许量化客观连续指标')
    if not _is_obviously_quantifiable((source_name, category, '', '')):
        raise ValueError('2.7 该表型不适合单值量化')
    match = _STRICT_QUANTIFIED_NAME_RE.fullmatch(str(quantified_name).strip())
    if not match or match.group(1).strip() != str(source_name).strip():
        raise ValueError('2.7 量化名称必须保留完整原名前缀')
    tokens = [token.strip() for token in match.group(2).split(',')]
    if len(tokens) != 4 or any(
        token.lower() != 'nan' and not _NUMBER_TOKEN_RE.fullmatch(token)
        for token in tokens
    ):
        raise ValueError('2.7 量化参数必须为 mean,sd,min,max 四值')
    if tokens[0].lower() == 'nan' or tokens[1].lower() == 'nan':
        raise ValueError('2.7 mean 和 sd 必须是有限数值')
    suffix = match.group(3).strip()
    values = [None if token.lower() == 'nan' else float(token) for token in tokens]
    normalized_values, normalized_suffix = _maybe_normalize_percent_scale_for_decimal_threshold(
        source_name, values, suffix
    )
    if normalized_values != values or normalized_suffix != suffix:
        values = normalized_values
        suffix = normalized_suffix
        tokens = [
            'nan' if value is None else _format_compact_number(value)
            for value in values
        ]
    mean, sd, lower, upper = values
    finite_values = [mean, sd] + [value for value in (lower, upper) if value is not None]
    if any(not math.isfinite(value) for value in finite_values) or sd < 0 or \
            (lower is not None and mean < lower) or \
            (upper is not None and mean > upper) or \
            (lower is not None and upper is not None and lower > upper):
        raise ValueError('2.7 量化参数范围不合法')
    metric_key = _objective_metric_key(source_name, category)
    source_predicates = (
        _objective_predicates(source_name, metric_key, suffix)
        if metric_key else _threshold_predicates(source_name)
    )
    for operator, threshold, _ in source_predicates:
        if operator in ('>', '>=', '≥'):
            valid = lower is not None and (
                lower > threshold if operator == '>' else lower >= threshold
            )
        else:
            valid = upper is not None and (
                upper < threshold if operator == '<' else upper <= threshold
            )
        if not valid:
            raise ValueError(
                f'2.7 量化范围不符合显式阈值 {operator}{threshold:g}'
            )
    if re.match(r'(?i)^10\^', suffix):
        suffix = f'×{suffix}'
    explicit_units = {
        _canonical_explicit_threshold_unit(match.group('unit'), source_name)
        for pattern in (_EXPLICIT_THRESHOLD_RE, _REVERSED_THRESHOLD_RE)
        for match in pattern.finditer(str(source_name))
        if match.group('unit')
    }
    suffix_unit = _canonical_explicit_threshold_unit(suffix, source_name)
    if explicit_units and suffix_unit not in explicit_units:
        raise ValueError(
            f'2.7 量化单位与原名显式阈值单位不一致: {suffix!r}'
        )
    if re.fullmatch(r'(?:级|分|等级|相对强度)', suffix):
        raise ValueError('2.7 离散分级不得伪装为连续数值')
    return f'{str(source_name).strip()}{{{",".join(tokens)}}}{suffix}'


def _validate_specific_timeline(quantified_items, specific_items):
    """校验 2.8 只将参数块替换为单个数值，不改写事件元数据。"""
    if not isinstance(quantified_items, list) or not isinstance(specific_items, list):
        raise ValueError('2.8 输入输出必须为列表')
    if len(quantified_items) != len(specific_items):
        raise ValueError('2.8 不得增删表型条目')
    for index, (quantified, specific) in enumerate(zip(quantified_items, specific_items)):
        if not isinstance(quantified, (list, tuple)) or len(quantified) != 4 or \
                not isinstance(specific, (list, tuple)) or len(specific) != 4:
            raise ValueError(f'2.8 条目 {index} 不是四元组')
        quantified_meta = tuple(str(value).strip() for value in quantified[1:])
        specific_meta = tuple(str(value).strip() for value in specific[1:])
        if quantified_meta != specific_meta:
            raise ValueError(f'2.8 条目 {index} 的类别/phase/时间被改写')

        quantified_name = str(quantified[0]).strip()
        specific_name = str(specific[0]).strip()
        match = _STRICT_QUANTIFIED_NAME_RE.fullmatch(quantified_name)
        if not match:
            if specific_name != quantified_name:
                raise ValueError(f'2.8 条目 {index} 的定性名称被改写')
            continue
        prefix, params, suffix = match.groups()
        tokens = [token.strip() for token in params.split(',')]
        value_match = re.fullmatch(
            re.escape(prefix) + r'；实际值=(' + _NUMBER_TOKEN_RE.pattern[1:-1] + ')' + re.escape(suffix),
            specific_name,
        )
        if not value_match:
            raise ValueError(f'2.8 条目 {index} 未正确将参数块替换为带语义分隔的实际数值')
        value = float(value_match.group(1))
        lower = None if tokens[2].lower() == 'nan' else float(tokens[2])
        upper = None if tokens[3].lower() == 'nan' else float(tokens[3])
        if (lower is not None and value < lower) or (upper is not None and value > upper):
            raise ValueError(f'2.8 条目 {index} 的采样值超出边界')
    return True


def _specific_value_from_name(source_name, suffix, specific_name):
    match = re.fullmatch(
        re.escape(source_name) + r'；实际值=(' + _NUMBER_TOKEN_RE.pattern[1:-1] + ')'
        + re.escape(suffix),
        str(specific_name).strip(),
    )
    if not match:
        raise ValueError(f'2.8 无法解析实际值: {specific_name!r}')
    return float(match.group(1))


_ACID_BASE_METRICS = {'blood_ph', 'blood_pco2', 'bicarbonate'}
_HENDERSON_HASSELBALCH_TOLERANCE = 0.02


def _acid_base_context_parts(record):
    """Return specimen and measurement-condition context for one blood-gas item."""
    context = set(str(record['measurement_key'][1] or '').split('+'))
    context.discard('default')
    specimen = 'unspecified'
    if 'arterial' in context:
        specimen = 'arterial'
    elif 'venous' in context:
        specimen = 'venous'
    context.discard('arterial')
    context.discard('venous')
    return specimen, tuple(sorted(context))


def _acid_base_complete_groups(records):
    """Yield unambiguous same-time pH/PaCO2/HCO3 record groups."""
    by_time = {}
    for record in records:
        metric = str(record['metric_key']).split(':')[-1]
        if metric in _ACID_BASE_METRICS:
            by_time.setdefault((record['phase'], record['time']), []).append(record)

    complete_groups = []
    for group in by_time.values():
        contexts = [_acid_base_context_parts(record) for record in group]
        explicit_specimens = {item[0] for item in contexts if item[0] != 'unspecified'}
        explicit_conditions = {item[1] for item in contexts if item[1]}
        if len(explicit_specimens) <= 1 and len(explicit_conditions) <= 1:
            candidates = [group]
        else:
            candidates = []
            for context in sorted(set(contexts)):
                if context[0] == 'unspecified' and not context[1]:
                    continue
                candidates.append([
                    record for record in group
                    if _acid_base_context_parts(record) == context
                ])
        for candidate in candidates:
            metrics = {
                str(record['metric_key']).split(':')[-1]
                for record in candidate
            }
            if _ACID_BASE_METRICS.issubset(metrics):
                complete_groups.append(candidate)
    return complete_groups


def _acid_base_unit_factor(metric, unit_key):
    """Return the multiplier from a displayed unit to the HH canonical unit."""
    unit = str(unit_key or _OBJECTIVE_DEFAULT_UNITS.get(metric, '')).lower()
    unit = (unit.replace('毫米汞柱', 'mmhg')
            .replace('·l⁻¹', '/l').replace('·l-1', '/l')
            .replace('mmol·l', 'mmol/l').replace('meq·l', 'meq/l'))
    if metric == 'blood_ph' and unit in {'', 'ph'}:
        return 1.0
    if metric == 'blood_pco2':
        if unit == 'mmhg':
            return 1.0
        if unit == 'kpa':
            return 7.50062
    if metric == 'bicarbonate' and unit in {'mmol/l', 'meq/l'}:
        return 1.0
    raise ValueError(
        f'Henderson-Hasselbalch 三联单位不受支持: {metric}={unit_key!r}'
    )


def _acid_base_group_state(group):
    """Build canonical ranges and actual values for one complete blood-gas group."""
    by_metric = {metric: [] for metric in _ACID_BASE_METRICS}
    for record in group:
        metric = str(record['metric_key']).split(':')[-1]
        by_metric[metric].append(record)

    state = {}
    for metric, metric_records in by_metric.items():
        canonical_records = []
        domain_lower, domain_upper = _metric_domain(metric)
        for record in metric_records:
            factor = _acid_base_unit_factor(metric, record['unit_key'])
            lower = domain_lower if record['values'][2] is None \
                else record['values'][2] * factor
            upper = domain_upper if record['values'][3] is None \
                else record['values'][3] * factor
            canonical_records.append((record, factor, lower, upper))
        lower = max(item[2] for item in canonical_records)
        upper = min(item[3] for item in canonical_records)
        if lower > upper or (metric != 'blood_ph' and lower <= 0):
            raise ValueError(
                f'Henderson-Hasselbalch 三联无可行量化区间: {metric}'
            )
        actuals = [record['actual'] * factor for record, factor, _, _ in canonical_records]
        actuals.sort()
        state[metric] = {
            'records': canonical_records,
            'lower': lower,
            'upper': upper,
            'actual': actuals[len(actuals) // 2],
        }
    return state


def _nearest_feasible_acid_base_values(state):
    """Project one sampled triad onto the bounded Henderson-Hasselbalch surface."""
    ph = state['blood_ph']
    pco2 = state['blood_pco2']
    hco3 = state['bicarbonate']
    feasible_ph_lower = max(
        ph['lower'],
        6.1 + math.log10(hco3['lower'] / (0.03 * pco2['upper'])),
    )
    feasible_ph_upper = min(
        ph['upper'],
        6.1 + math.log10(hco3['upper'] / (0.03 * pco2['lower'])),
    )
    if feasible_ph_lower > feasible_ph_upper + 1e-12:
        raise ValueError('Henderson-Hasselbalch 三联在给定范围内无解')

    chosen_ph = min(max(ph['actual'], feasible_ph_lower), feasible_ph_upper)
    ratio = 0.03 * (10 ** (chosen_ph - 6.1))
    pco2_lower = max(pco2['lower'], hco3['lower'] / ratio)
    pco2_upper = min(pco2['upper'], hco3['upper'] / ratio)
    if pco2_lower > pco2_upper + 1e-10:
        raise ValueError('Henderson-Hasselbalch 三联在给定范围内无解')

    candidates = {
        min(max(pco2['actual'], pco2_lower), pco2_upper),
        min(max(hco3['actual'] / ratio, pco2_lower), pco2_upper),
    }
    spans = {
        'blood_ph': max(ph['upper'] - ph['lower'], 0.02),
        'blood_pco2': max(pco2['upper'] - pco2['lower'], 1.0),
        'bicarbonate': max(hco3['upper'] - hco3['lower'], 1.0),
    }

    def score(candidate_pco2):
        candidate_hco3 = ratio * candidate_pco2
        return (
            ((chosen_ph - ph['actual']) / spans['blood_ph']) ** 2
            + ((candidate_pco2 - pco2['actual']) / spans['blood_pco2']) ** 2
            + ((candidate_hco3 - hco3['actual']) / spans['bicarbonate']) ** 2
        )

    chosen_pco2 = min(candidates, key=score)
    chosen_hco3 = ratio * chosen_pco2
    return {
        'blood_ph': chosen_ph,
        'blood_pco2': chosen_pco2,
        'bicarbonate': chosen_hco3,
    }


def _harmonize_acid_base_specific_values(quantified_items, specific_items):
    """Make each unambiguous blood-gas triad one bounded physiologic state."""
    _validate_specific_timeline(quantified_items, specific_items)
    result = [tuple(item) for item in (specific_items or [])]
    records = []
    for index, quantified in enumerate(quantified_items or []):
        record = _quantified_objective_record(quantified, index)
        if record is None:
            continue
        record['actual'] = _specific_value_from_name(
            record['source_name'], record['suffix'], result[index][0]
        )
        records.append(record)

    for group in _acid_base_complete_groups(records):
        state = _acid_base_group_state(group)
        values = _nearest_feasible_acid_base_values(state)
        for metric, metric_state in state.items():
            for record, factor, _, _ in metric_state['records']:
                displayed_value = values[metric] / factor
                original = result[record['index']]
                result[record['index']] = (
                    f"{record['source_name']}；实际值="
                    f"{_format_compact_number(displayed_value)}{record['suffix']}",
                    *original[1:],
                )
    _validate_specific_timeline(quantified_items, result)
    return result


def _validate_acid_base_consistency(records, tolerance=_HENDERSON_HASSELBALCH_TOLERANCE):
    for group in _acid_base_complete_groups(records):
        state = _acid_base_group_state(group)
        for metric, metric_state in state.items():
            canonical_values = [
                record['actual'] * factor
                for record, factor, _, _ in metric_state['records']
            ]
            alias_tolerance = tolerance if metric == 'blood_ph' else 0.2
            if max(canonical_values) - min(canonical_values) > alias_tolerance:
                raise ValueError(
                    f'2.8 同一血气指标换算后实际值不一致: {metric}'
                )
        ph = state['blood_ph']['actual']
        pco2 = state['blood_pco2']['actual']
        hco3 = state['bicarbonate']['actual']
        expected_ph = 6.1 + math.log10(hco3 / (0.03 * pco2))
        if abs(ph - expected_ph) > tolerance:
            raise ValueError(
                '2.8 Henderson-Hasselbalch 血气三联不一致: '
                f'pH={ph:g}, PaCO2={pco2:g}, HCO3={hco3:g}'
            )


def _predicate_is_true(value, predicate):
    operator, threshold, _ = predicate
    if operator == '>':
        return value > threshold
    if operator == '>=':
        return value >= threshold
    if operator == '<':
        return value < threshold
    return value <= threshold


def _validate_objective_value_consistency(quantified_items, specific_items,
                                          absent_by_category=None):
    """Validate shared aliases and ensure actual values do not satisfy absent predicates."""
    _validate_specific_timeline(quantified_items, specific_items)
    records = []
    for index, quantified in enumerate(quantified_items or []):
        record = _quantified_objective_record(quantified, index)
        if record is None:
            continue
        record['actual'] = _specific_value_from_name(
            record['source_name'], record['suffix'], specific_items[index][0]
        )
        records.append(record)

    groups = {}
    for record in records:
        groups.setdefault(record['group_key'], []).append(record)
    absent_predicates = _absent_predicates_by_metric(absent_by_category)
    for group in groups.values():
        first_value = group[0]['actual']
        if any(not math.isclose(record['actual'], first_value, rel_tol=0.0, abs_tol=1e-12)
               for record in group[1:]):
            raise ValueError(f"2.8 同指标同时点实际值不一致: {group[0]['metric_key']}")
        if group[0]['time'] not in {'D0', 'H0'}:
            continue
        for absent_name, predicates in absent_predicates.get(group[0]['measurement_key'], []):
            predicate_units = [
                _predicate_unit_key(predicate, group[0]['metric_key'])
                for predicate in predicates
            ]
            if any(
                unit and group[0]['unit_key'] and unit != group[0]['unit_key']
                for unit in predicate_units
            ):
                continue
            if predicates and all(
                _predicate_is_true(first_value, predicate)
                for predicate in predicates
            ):
                raise ValueError(
                    f"2.8 实际值命中阴性阈值: {absent_name}"
                )
    _validate_acid_base_consistency(records)
    return True


def _specific_values_with_shared_objective_values(quantified_items):
    """Sample once per canonical objective metric, time point and unit."""
    items = [tuple(item) for item in (quantified_items or [])]
    records_by_index = {}
    groups = {}
    for index, item in enumerate(items):
        record = _quantified_objective_record(item, index)
        if record is None:
            continue
        records_by_index[index] = record
        groups.setdefault(record['group_key'], []).append(record)

    realized_by_index = {}
    for group in groups.values():
        lower = max(
            record['values'][2] if record['values'][2] is not None else -math.inf
            for record in group
        )
        upper = min(
            record['values'][3] if record['values'][3] is not None else math.inf
            for record in group
        )
        if lower > upper:
            raise ValueError(f"2.8 同指标量化范围无交集: {group[0]['metric_key']}")
        means = sorted(record['values'][0] for record in group)
        mean = min(max(means[len(means) // 2], lower), upper)
        sds = sorted(max(0.0, record['values'][1]) for record in group)
        sd = sds[len(sds) // 2]
        if math.isfinite(lower) and math.isfinite(upper):
            sd = 0.0 if lower == upper else min(sd, (upper - lower) / 3.0)
        representative = group[0]
        params = ','.join((
            _format_compact_number(mean),
            _format_compact_number(sd),
            'nan' if not math.isfinite(lower) else _format_compact_number(lower),
            'nan' if not math.isfinite(upper) else _format_compact_number(upper),
        ))
        synthetic = (
            f"{representative['source_name']}{{{params}}}{representative['suffix']}"
        )
        sampled = _quantize_values(synthetic)
        prefix = f"{representative['source_name']}；实际值="
        if not sampled.startswith(prefix) or (
                representative['suffix'] and not sampled.endswith(representative['suffix'])):
            raise ValueError(f'2.8 共享采样结果无法解析: {sampled!r}')
        value_text = sampled[len(prefix):]
        if representative['suffix']:
            value_text = value_text[:-len(representative['suffix'])]
        for record in group:
            realized_by_index[record['index']] = (
                f"{record['source_name']}；实际值={value_text}{record['suffix']}"
            )

    specific_items = []
    for index, item in enumerate(items):
        name = realized_by_index.get(index)
        if name is None:
            name = _quantize_values(str(item[0]))
        specific_items.append((name, *item[1:]))
    return specific_items


def _validated_existing_specific_timeline(quantified_items, specific_text,
                                          absent_by_category=None):
    """返回通过当前 2.8 格式与守恒校验的具体值列表，否则返回 None。"""
    specific_items = parse_list_from_response(specific_text or '')
    try:
        _validate_objective_value_consistency(
            quantified_items, specific_items, absent_by_category
        )
    except ValueError:
        return None
    return [tuple(str(value).strip() for value in item) for item in specific_items]


def _infer_threshold_suffix(source_name, threshold_match):
    unit = (threshold_match.group('unit') or '').strip()
    if unit:
        normalized = _normalize_measurement_unit(unit)
        if _canonical_explicit_threshold_unit(unit, source_name) == 'agatston_unit':
            return 'AU'
        if normalized == '°c':
            return '℃'
        if normalized == '次/min':
            return '次/分'
        return unit
    name = str(source_name or '')
    threshold = abs(float(threshold_match.group('value')))
    if any(term in name for term in ('体温', '发热', '高热', '低热')) or 30 <= threshold <= 45:
        return '℃'
    if any(term in name for term in ('心率', '心动', '脉率', '呼吸频率', '呼吸急促')):
        return '次/分'
    if any(term in name.upper() for term in ('收缩压', '舒张压', '平均动脉压', 'SBP', 'DBP', 'MAP')):
        return 'mmHg'
    if any(term in name for term in ('血氧饱和度', 'SpO2', 'SPO2', '百分比')):
        return '%'
    return ''


def _fallback_quantified_name_for_threshold(source_item):
    """Build a conservative deterministic 2.7 value range for omitted thresholded items.

    This is only a safety net for objectively quantifiable phenotypes that already
    carry an explicit threshold.  It preserves the original phenotype name and lets
    _validate_quantified_name enforce category/unit/boundary correctness.
    """
    if not isinstance(source_item, (list, tuple)) or len(source_item) != 4:
        return None
    source_name, category = str(source_item[0]).strip(), str(source_item[1]).strip()
    if not source_name or not _is_obviously_quantifiable(source_item):
        return None
    threshold_match = next(_EXPLICIT_THRESHOLD_RE.finditer(source_name), None)
    if threshold_match is None:
        metric_key = _objective_metric_key(source_name, category)
        if not metric_key or not metric_key.endswith(':blood_ph'):
            return None
        lower, upper = _metric_domain(metric_key)
        predicates = _objective_predicates(source_name, metric_key)
        for predicate in predicates:
            lower, upper = _apply_predicate_bounds(lower, upper, predicate, present=True)
        lower, upper = max(lower, 6.8), min(upper, 7.8)
        if lower > upper:
            return None
        target = 7.30 if upper < 7.35 else 7.50
        mean = min(max(target, lower), upper)
        sd = min(0.03, max(0.0, (upper - lower) / 6.0))
        candidate = (
            f'{source_name}{{{_format_compact_number(mean)},'
            f'{_format_compact_number(sd)},{_format_compact_number(lower)},'
            f'{_format_compact_number(upper)}}}'
        )
        return _validate_quantified_name(source_name, candidate, category)

    operator = {'＞': '>', '＜': '<', '≥': '>=', '≤': '<='}.get(
        threshold_match.group('op'), threshold_match.group('op')
    )
    threshold = float(threshold_match.group('value'))
    suffix = _infer_threshold_suffix(source_name, threshold_match)
    margin = _threshold_margin(threshold, threshold_match.group('unit') or suffix)
    normalized_suffix = _normalize_measurement_unit(suffix)

    if normalized_suffix in {'°c', '°f'} or any(term in source_name for term in ('体温', '发热', '高热', '低热')):
        if operator in ('>', '>='):
            lower = threshold + (margin if operator == '>' else 0.0)
            upper = max(lower + margin, min(42.5, max(threshold + 2.5, lower + 1.0)))
        else:
            upper = threshold - (margin if operator == '<' else 0.0)
            lower = min(upper - margin, max(30.0, upper - 3.0))
    elif _is_unit_interval_ratio_name(source_name) and threshold <= 1:
        if operator in ('>', '>='):
            lower = max(0.0, threshold + (margin if operator == '>' else 0.0))
            upper = 1.0
        else:
            upper = min(1.0, threshold - (margin if operator == '<' else 0.0))
            lower = 0.0
        if lower > upper:
            return None
    elif normalized_suffix in {'%', '％'} or any(term in source_name for term in ('血氧饱和度', 'SpO2', 'SPO2', '百分比')):
        if operator in ('>', '>='):
            lower = threshold + (margin if operator == '>' else 0.0)
            upper = max(lower + margin, min(100.0, max(threshold + 20.0, lower + 5.0)))
        else:
            upper = threshold - (margin if operator == '<' else 0.0)
            lower = max(0.0, min(upper - margin, upper - max(5.0, abs(threshold) * 0.25)))
    elif any(term in source_name for term in ('心率', '心动', '脉率', '呼吸频率', '呼吸急促')):
        if operator in ('>', '>='):
            lower = threshold + (margin if operator == '>' else 0.0)
            upper = lower + max(10.0, abs(threshold) * 0.25)
        else:
            upper = threshold - (margin if operator == '<' else 0.0)
            lower = max(0.0, min(upper - margin, upper - max(10.0, abs(threshold) * 0.25)))
    else:
        span = max(margin * 10.0, abs(threshold) * 0.25, 1.0)
        if operator in ('>', '>='):
            lower = threshold + (margin if operator == '>' else 0.0)
            upper = lower + span
        else:
            upper = threshold - (margin if operator == '<' else 0.0)
            lower = max(0.0, min(upper - margin, upper - span))

    if lower > upper:
        return None
    mean = lower + (upper - lower) * 0.45
    sd = max((upper - lower) / 6.0, margin)
    candidate = f'{source_name}{{{_format_compact_number(mean)},{_format_compact_number(sd)},{_format_compact_number(lower)},{_format_compact_number(upper)}}}{suffix}'
    try:
        return _validate_quantified_name(source_name, candidate, category)
    except ValueError:
        repaired = _repair_quantified_name_threshold_bounds(
            source_name, candidate, category
        )
        if repaired is None:
            return None
        try:
            return _validate_quantified_name(source_name, repaired, category)
        except ValueError:
            return None


def _parse_quantification_updates(source_items, response_text, allowed_indices):
    """Parse one 2.7 batch and require exactly one update per eligible index."""
    if not isinstance(source_items, list):
        raise ValueError('2.7 源数据必须为列表')
    if any(not isinstance(item, (list, tuple)) or len(item) != 4 for item in source_items):
        raise ValueError('2.7 源数据含非四元组')
    text = str(response_text or '').strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError('2.7 输出必须是单一严格 JSON 对象') from exc
    if not isinstance(payload, dict) or set(payload) != {'schema_version', 'updates'}:
        raise ValueError('2.7 JSON 顶层字段不合法')
    if type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or \
            not isinstance(payload.get('updates'), list):
        raise ValueError('2.7 JSON 版本或 updates 不合法')

    allowed_indices = set(allowed_indices)
    seen = set()
    validated_updates = []
    for update in payload['updates']:
        if not isinstance(update, dict) or set(update) != {'source_index', 'quantified_name'}:
            raise ValueError('2.7 update 字段不合法')
        source_index = update['source_index']
        if type(source_index) is not int or source_index in seen:
            raise ValueError('2.7 source_index 必须是唯一整数')
        if source_index < 0 or source_index >= len(source_items):
            raise ValueError('2.7 source_index 越界')
        if source_index not in allowed_indices:
            raise ValueError('2.7 返回了当前批次之外或不可量化的 source_index')
        seen.add(source_index)

        source = source_items[source_index]
        quantified_name = update['quantified_name']
        if not isinstance(quantified_name, str):
            raise ValueError('2.7 quantified_name 必须是字符串')
        try:
            validated_name = _validate_quantified_name(source[0], quantified_name, source[1])
        except ValueError as exc:
            if '显式阈值' not in str(exc):
                raise
            repaired_name = _repair_quantified_name_threshold_bounds(
                source[0], quantified_name, source[1]
            )
            if not repaired_name:
                raise
            validated_name = _validate_quantified_name(source[0], repaired_name, source[1])
        validated_updates.append({
            'source_index': source_index,
            'quantified_name': validated_name,
        })

    missing = sorted(allowed_indices - seen)
    fallback_updates = []
    for source_index in missing:
        fallback_name = _fallback_quantified_name_for_threshold(source_items[source_index])
        if not fallback_name:
            raise ValueError(f'2.7 遗漏了明确可量化的条目: {missing}')
        fallback_updates.append({
            'source_index': source_index,
            'quantified_name': fallback_name,
        })
    if fallback_updates:
        print(f"    [模块2.7] ⚠️ LLM遗漏 {len(fallback_updates)} 个阈值型客观表型，已用本地保守范围补齐")
        validated_updates.extend(fallback_updates)
    return validated_updates


def _parse_quantification_patch(source_items, response_text):
    """对 2.7 的 JSON patch 做严格校验，并由程序重建四元组。"""
    required = {
        index for index, item in enumerate(source_items)
        if _is_obviously_quantifiable(item)
    }
    updates = _parse_quantification_updates(source_items, response_text, required)
    rebuilt = [tuple(str(value).strip() for value in item) for item in source_items]
    for update in updates:
        source_index = update['source_index']
        source = rebuilt[source_index]
        rebuilt[source_index] = (update['quantified_name'], *source[1:])

    return rebuilt


def module_2_7_quantify_values(seed_text, patient_stage, corrected_text, tag="模块2.7",
                               absent_by_category=None):
    """Quantify eligible phenotypes in bounded batches without dropping rich context."""
    source_items = parse_list_from_response(corrected_text)
    if not source_items:
        prompt = f"（患者 {seed_text} 无需量化的表型）"
        response = '{"schema_version":1,"updates":[]}'
        return prompt, response, []

    indexed_items = [
        {'source_index': index, 'name': item[0], 'category': item[1]}
        for index, item in enumerate(source_items)
        if _is_obviously_quantifiable(item)
    ]
    if not indexed_items:
        prompt = f"（患者 {seed_text} 没有需要量化的客观表型）"
        response = '{"schema_version":1,"updates":[]}'
        return prompt, response, [tuple(item) for item in source_items]

    prompts = []
    all_updates = []
    total_batches = math.ceil(len(indexed_items) / M2_QUANTIFICATION_BATCH_SIZE)
    t0 = time.time()
    for batch_no, start in enumerate(
            range(0, len(indexed_items), M2_QUANTIFICATION_BATCH_SIZE), start=1):
        batch = indexed_items[start:start + M2_QUANTIFICATION_BATCH_SIZE]
        allowed_indices = {item['source_index'] for item in batch}
        prompt = f"""你是临床医学专家。请为以下已经筛选出的可量化客观表型生成有界正态分布参数。

患者信息：{seed_text}
疾病分期：{patient_stage}
当前批次：{batch_no}/{total_batches}
候选条目（source_index 为 0 起始的稳定索引）：
{json.dumps(batch, ensure_ascii=False)}

只输出一个 JSON 对象：
{{"schema_version":1,"updates":[{{"source_index":0,"quantified_name":"完整原名{{mean,sd,min,max}}单位"}}]}}

要求：
1. 上方条目都已经过程序筛选，每个 source_index 必须且只能出现一次，不得遗漏。
2. 可量化类别包括体征、实验室/化验、影像和功能检查；不要改写类别或 source_index。
3. quantified_name 必须以完整原名开头，紧接 `{{mean,sd,min,max}}`和单位；无量纲指标可不写单位。
4. mean/sd 必须为数值且 sd>=0；min/max 无边界时填 nan，否则必须满足 min<=mean<=max。
5. 不得拆分、合并或遗漏原条目。候选中只有单一连续指标；不得用一个连续值代替复合指标、定性体征或离散分级。
6. 若原名含显式阈值，边界必须完全落在阈值定义的异常侧：如 `>38℃` 要求 min>38，`≥38℃` 要求 min≥38，`<0.70` 要求 max<0.70，`≤50%` 要求 max≤50%。
7. 对 FEV1/FVC、比值、比例等无量纲小数阈值，不要改成百分数；应使用 0.x 小数尺度且不写 %。
8. 数值之间应符合基本病理生理关系，且严重度与 `{patient_stage}` 一致；在此约束下尽量保留表型信息的丰富性，不因量化而删减条目。
"""
        prompts.append(prompt)
        print(f"    [{tag}] 正在量化数值型表型（批次 {batch_no}/{total_batches}，"
              f"{len(batch)} 项）...")
        response = call_gpt5(prompt, tag=f"{tag}-B{batch_no}")
        try:
            all_updates.extend(_parse_quantification_updates(
                source_items, response, allowed_indices
            ))
        except ValueError as exc:
            print(f"    [{tag}] ⚠️ 第 {batch_no} 批量化输出校验失败: {exc}")
            return None

    combined_response = json.dumps({
        'schema_version': 1,
        'updates': all_updates,
    }, ensure_ascii=False)
    try:
        quantified_items = _parse_quantification_patch(source_items, combined_response)
        quantified_items = _harmonize_objective_metric_ranges(
            quantified_items, absent_by_category
        )
    except ValueError as exc:
        print(f"    [{tag}] ⚠️ 合并后的量化输出校验失败: {exc}")
        return None
    print(f"    [{tag}] {total_batches} 批量化完成，总用时 {round(time.time()-t0,1)}s")
    return '\n\n'.join(prompts), combined_response, quantified_items


def module_2_8_specific_values(seed_text, quantified_text, tag="模块2.8",
                                 preset_age=None, preset_gender='',
                                 absent_by_category=None):
    """
    模块2.8 具体数值赋值 & seed解析（无GPT调用，纯本地计算）。
    若提供 preset_age / preset_gender（例如由模块2.2已采样确定），则优先使用它们，
    避免多次随机造成 age 不一致。
    """
    t0 = time.time()
    age_min, age_max, gender, diagnosis, acuity = _parse_seed(seed_text)
    if preset_age is not None:
        try:
            age = int(preset_age)
        except (ValueError, TypeError):
            age = random.randint(age_min, age_max)
    else:
        age = random.randint(age_min, age_max)
    if preset_gender:
        gender = preset_gender
    quantified_items = parse_list_from_response(quantified_text or '')
    specific_items = _specific_values_with_shared_objective_values(quantified_items)
    specific_items = _harmonize_acid_base_specific_values(
        quantified_items, specific_items
    )
    _validate_objective_value_consistency(
        quantified_items, specific_items, absent_by_category
    )
    specific_text = repr(specific_items)
    print(f"    [{tag}] 年龄={age}岁, 性别={gender}, 具体化用时 {round(time.time()-t0,3)}s")
    return age, gender, diagnosis, specific_text


def module_2_9_derive_chief_complaint(*args, **kwargs):
    """模块2.9：从已冻结的症状时间轴中提取本次主诉。"""
    return _derive_chief_complaint(*args, **kwargs)


def _sample_complication_phenotypes(comorbidity_name, complication_phenotypes_text,
                                    patient_gender=''):
    """
    从 module_1_51 生成的4元组列表中，过滤出属于 comorbidity_name 的条目，
    按概率 Bernoulli 采样，返回四类阳性列表和四类阴性名称列表。
    """
    _TYPE_MAP = {
        '体征': 'signs', '阳性体征': 'signs',
        '实验室': 'labs', '实验室检查': 'labs', '化验': 'labs',
        '影像': 'imaging', '影像检查': 'imaging',
        '功能': 'functional', '功能检查': 'functional',
    }
    signs, labs, imaging, functional = [], [], [], []
    absent_signs, absent_labs, absent_imaging, absent_functional = [], [], [], []
    if not complication_phenotypes_text or complication_phenotypes_text.strip() in ('', '[]'):
        return (signs, labs, imaging, functional,
                absent_signs, absent_labs, absent_imaging, absent_functional)

    entries = parse_list_from_response(complication_phenotypes_text)
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            continue
        name_c, ptype, pname, prob = entry[0], entry[1], entry[2], entry[3]
        if str(name_c).strip() != comorbidity_name.strip():
            continue
        pname = _personalize_phenotype_name_for_gender(pname, patient_gender)
        if pname is None:
            continue
        try:
            p = float(prob)
        except (TypeError, ValueError):
            p = 0.0
        bucket = _TYPE_MAP.get(str(ptype).strip(), '')
        if not bucket:
            continue
        if random.random() < p:
            item = (str(pname), '伴随疾病')
            if bucket == 'signs':
                signs.append(item)
            elif bucket == 'labs':
                labs.append(item)
            elif bucket == 'imaging':
                imaging.append(item)
            elif bucket == 'functional':
                functional.append(item)
        else:
            name = str(pname)
            if bucket == 'signs':
                absent_signs.append(name)
            elif bucket == 'labs':
                absent_labs.append(name)
            elif bucket == 'imaging':
                absent_imaging.append(name)
            elif bucket == 'functional':
                absent_functional.append(name)
    return (signs, labs, imaging, functional,
            absent_signs, absent_labs, absent_imaging, absent_functional)


# ============================================================
# 单患者 Worker（模块2全流程）
# ============================================================

def _generate_single_patient_m2(args):
    """
    线程池Worker：为单个患者运行模块2全流程
    流程：2.0 → 侧别 → acuity → 2.1（双轨） → 2.2（合并症注入）
          → 2.3（就诊状态） → 2.4（全模态收敛） → 2.5（总病程/症状时序）
          → 2.6（仅就诊过） → 2.7（数值分布） → 2.8（具体值） → 2.9（主诉）
    将结果写入 update_queue 并返回 row dict
    """
    (patient_idx, seed_text, symptoms_text, signs_text,
     lab_tests_text, imaging_text, functional_tests_text,
     comorbidities_text,
     staging_system, patient_stage,
     update_queue, existing_row,
     laterality_type, complication_phenotypes_text) = args

    tag_prefix = f"患者{patient_idx}"
    worker_start = time.time()
    print(f"\n  --- 模拟{tag_prefix}（模块2）---")

    if existing_row:
        row = _initialize_patient_row_identity(patient_idx, existing_row)
        print(f"    [{tag_prefix}] 📂 加载已有数据，检查续接点...")
    else:
        row = _initialize_patient_row_identity(patient_idx)
    row['_patient_idx'] = patient_idx
    row[COL_SEED] = seed_text
    patient_age, patient_gender = _resolve_patient_demographics(row, seed_text)
    if not row.get(COL_STAGE, '').strip():
        row[COL_STAGE] = patient_stage

    def _push_update():
        try:
            update_queue.put((patient_idx, dict(row)))
        except Exception:
            pass

    try:
        _err = 'ERROR'
        has_21 = all(
            bool(row.get(col, '').strip()) and _err not in row.get(col, '')
            for col in (
                COL_SYMPTOMS, COL_SIGNS, COL_LAB_TESTS, COL_IMAGING,
                COL_FUNCTIONAL_TESTS, COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS,
                COL_ABSENT_LAB_TESTS, COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL,
            )
        )
        has_correlation = False
        has_timeline = False
        has_comorbidities = bool(row.get(COL_COMORBIDITIES, '').strip())
        has_quantified = bool(row.get(COL_QUANTIFIED, '').strip())
        has_specific = bool(row.get(COL_SPECIFIC, '').strip())
        has_chief = bool(row.get(COL_CHIEF_COMPLAINT, '').strip())
        # v4 新增的续跑判断
        has_acuity = bool(row.get(COL_ACUITY, '').strip())
        has_prior_status = bool(row.get(COL_PRIOR_VISITED, '').strip())
        has_prior_history = bool(row.get(COL_PRIOR_VISIT_HISTORY, '').strip())

        num_levels = _get_num_levels(staging_system)
        level_names = _get_level_names(staging_system)
        patient_stage_val = _validate_patient_stage_name(row.get(COL_STAGE, ''), level_names)
        patient_level_index = level_names.index(patient_stage_val)
        print(f"    [{tag_prefix}-2.0] 疾病级别: {patient_stage_val} (索引{patient_level_index}/{num_levels}级)")

        # ---- 模块1.02 侧别赋值 ----
        fixed_laterality = fixed_anatomic_laterality(f"{row.get(COL_DIAGNOSIS) or ''} {seed_text}")
        if fixed_laterality:
            if str(row.get(COL_PATIENT_LATERALITY, '') or '').strip() != fixed_laterality:
                row[COL_PATIENT_LATERALITY] = fixed_laterality
                _push_update()
        elif not row.get(COL_PATIENT_LATERALITY, '').strip():
            lat_type = (laterality_type or 'either').strip().lower()
            if lat_type == 'bilateral':
                row[COL_PATIENT_LATERALITY] = '双侧'
            elif lat_type == 'unilateral':
                row[COL_PATIENT_LATERALITY] = random.choice(['左侧', '右侧'])
            else:
                row[COL_PATIENT_LATERALITY] = random.choices(
                    ['左侧', '右侧', '双侧'], weights=[0.35, 0.35, 0.30])[0]

        # ---- 写 acuity 列（来自 _parse_seed） ----
        if has_acuity:
            acuity_val = (row.get(COL_ACUITY, '') or '').strip() or ACUITY_ACUTE
        else:
            try:
                _, _, _, _, acuity_val = _parse_seed(seed_text)
            except Exception:
                acuity_val = ACUITY_ACUTE
            acuity_val = (acuity_val or ACUITY_ACUTE).strip() or ACUITY_ACUTE
            row[COL_ACUITY] = acuity_val
            _push_update()
        if acuity_val not in _VALID_ACUITIES:
            raise ValueError(f'非法急慢性类型: {acuity_val!r}')

        # ---- 模块2.1（双轨采样：selected + absent）----
        if has_21:
            print(f"    [{tag_prefix}-2.1] ⏭️  已有采样数据，跳过")
            selected_symptoms = parse_list_from_response(row[COL_SYMPTOMS])
            selected_signs = parse_list_from_response(row[COL_SIGNS])
            selected_lab_tests = parse_list_from_response(row[COL_LAB_TESTS])
            selected_imaging = parse_list_from_response(row[COL_IMAGING])
            selected_functional = parse_list_from_response(row[COL_FUNCTIONAL_TESTS])
            absent_symptoms = parse_list_from_response(row.get(COL_ABSENT_SYMPTOMS, '') or '')
            absent_signs = parse_list_from_response(row.get(COL_ABSENT_SIGNS, '') or '')
            absent_lab_tests = parse_list_from_response(row.get(COL_ABSENT_LAB_TESTS, '') or '')
            absent_imaging = parse_list_from_response(row.get(COL_ABSENT_IMAGING, '') or '')
            absent_functional = parse_list_from_response(row.get(COL_ABSENT_FUNCTIONAL, '') or '')
        else:
            t_step = time.time()
            (selected_symptoms, selected_signs, selected_lab_tests,
             selected_imaging, selected_functional,
             absent_symptoms, absent_signs, absent_lab_tests,
            absent_imaging, absent_functional) = module_2_1_sample_phenotypes(
                symptoms_text, signs_text, lab_tests_text, imaging_text, functional_tests_text,
                patient_level_index, num_levels, patient_stage=patient_stage_val,
                patient_gender=patient_gender,
            )
            print(f"    [{tag_prefix}-2.1] 采样({patient_stage_val}): "
                  f"症状{len(selected_symptoms)}/未中{len(absent_symptoms)}, "
                  f"体征{len(selected_signs)}/未中{len(absent_signs)}, "
                  f"实验室{len(selected_lab_tests)}/未中{len(absent_lab_tests)}, "
                  f"影像{len(selected_imaging)}/未中{len(absent_imaging)}, "
                  f"功能{len(selected_functional)}/未中{len(absent_functional)}，"
                  f"用时 {round(time.time()-t_step,2)}s")
            row[COL_SYMPTOMS] = str(selected_symptoms)
            row[COL_SIGNS] = str(selected_signs)
            row[COL_LAB_TESTS] = str(selected_lab_tests)
            row[COL_IMAGING] = str(selected_imaging)
            row[COL_FUNCTIONAL_TESTS] = str(selected_functional)
            row[COL_ABSENT_SYMPTOMS] = str(list(absent_symptoms))
            row[COL_ABSENT_SIGNS] = str(list(absent_signs))
            row[COL_ABSENT_LAB_TESTS] = str(list(absent_lab_tests))
            row[COL_ABSENT_IMAGING] = str(list(absent_imaging))
            row[COL_ABSENT_FUNCTIONAL] = str(list(absent_functional))
            _push_update()

        # ---- 模块2.2：先注入合并症表型，再进入全模态相关性修正 ----
        if has_comorbidities:
            print(f"    [{tag_prefix}-2.2] ⏭️  已有伴随疾病数据，跳过")
            selected_comorbidities = [
                item if isinstance(item, str) else item[0]
                for item in parse_list_from_response(row[COL_COMORBIDITIES]) if item
            ]
        else:
            t_step = time.time()
            selected_comorbidities = module_2_2_sample_comorbidities(
                comorbidities_text,
                patient_age=patient_age,
                patient_gender=patient_gender,
            )
            if selected_comorbidities and complication_phenotypes_text and \
                    complication_phenotypes_text.strip() not in ('', '[]'):
                for comorbidity_name in selected_comorbidities:
                    (c_signs, c_labs, c_imaging, c_functional,
                     c_absent_signs, c_absent_labs,
                    c_absent_imaging, c_absent_functional) = _sample_complication_phenotypes(
                        comorbidity_name, complication_phenotypes_text,
                        patient_gender=patient_gender,
                    )
                    selected_signs = list(selected_signs) + c_signs
                    selected_lab_tests = list(selected_lab_tests) + c_labs
                    selected_imaging = list(selected_imaging) + c_imaging
                    selected_functional = list(selected_functional) + c_functional
                    absent_signs = list(absent_signs) + c_absent_signs
                    absent_lab_tests = list(absent_lab_tests) + c_absent_labs
                    absent_imaging = list(absent_imaging) + c_absent_imaging
                    absent_functional = list(absent_functional) + c_absent_functional

            row[COL_COMORBIDITIES] = str(selected_comorbidities)
            row[COL_SYMPTOMS] = str(selected_symptoms)
            row[COL_SIGNS] = str(selected_signs)
            row[COL_LAB_TESTS] = str(selected_lab_tests)
            row[COL_IMAGING] = str(selected_imaging)
            row[COL_FUNCTIONAL_TESTS] = str(selected_functional)
            row[COL_ABSENT_SYMPTOMS] = str(absent_symptoms)
            row[COL_ABSENT_SIGNS] = str(absent_signs)
            row[COL_ABSENT_LAB_TESTS] = str(absent_lab_tests)
            row[COL_ABSENT_IMAGING] = str(absent_imaging)
            row[COL_ABSENT_FUNCTIONAL] = str(absent_functional)
            print(f"    [{tag_prefix}-2.2] 合并症及其表型已在相关性修正前注入，"
                  f"用时 {round(time.time()-t_step,3)}s")
            _push_update()

        current_state_hash = _phenotype_state_hash_from_lists(
            [selected_symptoms, selected_signs, selected_lab_tests,
             selected_imaging, selected_functional],
            [absent_symptoms, absent_signs, absent_lab_tests,
             absent_imaging, absent_functional],
        )
        diagnosis_for_exercise = (
            row.get(COL_DIAGNOSIS, '')
            or str(seed_text or '').split('#', 1)[0]
        ).strip()
        kussmaul_contradiction = _has_kussmaul_without_acidosis(
            selected_signs, selected_lab_tests, absent_lab_tests
        )
        has_correlation = (
            not kussmaul_contradiction
            and _correlation_audit_is_converged(
                row.get(COL_M24_OUTPUT, ''), expected_hash=current_state_hash
            )
        )
        if not has_correlation:
            has_timeline = has_prior_history = has_quantified = has_specific = has_chief = False
            for stale_col in (
                COL_M25_INPUT, COL_M25_OUTPUT, COL_TIME_ORDER, COL_DURATION_TOTAL,
                COL_M26_INPUT, COL_M26_OUTPUT, COL_PRIOR_VISIT_HISTORY,
                COL_M27_INPUT, COL_M27_OUTPUT, COL_QUANTIFIED,
                COL_SPECIFIC, COL_CHIEF_COMPLAINT,
            ):
                row[stale_col] = ''

        # ---- 模块2.3（既往就诊状态 roll）----
        if has_prior_status:
            print(f"    [{tag_prefix}-2.3] ⏭️  已有就诊状态数据，跳过")
            prior_visited = (row.get(COL_PRIOR_VISITED, '') or '未就诊').strip() or '未就诊'
            try:
                prior_visit_count = int(str(row.get(COL_PRIOR_VISIT_COUNT, '') or '0').strip() or '0')
            except (ValueError, TypeError):
                prior_visit_count = 0
        else:
            t_step = time.time()
            prompt_23, response_23, info_23 = module_2_3_prior_visit_roll(
                acuity_val, tag=f"{tag_prefix}-2.3"
            )
            prior_visited = info_23.get('visited', '未就诊')
            prior_visit_count = int(info_23.get('count', 0) or 0)
            row[COL_PRIOR_VISITED] = prior_visited
            row[COL_PRIOR_VISIT_COUNT] = str(prior_visit_count)
            row[COL_M23_INPUT] = prompt_23 or ''
            row[COL_M23_OUTPUT] = response_23 or ''
            print(f"    [{tag_prefix}-2.3] visited={prior_visited}, count={prior_visit_count}，"
                  f"用时 {round(time.time()-t_step,3)}s")
            _push_update()

        # ---- 模块2.4：五类阳性/阴性共同进入多轮相关性修正 ----
        if has_correlation:
            print(f"    [{tag_prefix}-2.4] ⏭️  已有已收敛的全模态校正数据，跳过")
        else:
            t_step = time.time()
            correlation_result = _retry_module_call(
                module_2_4_correlation_correction,
                args=(
                    selected_symptoms, selected_signs, selected_lab_tests,
                    selected_imaging, selected_functional,
                    symptoms_text, signs_text, lab_tests_text,
                    imaging_text, functional_tests_text,
                    seed_text, patient_stage_val,
                ),
                kwargs={
                    'absent_symptoms': absent_symptoms,
                    'absent_signs': absent_signs,
                    'absent_lab_tests': absent_lab_tests,
                    'absent_imaging': absent_imaging,
                    'absent_functional': absent_functional,
                    'patient_level_index': patient_level_index,
                    'patient_gender': patient_gender,
                    'acuity': acuity_val,
                    'diagnosis': diagnosis_for_exercise,
                    'force_exercise_contraindication': False,
                    'tag': f"{tag_prefix}-2.4",
                },
                max_retries=M2_CORRELATION_MAX_RETRIES,
                module_name=f"{tag_prefix}-2.4",
            )
            (prompt_24, response_24,
             selected_symptoms, selected_signs,
             selected_lab_tests, selected_imaging, selected_functional,
             absent_symptoms, absent_signs, absent_lab_tests,
             absent_imaging, absent_functional) = correlation_result
            row[COL_M24_INPUT] = prompt_24 or ''
            row[COL_M24_OUTPUT] = response_24 or ''
            row[COL_SYMPTOMS] = str(selected_symptoms)
            row[COL_SIGNS] = str(selected_signs)
            row[COL_LAB_TESTS] = str(selected_lab_tests)
            row[COL_IMAGING] = str(selected_imaging)
            row[COL_FUNCTIONAL_TESTS] = str(selected_functional)
            row[COL_ABSENT_SYMPTOMS] = str(absent_symptoms)
            row[COL_ABSENT_SIGNS] = str(absent_signs)
            row[COL_ABSENT_LAB_TESTS] = str(absent_lab_tests)
            row[COL_ABSENT_IMAGING] = str(absent_imaging)
            row[COL_ABSENT_FUNCTIONAL] = str(absent_functional)
            print(f"    [{tag_prefix}-2.4] 全模态校正已收敛，用时 {round(time.time()-t_step,1)}s")
            _push_update()

        # ---- 模块2.5（时间排序，仅对症状；按 acuity / visited 切换格式）----
        existing_time_model = _validated_existing_time_model(row, selected_symptoms)
        has_timeline = existing_time_model is not None
        if has_timeline:
            print(f"    [{tag_prefix}-2.5] ⏭️  已有排序数据，跳过")
            time_model = existing_time_model
            time_ordered_items = time_model['symptom_timeline']
            duration_total_str = time_model['current_episode_duration']
            prior_history_duration_str = _time_model_prior_history_duration(time_model)
        else:
            has_prior_history = has_quantified = has_specific = has_chief = False
            for stale_col in (
                COL_M26_INPUT, COL_M26_OUTPUT, COL_PRIOR_VISIT_HISTORY,
                COL_M27_INPUT, COL_M27_OUTPUT, COL_QUANTIFIED,
                COL_SPECIFIC, COL_CHIEF_COMPLAINT,
            ):
                row[stale_col] = ''
            t_step = time.time()
            ret_25 = _retry_module_call(
                module_2_5_build_timeline,
                args=(seed_text, patient_stage_val,
                      selected_symptoms, selected_signs,
                      selected_lab_tests, selected_imaging, selected_functional),
                kwargs={'tag': f"{tag_prefix}-2.5",
                        'acuity': acuity_val,
                        'prior_visited': prior_visited,
                        'prior_visit_count': prior_visit_count},
                max_retries=M2_TIMELINE_MAX_RETRIES,
                module_name=f"{tag_prefix}-2.5"
            )
            prompt_25, response_25, time_model = ret_25
            time_ordered_items = time_model['symptom_timeline']
            duration_total_str = time_model['current_episode_duration']
            prior_history_duration_str = _time_model_prior_history_duration(time_model)

            row[COL_M25_INPUT] = prompt_25 or ''
            row[COL_M25_OUTPUT] = response_25 or ''
            row[COL_TIME_ORDER] = str(time_ordered_items)
            row[COL_DURATION_TOTAL] = duration_total_str
            print(f"    [{tag_prefix}-2.5] 步骤用时 {round(time.time()-t_step,1)}s（共 {len(time_ordered_items)} 条，"
                  f"current_episode_duration={duration_total_str or '(未提取)'}）")
            _push_update()

        selected_symptoms = _realize_symptom_durations(
            selected_symptoms, time_ordered_items
        )
        row[COL_SYMPTOMS] = repr(selected_symptoms)
        _push_update()

        # ---- 模块2.6：总病程已由2.5确定后再生成既往就诊史 ----
        suggested_chief_complaint = ''
        if prior_visited == '就诊过' and prior_visit_count >= 1:
            existing_history_info = _validate_prior_visit_history(
                row.get(COL_M26_OUTPUT, '') or '',
                prior_visit_count,
                prior_history_duration_str,
                selected_symptoms,
                symptom_timeline=time_ordered_items,
            )
            has_prior_history = existing_history_info is not None
            if has_prior_history:
                print(f"    [{tag_prefix}-2.6] ⏭️  已有既往就诊经历数据，跳过")
                row[COL_PRIOR_VISIT_HISTORY] = existing_history_info['history_text']
                suggested_chief_complaint = existing_history_info['suggested_chief_complaint']
            else:
                row[COL_M26_INPUT] = ''
                row[COL_M26_OUTPUT] = ''
                row[COL_PRIOR_VISIT_HISTORY] = ''
                t_step = time.time()
                diagnosis_for_26 = (row.get(COL_DIAGNOSIS, '') or '').strip()
                if not diagnosis_for_26:
                    _, _, _, diagnosis_for_26, _ = _parse_seed(seed_text)
                prompt_26, response_26, info_26 = _retry_module_call(
                    module_2_6_prior_visit_history,
                    args=(diagnosis_for_26, acuity_val, prior_visit_count,
                          prior_history_duration_str, selected_symptoms,
                          selected_signs, selected_lab_tests),
                    kwargs={
                        'symptom_timeline': time_ordered_items,
                        'tag': f"{tag_prefix}-2.6",
                    },
                    max_retries=M2_PRIOR_HISTORY_MAX_RETRIES,
                    module_name=f"{tag_prefix}-2.6",
                )
                row[COL_M26_INPUT] = prompt_26 or ''
                row[COL_M26_OUTPUT] = response_26 or ''
                row[COL_PRIOR_VISIT_HISTORY] = info_26.get('history_text', '') or ''
                suggested_chief_complaint = info_26.get('suggested_chief_complaint', '') or ''
                print(f"    [{tag_prefix}-2.6] 使用基础病程 {prior_history_duration_str} 生成既往就诊史，"
                      f"用时 {round(time.time()-t_step,1)}s")
                _push_update()
        else:
            has_prior_history = False
            row[COL_M26_INPUT] = ''
            row[COL_M26_OUTPUT] = ''
            row[COL_PRIOR_VISIT_HISTORY] = ''

        # 相关性收敛后的五类阳性底表只在这里投影为最终时间视图；不再二次回填旧状态。
        final_time_order = _build_final_time_order(
            time_ordered_items, selected_signs, selected_lab_tests,
            selected_imaging, selected_functional,
        )
        row[COL_TIME_ORDER] = str(final_time_order)
        corrected = row[COL_TIME_ORDER]
        _push_update()

        absent_objective_by_category = {
            '体征': absent_signs,
            '实验室检查': absent_lab_tests,
            '影像检查': absent_imaging,
            '功能检查': absent_functional,
        }

        # ---- 模块2.7 ----
        try:
            existing_quantified = _validate_quantified_timeline(
                final_time_order,
                parse_list_from_response(row.get(COL_QUANTIFIED, '') or ''),
            )
            existing_quantified = _harmonize_objective_metric_ranges(
                existing_quantified, absent_objective_by_category
            )
        except ValueError:
            existing_quantified = None
        has_quantified = existing_quantified is not None
        if not has_quantified:
            has_specific = False
            for stale_col in (COL_M27_INPUT, COL_M27_OUTPUT, COL_QUANTIFIED, COL_SPECIFIC):
                row[stale_col] = ''
        if has_quantified:
            print(f"    [{tag_prefix}-2.7] ⏭️  已有量化数据，跳过")
            quantified = repr(existing_quantified)
            row[COL_QUANTIFIED] = quantified
        else:
            t_step = time.time()
            prompt_27, response_27, quantified_items = _retry_module_call(
                module_2_7_quantify_values,
                args=(seed_text, patient_stage_val, corrected),
                kwargs={
                    'tag': f"{tag_prefix}-2.7",
                    'absent_by_category': absent_objective_by_category,
                },
                max_retries=M2_QUANTIFICATION_MAX_RETRIES,
                module_name=f"{tag_prefix}-2.7"
            )
            row[COL_M27_INPUT] = prompt_27 or ''
            row[COL_M27_OUTPUT] = response_27 or ''
            quantified_items = _validate_quantified_timeline(
                final_time_order, quantified_items
            )
            quantified = repr(quantified_items)
            row[COL_QUANTIFIED] = quantified
            print(f"    [{tag_prefix}-2.7] 步骤用时 {round(time.time()-t_step,1)}s")
            _push_update()

        quantified_items_for_28 = _harmonize_objective_metric_ranges(
            parse_list_from_response(quantified),
            absent_objective_by_category,
        )
        quantified = repr(quantified_items_for_28)
        row[COL_QUANTIFIED] = quantified

        # ---- 模块2.8 ----
        existing_specific = _validated_existing_specific_timeline(
            quantified_items_for_28, row.get(COL_SPECIFIC, '') or '',
            absent_objective_by_category,
        )
        has_specific = existing_specific is not None
        if has_specific:
            row[COL_SPECIFIC] = repr(existing_specific)
            print(f"    [{tag_prefix}-2.8] ⏭️  已有具体数值数据，跳过")
        else:
            t_step = time.time()
            age, gender, diagnosis, specific_text = module_2_8_specific_values(
                seed_text, quantified, tag=f"{tag_prefix}-2.8",
                preset_age=patient_age, preset_gender=patient_gender,
                absent_by_category=absent_objective_by_category,
            )
            _validate_objective_value_consistency(
                quantified_items_for_28,
                parse_list_from_response(specific_text),
                absent_objective_by_category,
            )
            row[COL_AGE] = str(age)
            row[COL_GENDER] = gender
            row[COL_DIAGNOSIS] = diagnosis
            row[COL_SPECIFIC] = specific_text
            _push_update()


        # ---- 模块2.81：终态固定点校验并冻结事实账本 ----
        row, fact_ledger = module_2_81_finalize_fact_ledger(
            row, staging_system, max_rounds=5
        )
        _push_update()
        if fact_ledger.get('status') != 'converged':
            blockers = fact_ledger.get('audit', {}).get('blockers') or []
            raise ValueError(f'M2.81 fact ledger did not converge: {blockers}')

        # ---- 模块2.9：主诉提取 ----
        if has_chief:
            print(f"    [{tag_prefix}-2.9] ⏭️  已有主诉数据，跳过")
        else:
            chief_complaint_val = module_2_9_derive_chief_complaint(
                row.get(COL_TIME_ORDER, '') or '',
                seed_text=seed_text,
                tag=f"{tag_prefix}-2.9",
                symptoms_list=selected_symptoms,
                diagnosis=row.get(COL_DIAGNOSIS, '') or '',
                patient_stage=patient_stage_val,
                suggested=suggested_chief_complaint,
                time_ordered_items=parse_list_from_response(row.get(COL_TIME_ORDER, '') or ''),
            )
            row[COL_CHIEF_COMPLAINT] = chief_complaint_val
            if chief_complaint_val:
                print(f"    [{tag_prefix}-2.9] 主诉提取: {chief_complaint_val}")
            _push_update()

        worker_elapsed = round(time.time() - worker_start, 1)
        print(f"    [{tag_prefix}] ✅ 模块2完成，总用时 {worker_elapsed}s")
        _push_update()

    except Exception as e:
        worker_elapsed = round(time.time() - worker_start, 1)
        print(f"    [{tag_prefix}] ❌ 模块2失败（{worker_elapsed}s）: {e}")
        traceback.print_exc()
        if not row.get(COL_SYMPTOMS, ''):
            row[COL_SYMPTOMS] = f'ERROR: {e}'
        _push_update()
        raise RuntimeError(f'{tag_prefix}模块2生成失败') from e

    return row


# ============================================================
# 单CSV文件处理函数
# ============================================================

def process_csv_module2(csv_path, num_patients=5):
    """
    对单个 CSV 文件运行模块2：
    - 读取 row_1（概率库）
    - 为每个患者（row_2..N+1）运行模块2（如已有数据则跳过）
    - 结果写回同一 CSV

    Returns:
        dict: {'status': 'success'/'error/skip', 'file': ...}
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*60}")
    print(f"[模块2] 处理: {csv_path}")
    print(f"{'='*60}")

    try:
        row_1, existing_patients = _load_existing_csv(csv_path)

        if row_1 is None:
            print(f"  ⚠️ CSV不存在或为空，跳过: {csv_path}")
            return {'status': 'skip', 'file': csv_path}

        seed_text = row_1.get(COL_SEED, '')
        staging_system = _validate_m1_row_for_m2(row_1)

        symptoms_text = row_1[COL_SYMPTOMS]
        signs_text = row_1[COL_SIGNS]
        lab_tests_text = row_1[COL_LAB_TESTS]
        imaging_text = row_1[COL_IMAGING]
        functional_tests_text = row_1[COL_FUNCTIONAL_TESTS]
        comorbidities_text = row_1.get(COL_COMORBIDITIES, '[]')
        laterality_type = row_1.get(COL_LATERALITY_TYPE, 'either') or 'either'
        complication_phenotypes_text = row_1.get(COL_COMPLICATION_PHENOTYPES, '[]') or '[]'

        level_names = _get_level_names(staging_system)

        # 分配时期
        existing_stages = {}
        new_stage_needed = []
        for i in range(1, num_patients + 1):
            existing_row = existing_patients.get(i, None)
            if existing_row and existing_row.get(COL_STAGE, '').strip():
                existing_stages[i] = _validate_patient_stage_name(existing_row[COL_STAGE], level_names)
            else:
                new_stage_needed.append(i)

        if new_stage_needed:
            new_stages = _module_2_0_assign_stages_local(len(new_stage_needed), staging_system)
            for idx_pos, patient_i in enumerate(new_stage_needed):
                existing_stages[patient_i] = new_stages[idx_pos]

        # 判断哪些患者需要运行模块2
        patient_latest = {}
        tasks = []
        skip_count = 0
        update_queue = _queue_module.Queue()
        reserved_case_ids = _existing_case_ids(existing_patients)

        for i in range(1, num_patients + 1):
            patient_stage_i = existing_stages.get(i, level_names[0] if level_names else '轻度')
            existing_row = existing_patients.get(i, None)
            prepared_row = _initialize_patient_row_identity(i, existing_row, reserved_case_ids)
            reserved_case_ids.add(prepared_row[COL_CASE_ID])
            patient_latest[i] = prepared_row

            if existing_row and _row_is_m2_complete(prepared_row, staging_system, csv_path, i):
                skip_count += 1
                continue

            tasks.append(
                (i, seed_text, symptoms_text, signs_text,
                 lab_tests_text, imaging_text, functional_tests_text,
                 comorbidities_text,
                 staging_system, patient_stage_i,
                 update_queue, prepared_row,
                 laterality_type, complication_phenotypes_text)
            )

        if skip_count > 0:
            print(f"  ⏭️  已跳过 {skip_count} 个已完成模块2的患者")

        if not tasks:
            print(f"  ⏭️  所有 {num_patients} 个患者的模块2已完成")
        else:
            print(f"  🔄 需要运行模块2 的患者: {len(tasks)} 个")

            worker_errors = []
            with ThreadPoolExecutor(max_workers=min(len(tasks), num_patients)) as executor:
                futures = {
                    executor.submit(_generate_single_patient_m2, task): task[0]
                    for task in tasks
                }
                all_done = False
                while not all_done:
                    all_done = all(f.done() for f in futures)
                    drained = False
                    while not drained:
                        try:
                            patient_idx, row_data = update_queue.get(
                                timeout=1 if not all_done else 0.1
                            )
                            patient_latest[patient_idx] = row_data
                            patient_rows_sorted = [
                                patient_latest[k] for k in sorted(patient_latest.keys())
                            ]
                            _save_csv_with_fact_ledgers(csv_path, [row_1] + patient_rows_sorted)
                            print(f"  📝 实时保存: 患者{patient_idx}有更新 "
                                  f"(进行中{len(patient_latest)}个)")
                        except _queue_module.Empty:
                            drained = True

            for future, patient_idx in futures.items():
                try:
                    completed_row = future.result()
                    patient_latest[patient_idx] = completed_row
                    patient_rows_sorted = [
                        patient_latest[k] for k in sorted(patient_latest.keys())
                    ]
                    _save_csv_with_fact_ledgers(csv_path, [row_1] + patient_rows_sorted)
                    if not _row_is_m2_complete(completed_row, staging_system, csv_path, patient_idx):
                        worker_errors.append(f'患者{patient_idx}: 模块2输出不完整或校验失败')
                except Exception as exc:
                    worker_errors.append(f'患者{patient_idx}: {exc}')

            while True:
                try:
                    patient_idx, row_data = update_queue.get_nowait()
                    patient_latest[patient_idx] = row_data
                except _queue_module.Empty:
                    break

        # 最终保存
        patient_rows_sorted = [patient_latest[k] for k in sorted(patient_latest.keys())]
        _save_csv_with_fact_ledgers(csv_path, [row_1] + patient_rows_sorted)
        if tasks and worker_errors:
            error_text = '; '.join(worker_errors)
            print(f"  ❌ 模块2存在失败患者，已保存当前进度: {error_text}")
            return {'status': 'error', 'file': csv_path, 'error': error_text}
        print(f"  ✅ 模块2完成，保存至: {csv_path}")
        print(f"{'='*60}")
        print(f"\n下一步：运行 virtual_clinical_interaction.py --csv_file {csv_path}")

        return {'status': 'success', 'file': csv_path}

    except Exception as e:
        print(f"  ❌ 处理CSV [{csv_path}] 时出错: {e}")
        traceback.print_exc()
        return {'status': 'error', 'file': csv_path, 'error': str(e)}


def _module_2_0_assign_stages_local(num_patients, staging_system):
    """内部：同 module_2_0_assign_stages"""
    return module_2_0_assign_stages(num_patients, staging_system)


# ============================================================
# 批量处理多个 CSV
# ============================================================

class _CsvScopedQueue:
    """
    将 worker 往 update_queue 投递的 (patient_idx, row_data) 转发到
    全局队列，并附加所属 csv_path，实现跨 CSV 的统一 drain。
    """
    def __init__(self, csv_path, global_queue):
        self.csv_path = csv_path
        self.global_queue = global_queue

    def put(self, item):
        try:
            patient_idx, row_data = item
            self.global_queue.put((self.csv_path, patient_idx, row_data))
        except Exception:
            pass


def _prepare_csv_m2(csv_path, num_patients, global_queue):
    """
    CSV 级预处理：读取 CSV → 校验 → stage 分配 → 构造 tasks 列表。
    返回 (state, tasks)。
        - state: {'csv_path','row_1','patient_latest','lock'}，用于后续写盘
        - tasks: 送入全局线程池的 task 元组列表（update_queue 已绑定为 scoped）
    若 CSV 不可处理（row_1 缺失/缺分级系统），返回 (None, [])。
    """
    print(f"\n[模块2 预处理] {csv_path}")
    try:
        row_1, existing_patients = _load_existing_csv(csv_path)
        if row_1 is None:
            print(f"  ⚠️ CSV不存在或为空，跳过: {csv_path}")
            return None, []
        seed_text = row_1.get(COL_SEED, '')
        staging_system = _validate_m1_row_for_m2(row_1)

        symptoms_text = row_1[COL_SYMPTOMS]
        signs_text = row_1[COL_SIGNS]
        lab_tests_text = row_1[COL_LAB_TESTS]
        imaging_text = row_1[COL_IMAGING]
        functional_tests_text = row_1[COL_FUNCTIONAL_TESTS]
        comorbidities_text = row_1.get(COL_COMORBIDITIES, '[]')
        laterality_type = row_1.get(COL_LATERALITY_TYPE, 'either') or 'either'
        complication_phenotypes_text = row_1.get(COL_COMPLICATION_PHENOTYPES, '[]') or '[]'

        level_names = _get_level_names(staging_system)

        # 分配时期
        existing_stages = {}
        new_stage_needed = []
        for i in range(1, num_patients + 1):
            existing_row = existing_patients.get(i, None)
            if existing_row and existing_row.get(COL_STAGE, '').strip():
                existing_stages[i] = _validate_patient_stage_name(existing_row[COL_STAGE], level_names)
            else:
                new_stage_needed.append(i)
        if new_stage_needed:
            new_stages = _module_2_0_assign_stages_local(len(new_stage_needed), staging_system)
            for idx_pos, patient_i in enumerate(new_stage_needed):
                existing_stages[patient_i] = new_stages[idx_pos]

        scoped_queue = _CsvScopedQueue(csv_path, global_queue)

        tasks = []
        skip_count = 0
        patient_latest = {}
        reserved_case_ids = _existing_case_ids(existing_patients)
        for i in range(1, num_patients + 1):
            patient_stage_i = existing_stages.get(i, level_names[0] if level_names else '轻度')
            existing_row = existing_patients.get(i, None)
            prepared_row = _initialize_patient_row_identity(i, existing_row, reserved_case_ids)
            reserved_case_ids.add(prepared_row[COL_CASE_ID])
            patient_latest[i] = prepared_row

            if existing_row and _row_is_m2_complete(prepared_row, staging_system, csv_path, i):
                skip_count += 1
                continue

            tasks.append(
                (i, seed_text, symptoms_text, signs_text,
                 lab_tests_text, imaging_text, functional_tests_text,
                 comorbidities_text,
                 staging_system, patient_stage_i,
                 scoped_queue, prepared_row,
                 laterality_type, complication_phenotypes_text)
            )

        if skip_count > 0:
            print(f"  ⏭️  [{os.path.basename(csv_path)}] 已跳过 {skip_count} 个已完成患者")
        if tasks:
            print(f"  🔄 [{os.path.basename(csv_path)}] 待运行模块2 患者: {len(tasks)} 个")
        else:
            print(f"  ✅ [{os.path.basename(csv_path)}] 所有患者的模块2已完成")

        state = {
            'csv_path': csv_path,
            'row_1': row_1,
            'patient_latest': patient_latest,
            'staging_system': staging_system,
            'lock': threading.Lock(),
        }
        return state, tasks
    except Exception as e:
        print(f"  ❌ 预处理出错 [{csv_path}]: {e}")
        traceback.print_exc()
        return None, []


def _apply_update_m2(state, patient_idx, row_data):
    """线程安全：收到 worker 的实时更新，落盘。"""
    with state['lock']:
        state['patient_latest'][patient_idx] = row_data
        rows = [state['row_1']] + [
            state['patient_latest'][k] for k in sorted(state['patient_latest'].keys())
        ]
        _save_csv_with_fact_ledgers(state['csv_path'], rows)
    print(f"  📝 [{os.path.basename(state['csv_path'])}] 实时保存: 患者{patient_idx}有更新")


def _finalize_csv_m2(state):
    """最终确保 CSV 包含全部患者数据。"""
    with state['lock']:
        rows = [state['row_1']] + [
            state['patient_latest'][k] for k in sorted(state['patient_latest'].keys())
        ]
        _save_csv_with_fact_ledgers(state['csv_path'], rows)
    print(f"  ✅ [模块2] 最终保存: {state['csv_path']}")


def run_module2(csv_dir=None, csv_file=None, num_patients=5, num_workers=50):
    """
    主入口：扫描目录中的所有 CSV 并运行模块2，或处理单个 CSV。
    使用**跨 CSV 全局线程池**：num_workers 个线程同时服务来自不同 CSV 的患者任务。
    """
    if csv_file:
        csv_files = [csv_file]
    elif csv_dir:
        csv_files = sorted(glob(os.path.join(csv_dir, '**', '*.csv'), recursive=True))
        # 排除汇总文件
        csv_files = [f for f in csv_files if '处理汇总' not in os.path.basename(f)
                     and '循证调用记录' not in f]
    else:
        print("错误：请指定 --csv_dir 或 --csv_file")
        return {'status': 'error', 'error': '未指定CSV输入'}

    if not csv_files:
        print(f"未找到CSV文件: {csv_dir or csv_file}")
        return {'status': 'error', 'error': '未找到CSV文件'}

    print(f"找到 {len(csv_files)} 个CSV文件，开始模块2处理（全局并发 {num_workers}，每 CSV 患者数 {num_patients}）...")
    start_time = time.time()

    global_queue = _queue_module.Queue()
    csv_states = {}     # csv_path -> state
    all_tasks = []      # [(csv_path, task_tuple), ...]
    prep_skipped = 0
    prep_errors = []

    for csv_path in csv_files:
        state, tasks = _prepare_csv_m2(csv_path, num_patients, global_queue)
        if state is None:
            prep_skipped += 1
            prep_errors.append(csv_path)
            continue
        csv_states[csv_path] = state
        for t in tasks:
            all_tasks.append((csv_path, t))

    worker_errors = []
    if not all_tasks:
        if prep_errors:
            print("\n没有可执行的模块2任务，且存在 CSV 预处理失败。")
        else:
            print("\n所有 CSV 的模块2均已完成，无需运行。")
    else:
        total = len(all_tasks)
        print(f"\n{'='*60}")
        print(f"共 {total} 个患者任务将通过全局并发池（workers={num_workers}）处理")
        print(f"{'='*60}")

        effective_workers = min(total, num_workers)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_generate_single_patient_m2, task): (csv_path, task[0])
                for (csv_path, task) in all_tasks
            }
            all_done = False
            while not all_done:
                all_done = all(f.done() for f in futures)
                drained = False
                while not drained:
                    try:
                        csv_path, patient_idx, row_data = global_queue.get(
                            timeout=1 if not all_done else 0.1
                        )
                        state = csv_states.get(csv_path)
                        if state is not None:
                            _apply_update_m2(state, patient_idx, row_data)
                    except _queue_module.Empty:
                        drained = True

        for future, (csv_path, patient_idx) in futures.items():
            try:
                row_data = future.result()
                state = csv_states.get(csv_path)
                if state is not None:
                    _apply_update_m2(state, patient_idx, row_data)
                if not _row_is_m2_complete(row_data, state.get('staging_system') if state else None, csv_path, patient_idx):
                    worker_errors.append({
                        'file': csv_path,
                        'patient_idx': patient_idx,
                        'error': '模块2输出不完整或校验失败',
                    })
            except Exception as exc:
                worker_errors.append({
                    'file': csv_path,
                    'patient_idx': patient_idx,
                    'error': str(exc),
                })

        # drain 残留
        while True:
            try:
                csv_path, patient_idx, row_data = global_queue.get_nowait()
                state = csv_states.get(csv_path)
                if state is not None:
                    _apply_update_m2(state, patient_idx, row_data)
            except _queue_module.Empty:
                break

    # 每个 CSV 最终保存
    for state in csv_states.values():
        _finalize_csv_m2(state)

    total_time = round(time.time() - start_time, 2)
    if prep_errors or worker_errors:
        print(f"\n❌ 模块2未全部完成：预处理失败 {len(prep_errors)} 个CSV，"
              f"患者失败 {len(worker_errors)} 个")
        return {
            'status': 'error',
            'files': len(csv_states),
            'prep_errors': prep_errors,
            'patient_errors': worker_errors,
            'elapsed_seconds': total_time,
        }
    print(f"\n{'='*60}")
    print(f"模块2全部处理完成")
    print(f"处理 CSV: {len(csv_states)}  预处理跳过: {prep_skipped}")
    print(f"总耗时: {total_time} 秒")
    print(f"{'='*60}")
    return {
        'status': 'success',
        'files': len(csv_states),
        'prep_errors': [],
        'patient_errors': [],
        'elapsed_seconds': total_time,
    }


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='atlas_based_patient_generation.py：模块2 - 基于概率库生成患者表型'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv_dir', type=str,
                       help='扫描目录中所有由 phenotypic_atlas.py 生成的 CSV')
    group.add_argument('--csv_file', type=str,
                       help='处理单个 CSV 文件')
    parser.add_argument('--num_patients', type=int, default=5,
                        help='每个种子生成的模拟患者数（默认: 5）')
    parser.add_argument('--num_workers', type=int, default=50,
                        help='跨 CSV 全局并发线程数，同时服务不同 CSV 的不同患者（默认: 50）')
    args = parser.parse_args()

    result = run_module2(
        csv_dir=args.csv_dir,
        csv_file=args.csv_file,
        num_patients=args.num_patients,
        num_workers=args.num_workers,
    )
    if not result or result.get('status') != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    main()
