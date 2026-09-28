# -*- coding: utf-8 -*-
"""
virtual_clinical_interaction.py
模块3：虚拟临床交互

功能：读取 atlas_based_patient_generation.py 输出的 CSV（患者行已含具体表型），
为每个患者运行医生-患者-上帝三方交互问诊。
治疗方案、ADR 采样和治疗效果评估不进入当前主流程。

输出：更新同一 {csv_dir}/{seed_name}.csv（在现有患者行基础上追加后续列）

上游：atlas_based_patient_generation.py（须先运行，填充 COL_SPECIFIC）
"""

import sys
import os
import re
import json
import time
import random
import argparse
import traceback
import queue as _queue_module
import threading
from concurrent.futures import ThreadPoolExecutor
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DEFAULT_MAX_INTERACTION_TURNS, DEFAULT_MAX_FOLLOWUP_TURNS,
    call_gpt5, _retry_module_call,
    parse_list_from_response, parse_staging_system, _get_level_names,
    COL_SEED, COL_STAGING_SYSTEM, COL_STAGE,
    COL_SYMPTOMS, COL_SIGNS, COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS,
    COL_STANDARD_TREATMENT, COL_ADR_LIBRARY,
    COL_TIME_ORDER, COL_COMORBIDITIES,
    COL_SPECIFIC, COL_CHIEF_COMPLAINT,
    COL_AGE, COL_GENDER, COL_DIAGNOSIS,
    COL_PATIENT_LATERALITY,
    COL_INTERACTION, COL_TREATMENT, COL_PATIENT_ADR, COL_TREATMENT_OUTCOME,
    COL_M41_INPUT, COL_M41_OUTPUT, COL_M42_INPUT, COL_M42_OUTPUT,
    COL_ACUITY, COL_PRIOR_VISITED, COL_PRIOR_VISIT_COUNT,
    COL_DURATION_TOTAL, COL_PRIOR_VISIT_HISTORY,
    COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS, COL_ABSENT_LAB_TESTS,
    COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL,
    _empty_row, _load_existing_csv, _save_csv,
)
from phenotypic_atlas import _parse_staged_treatment


MAX_DIAGNOSTIC_TEST_ITEMS = 8
M3_LEDGER_METADATA_KEY = '模块3_账本元数据'
_M345_ERROR_ROW_KEY = '_m345_error'


def _split_top_level_diagnostic_items(text: str) -> list:
    bracket_pairs = {
        '(': ')', '（': '）', '[': ']', '［': '］', '【': '】',
        '{': '}', '｛': '｝',
    }
    closing_brackets = set(bracket_pairs.values())
    numbered_item = re.compile(
        r'(?:\(\d+\)|（\d+）|\d+\.(?!\d)|\d+\)|\d+、)\s*'
    )
    separators = frozenset('、，,；;\n\r')
    items = []
    current = []
    bracket_stack = []

    def flush():
        item = re.sub(r'^(?:[-*•]\s*)+', '', ''.join(current).strip())
        if item:
            items.append(item)
        current.clear()

    index = 0
    while index < len(text):
        if (
            not bracket_stack
            and (index == 0 or text[index - 1].isspace() or text[index - 1] in separators)
            and (match := numbered_item.match(text, index)) is not None
        ):
            flush()
            index = match.end()
            continue
        character = text[index]
        if character in bracket_pairs:
            bracket_stack.append(bracket_pairs[character])
            current.append(character)
        elif character in closing_brackets:
            if not bracket_stack or character != bracket_stack[-1]:
                return [text.strip()] if text.strip() else []
            bracket_stack.pop()
            current.append(character)
        elif not bracket_stack and character in separators:
            flush()
        else:
            current.append(character)
        index += 1
    if bracket_stack:
        return [text.strip()] if text.strip() else []
    flush()
    return items


def _extract_marked_section(text: str, marker: str) -> str:
    pattern = re.compile(
        rf'(?:\[|【){re.escape(marker)}(?:\]|】)\s*[:：]\s*(.*?)'
        rf'(?=(?:\r?\n)?\s*(?:\[|【)[^\]】\r\n]+(?:\]|】)\s*[:：]|$)',
        flags=re.S,
    )
    match = pattern.search(str(text or ''))
    return match.group(1).strip() if match else ''


def _has_marked_section(text: str, marker: str) -> bool:
    return bool(re.search(rf'(?:\[|【){re.escape(marker)}(?:\]|】)\s*[:：]', str(text or '')))



def _cap_laboratory_request(text: str, max_items: int = MAX_DIAGNOSTIC_TEST_ITEMS) -> str:
    raw_text = str(text or '')
    request_pattern = re.compile(
        r'((?:\[|【)申请化验(?:\]|】)\s*[:：]\s*)(.*?)'
        r'(?=(?:\r?\n)?\s*(?:\[|【)[^\]】\r\n]+(?:\]|】)\s*[:：]|$)',
        flags=re.S,
    )

    def cap_match(match):
        body = match.group(2).strip()
        instruction_match = re.match(r'((?:请)?(?:检测|检查)\s*[:：]\s*)', body)
        instruction = instruction_match.group(1) if instruction_match else ''
        item_text = body[instruction_match.end():] if instruction_match else body
        items = _split_top_level_diagnostic_items(item_text)
        if len(items) <= max_items:
            return match.group(0)
        return match.group(1) + instruction + '、'.join(items[:max_items])

    return request_pattern.sub(cap_match, raw_text)


def _ledger_module():
    import patient_fact_ledger as pfl
    return pfl


def route_clinical_requests(text: str) -> dict[str, list[str]]:
    return _ledger_module().route_clinical_requests(text)


def _fact_name(fact: dict) -> str:
    concept = fact.get('concept') if isinstance(fact.get('concept'), dict) else {}
    return str(
        concept.get('raw') or fact.get('raw')
        or concept.get('canonical') or fact.get('canonical') or ''
    ).strip()


def _fact_canonical(fact: dict) -> str:
    concept = fact.get('concept') if isinstance(fact.get('concept'), dict) else {}
    return str(
        concept.get('canonical') or fact.get('canonical')
        or concept.get('raw') or fact.get('raw') or ''
    ).strip()


def _compact_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fact_value_text(fact: dict) -> str:
    value = fact.get('value') if isinstance(fact.get('value'), dict) else None
    if value and value.get('number') is not None:
        unit = value.get('unit') or ''
        return f"{_compact_number(value.get('number'))}{unit}"
    state = fact.get('clinical_state')
    interpretation = fact.get('interpretation')
    if state == 'absent' or interpretation in {'negative', 'normal'}:
        return '正常/阴性'
    if interpretation == 'high':
        return '升高'
    if interpretation == 'low':
        return '降低'
    if interpretation == 'positive':
        return '阳性'
    if interpretation == 'abnormal':
        return '异常'
    if state == 'present':
        return '阳性/异常'
    return '未见异常'


def _fact_line(fact: dict) -> str:
    line = f"- {_fact_name(fact)}: {_fact_value_text(fact)}"
    if fact.get('feasibility') == 'clinically_questionable':
        line += '（临床可行性需结合场景；按账本原值返回）'
    return line


def _format_fact_list_for_prompt(facts: list, include_values: bool) -> str:
    lines = []
    for fact in facts or []:
        name = _fact_name(fact)
        if not name:
            continue
        if include_values:
            lines.append(_fact_line(fact))
            continue
        state = fact.get('clinical_state') or 'unknown'
        time_payload = fact.get('time') if isinstance(fact.get('time'), dict) else {}
        time_label = time_payload.get('clinical_onset') or time_payload.get('observed_at') or ''
        suffix = f"（{time_label}）" if time_label else ''
        lines.append(f"- {name}: {state}{suffix}")
    return '\n'.join(lines) if lines else '- 无'


def _format_symptom_attributes_for_prompt(attributes) -> str:
    lines = []
    for item in attributes or []:
        if isinstance(item, dict):
            name = item.get('name')
            duration = str(item.get('duration') or '').strip()
            trigger = str(item.get('trigger') or '').strip()
            nature = str(item.get('nature') or '').strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            name = item[0]
            duration = str(item[1] or '').strip()
            trigger = str(item[2] or '').strip()
            nature = str(item[3] or '').strip()
        else:
            continue
        bits = []
        if duration:
            bits.append(f'持续时间 {duration}')
        if trigger:
            bits.append(f'诱发/加重 {trigger}')
        if nature:
            bits.append(f'性质 {nature}')
        if name and bits:
            lines.append(f'- {name}：' + '；'.join(bits))
    return '\n'.join(lines)



def _format_patient_view_for_prompt(patient_view: dict) -> str:
    course = patient_view.get('course') if isinstance(patient_view.get('course'), dict) else {}
    pieces = []
    if course.get('current_episode_duration'):
        pieces.append(f"当前发作时长：{course.get('current_episode_duration')}")
    if course.get('underlying_duration'):
        pieces.append(f"基础病程：{course.get('underlying_duration')}")
    pieces.append('可陈述的症状：')
    pieces.append(_format_fact_list_for_prompt(patient_view.get('symptoms') or [], include_values=False))
    home_measurements = patient_view.get('home_measurements') or []
    if home_measurements:
        pieces.append('患者可自测/可感知的生命体征：')
        pieces.append(_format_fact_list_for_prompt(home_measurements, include_values=True))
    return '\n'.join(pieces)


def _classify_physical_section(name: str) -> str:
    text = str(name or '')
    upper = text.upper()
    if any(token in text for token in ('胸', '肺', '呼吸音', '啰音', '心音', '心脏', '心律')):
        return '胸部查体'
    if any(token in text for token in ('腹', '肝', '脾', '肠鸣音', '压痛', '反跳痛', '胆囊', '墨菲')) or 'MURPHY' in upper:
        return '腹部查体'
    if any(token in text for token in ('神经', '肌力', '病理征', '皮肤', '黏膜', '淋巴', '水肿', '颈静脉')):
        return '其他系统'
    return '一般状态'


_PHYSICAL_EXAM_PANELS = (
    (('肺部听诊', '肺部查体', '呼吸系统', '双肺', '心肺听诊'), ('肺', '呼吸音', '啰音', '哮鸣', '喘鸣', '胸膜摩擦')),
    (('心脏听诊', '心脏查体', '心前区', '心肺听诊'), ('心音', '杂音', '心律', '心界', '心率')),
    (('颈静脉', '颈部血管'), ('颈静脉', 'JVP')),
    (('腹部触诊', '腹部查体', '腹部', '胆囊', 'Murphy', '墨菲'), ('腹', '肝', '脾', '胆囊', '压痛', '反跳痛', 'Murphy', '墨菲')),
    (('水肿', '下肢水肿', '双下肢'), ('水肿', '凹陷性')),
)
_BROAD_PHYSICAL_REQUESTS = ('全身查体', '体格检查', '全面查体')


def _normalize_exam_text(text: str) -> str:
    return re.sub(r'[\s\-_/：:；;，,、（）()\[\]【】]+', '', str(text or '')).upper()


def _physical_exam_request_items(exam_request: str) -> list[str]:
    body = _extract_marked_section(exam_request, '申请查体') or str(exam_request or '')
    body = re.sub(r'^(?:请)?(?:进行)?(?:检查|查体)\s*[:：]?\s*', '', body.strip())
    return _split_top_level_diagnostic_items(body)


def _physical_request_is_broad(items: list[str]) -> bool:
    return any(
        any(_normalize_exam_text(keyword) in _normalize_exam_text(item) for keyword in _BROAD_PHYSICAL_REQUESTS)
        for item in items
    )


def _physical_fact_matches_item(fact: dict, item: str) -> bool:
    item_norm = _normalize_exam_text(item)
    fact_norm = _normalize_exam_text(_fact_name(fact) + _fact_canonical(fact))
    if not item_norm or not fact_norm:
        return False
    if item_norm in fact_norm or fact_norm in item_norm:
        return True
    for request_aliases, fact_aliases in _PHYSICAL_EXAM_PANELS:
        if not any(_normalize_exam_text(alias) in item_norm for alias in request_aliases):
            continue
        if any(_normalize_exam_text(alias) in fact_norm for alias in fact_aliases):
            return True
    return False


def _physical_fact_requested(fact: dict, request_items: list[str], include_all: bool) -> bool:
    if include_all:
        return True
    return any(_physical_fact_matches_item(fact, item) for item in request_items)


def _vital_tokens(fact: dict) -> set[str]:
    tokens = set()
    for text in (_fact_name(fact), _fact_canonical(fact)):
        normalized = str(text or '').upper().strip()
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(token for token in re.split(r'[^A-Z0-9]+', normalized) if token)
    return tokens


def _vital_label_or_none(fact: dict) -> str:
    texts = [_fact_name(fact), _fact_canonical(fact)]
    joined_text = ''.join(texts)
    tokens = _vital_tokens(fact)
    if '体温' in joined_text or '腋温' in joined_text or tokens.intersection({'T', 'TEMP', 'TEMPERATURE'}):
        return 'T'
    if '血压' in joined_text or tokens.intersection({'BP', 'SBP', 'DBP'}):
        return 'BP'
    if '呼吸频率' in joined_text or '呼吸次数' in joined_text or tokens.intersection({'R', 'RR'}):
        return 'R'
    if 'SPO2' in tokens or '血氧' in joined_text or '氧饱和' in joined_text:
        return 'SpO2'
    if '心率' in joined_text or '脉率' in joined_text or '脉搏' in joined_text or tokens.intersection({'P', 'PR', 'HR', 'PULSE'}):
        return 'P'
    return ''


def _is_vital_fact(fact: dict) -> bool:
    return bool(_vital_label_or_none(fact))


def _vital_label(fact: dict) -> str:
    return _vital_label_or_none(fact) or 'P'


def render_physical_exam_result(physical_exam_view: dict, exam_request: str = '') -> str:
    facts = physical_exam_view.get('facts') if isinstance(physical_exam_view, dict) else []
    request_items = _physical_exam_request_items(exam_request)
    include_all = _physical_request_is_broad(request_items)
    vitals = {
        'T': '36.8℃',
        'P': '75次/分',
        'R': '16次/分',
        'BP': '120/80mmHg',
        'SpO2': '98%',
    }
    section_facts = {
        '一般状态': [],
        '胸部查体': [],
        '腹部查体': [],
        '其他系统': [],
    }
    for fact in facts or []:
        if fact.get('clinical_state') == 'absent':
            continue
        name = _fact_name(fact)
        if not name:
            continue
        if _is_vital_fact(fact):
            if _physical_fact_requested(fact, request_items, include_all):
                vitals[_vital_label(fact)] = f"{_fact_value_text(fact)}（{name}）"
            continue
        if not _physical_fact_requested(fact, request_items, include_all):
            continue
        section_facts[_classify_physical_section(name)].append(_fact_line(fact))

    lines = [
        '【生命体征】',
        f"- T: {vitals['T']}",
        f"- P: {vitals['P']}",
        f"- R: {vitals['R']}",
        f"- BP: {vitals['BP']}",
        f"- SpO2: {vitals['SpO2']}",
    ]
    for section in ('一般状态', '胸部查体', '腹部查体', '其他系统'):
        lines.append(f'【{section}】')
        section_lines = section_facts[section]
        lines.extend(section_lines if section_lines else ['- 无异常'])
    return '\n'.join(lines)


def render_diagnostic_view(diagnostic_view: dict) -> str:
    facts_by_domain = diagnostic_view.get('facts') if isinstance(diagnostic_view, dict) else {}
    titles = (
        ('lab', '化验检查'),
        ('imaging', '影像检查'),
        ('functional', '功能学检查'),
    )
    lines = []
    for domain, title in titles:
        lines.append(f'【{title}】')
        facts = facts_by_domain.get(domain) or []
        if not facts:
            lines.append('- 无')
            continue
        for fact in facts:
            lines.append(_fact_line(fact))
    return '\n'.join(lines)


def _load_m3_ledger_context(csv_path, patient_idx, row, staging_system):
    return _ledger_module().load_verified_fact_ledger_for_m3(
        csv_path, patient_idx, row, staging_system
    )


def _m3_ledger_metadata_current(csv_path, patient_idx, expected_metadata):
    return _ledger_module().m3_ledger_metadata_current(
        csv_path, patient_idx, expected_metadata
    )


def _write_m3_ledger_metadata(csv_path, patient_idx, metadata):
    _ledger_module().write_m3_ledger_metadata(csv_path, patient_idx, metadata)


# ============================================================
# 0528：seed → 科室静态映射
# ============================================================

SEED_DEPARTMENT_MAP = {
    # 呼吸科
    '社区获得性肺炎': '呼吸科', '慢性阻塞性肺疾病': '呼吸科', '支气管哮喘': '呼吸科',
    '急性支气管炎': '呼吸科', '急性上呼吸道感染': '呼吸科', '胸腔积液': '呼吸科',
    '肺结核': '呼吸科', '急性肺栓塞': '呼吸科', '特发性肺纤维化': '呼吸科',
    '弥漫性肺泡出血综合征': '呼吸科',
    # 消化科
    '胃食管反流病': '消化科', '消化性溃疡': '消化科', '急性胰腺炎': '消化科',
    '急性胆囊炎': '消化科', '急性胃肠炎': '消化科', '肝硬化': '消化科',
    '急性阑尾炎': '消化科', '克罗恩病': '消化科', '自身免疫性肝炎': '消化科',
    '嗜酸性胃肠炎': '消化科',
    # 心内科
    '原发性高血压': '心内科', '冠状动脉粥样硬化性心脏病': '心内科',
    '慢性心力衰竭': '心内科', '心房颤动': '心内科', '急性心肌梗死': '心内科',
    '急性心包炎': '心内科', '主动脉夹层': '心内科', '肥厚型心肌病': '心内科',
    '心脏淀粉样变性': '心内科', '巨细胞性心肌炎': '心内科',
    # 内分泌科
    '2型糖尿病': '内分泌科', '1型糖尿病': '内分泌科', '甲状腺功能亢进症': '内分泌科',
    '甲状腺功能减退症': '内分泌科', '血脂异常': '内分泌科', '痛风': '内分泌科',
    '库欣综合征': '内分泌科', '嗜铬细胞瘤': '内分泌科', '原发性醛固酮增多症': '内分泌科',
    '艾迪生病': '内分泌科',
    # 血液科
    '缺铁性贫血': '血液科', '巨幼细胞性贫血': '血液科', '免疫性血小板减少症': '血液科',
    '急性髓系白血病': '血液科', '弥漫大B细胞淋巴瘤': '血液科', '多发性骨髓瘤': '血液科',
    '弥散性血管内凝血': '血液科', '阵发性睡眠性血红蛋白尿症': '血液科',
    '噬血细胞综合征': '血液科', '真性红细胞增多症': '血液科',
    # 肾内科 / 泌尿外科
    '泌尿道感染': '肾内科', '肾结石': '肾内科', '急性肾损伤': '肾内科',
    '慢性肾脏病': '肾内科', '良性前列腺增生': '泌尿外科', '肾病综合征': '肾内科',
    '急性肾小球肾炎': '肾内科', 'IgA肾病': '肾内科', '多囊肾': '肾内科',
    'Alport综合征': '肾内科',
    # 风湿免疫科 / 皮肤科
    '系统性红斑狼疮': '风湿免疫科', '类风湿关节炎': '风湿免疫科',
    '强直性脊柱炎': '风湿免疫科', '干燥综合征': '风湿免疫科', '过敏性紫癜': '风湿免疫科',
    '银屑病': '皮肤科', '系统性硬化症': '风湿免疫科', 'IgG4相关性疾病': '风湿免疫科',
    '抗合成酶综合征': '风湿免疫科', '白塞病': '风湿免疫科',
    # 神经科
    '脑梗死': '神经科', '脑出血': '神经科', '偏头痛': '神经科',
    '癫痫': '神经科', '周围神经病变': '神经科', '帕金森病': '神经科',
    '多发性硬化': '神经科', '重症肌无力': '神经科', '视神经脊髓炎谱系疾病': '神经科',
    '进行性核上性麻痹': '神经科',
    # 骨科
    '腰椎间盘突出症': '骨科', '颈椎病': '骨科', '膝关节骨关节炎': '骨科',
    '半月板损伤': '骨科', '肩袖损伤': '骨科', '骨质疏松性椎体压缩骨折': '骨科',
    '痛风性关节炎': '骨科', '滑膜软骨瘤病': '骨科', '复杂区域疼痛综合征': '骨科',
    '弥漫性特发性骨肥厚': '骨科',
    # 五官科
    '急性鼻窦炎': '耳鼻喉科', '化脓性中耳炎': '耳鼻喉科', '急性结膜炎': '眼科',
    '原发性开角型青光眼': '眼科', '突发性耳聋': '耳鼻喉科', '慢性扁桃体炎': '耳鼻喉科',
    '老年性白内障': '眼科', '鼻咽癌': '耳鼻喉科', '梅尼埃病': '耳鼻喉科',
    '鳃裂囊肿': '耳鼻喉科',
}

_LONG_DURATION_KEYWORDS = frozenset([
    '多年', '多月', '多周', '长期', '反复', '年了', '月了', '周了',
    '几年', '几个月', '慢性', '一直', '持续',
])


def get_department(seed: str) -> str:
    return SEED_DEPARTMENT_MAP.get(seed, '内科')


def _has_long_duration(history_entries: list) -> bool:
    for role, text in history_entries:
        if role == '患者' and any(kw in text for kw in _LONG_DURATION_KEYWORDS):
            return True
    return False


# ============================================================
# 模块3 prompt 构建函数
# ============================================================

def _format_history_for_display(history_entries: list) -> str:
    lines = []
    for role, content in history_entries:
        lines.append(f'[{role}]: {content}')
    return '\n'.join(lines)


def format_specific_for_prompt(specific_text) -> str:
    """
    把 COL_SPECIFIC 的 Python 字面量列表转成自然语言段落，按阶段+类型分组。

    输入格式：
      - 4 元组 (name, category, phase, rel_day)

    输出段落结构：
      【起病期（发病至就诊前）】
      - 症状：X、Y
      - 体征：...
      【就诊期（就诊/入院 D0-D1）】
      ...
      【入院后（D2 及以后）】
      ...
    空输入返回空串；无法解析时返回原字符串（安全兜底）。
    """
    raw = specific_text or ''
    if not str(raw).strip():
        return ''
    # 附加文本（如"（本患者侧别：左侧）"）—— 剥离出来，末尾再拼回
    extra_suffix = ''
    main_text = str(raw)
    import re as _re
    paren_tail = _re.search(r'\n?（[^（）]*）\s*$', main_text)
    if paren_tail:
        extra_suffix = paren_tail.group(0).strip()
        main_text = main_text[:paren_tail.start()]

    items = parse_list_from_response(main_text)
    if not items:
        return str(raw)

    _CATS = {'症状', '体征', '实验室检查', '化验检查', '影像检查', '功能检查'}
    _ONSET_PHASES = {'起病', '起病期', 'D-N', 'D-14', 'D-7', 'D-3', 'D-1', '发病'}
    _VISIT_PHASES = {'就诊', '入院', 'D0', 'D+0', 'D1', 'D+1'}

    def _classify(it):
        """返回 (category, phase_bucket, name)。phase_bucket ∈ {'起病','就诊','入院后'}。"""
        if not isinstance(it, (list, tuple)) or not it:
            return '', '入院后', str(it)
        name = str(it[0]).strip()
        cat = ''
        phase_raw = ''
        rel_day = ''
        if len(it) >= 4 and str(it[1]).strip() in _CATS:
            cat = str(it[1]).strip()
            phase_raw = str(it[2]).strip()
            rel_day = str(it[3]).strip()
        else:
            return '', '入院后', ''
        # 归并"化验检查" → "实验室检查"
        if cat == '化验检查':
            cat = '实验室检查'
        # 归类阶段：优先看 rel_day，再看 phase_raw
        rel_num = None
        m = _re.match(r'([HDMY])\s*([+\-]?\d+)', rel_day or phase_raw, _re.IGNORECASE)
        if m:
            try:
                rel_num = int(m.group(2))
            except ValueError:
                rel_num = None
        if rel_num is not None:
            if rel_num < 0:
                bucket = '起病'
            elif rel_num <= 1:
                bucket = '就诊'
            else:
                bucket = '入院后'
        elif any(k in phase_raw for k in _ONSET_PHASES):
            bucket = '起病'
        elif any(k in phase_raw for k in _VISIT_PHASES):
            bucket = '就诊'
        else:
            bucket = '就诊'  # 默认归就诊期
        return cat, bucket, name

    buckets = {'起病': {}, '就诊': {}, '入院后': {}}
    for it in items:
        cat, bucket, name = _classify(it)
        if not name:
            continue
        cat_key = cat or '其他'
        buckets[bucket].setdefault(cat_key, []).append(name)

    _BUCKET_TITLE = {
        '起病': '【起病期（发病至就诊前）】',
        '就诊': '【就诊期（就诊/入院 D0–D1）】',
        '入院后': '【入院后（D2 及以后）】',
    }
    _CAT_ORDER = ['症状', '体征', '实验室检查', '影像检查', '功能检查', '其他']

    # 症状/其他类用顿号合并即可；体征/化验/影像/功能等含具体数值的检查项
    # 必须逐条独立成行呈现，避免顿号合并把 M2 已生成的具体数值挤成名称串，
    # 导致下游 oracle 无法逐项识别、报告不全（体征化验 atom 密度偏低的主因之一）。
    _MERGE_CATS = {'症状', '其他'}

    lines = []
    for bucket in ['起病', '就诊', '入院后']:
        groups = buckets[bucket]
        if not groups:
            continue
        lines.append(_BUCKET_TITLE[bucket])
        for cat in _CAT_ORDER:
            if cat not in groups:
                continue
            names = groups[cat]
            # 去重保序
            seen = set()
            uniq = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    uniq.append(n)
            if cat in _MERGE_CATS:
                lines.append(f'- {cat}：{"、".join(uniq)}')
            else:
                # 检查类：逐条独立成行，保留每项的具体数值
                lines.append(f'- {cat}：')
                for n in uniq:
                    lines.append(f'    · {n}')
    if extra_suffix:
        lines.append(extra_suffix)
    return '\n'.join(lines) if lines else ''


_DISCLOSURE_DIMENSION_KEYWORDS = {
    '诱因': ['诱因', '诱发', '一吹', '受凉', '吃了', '劳累', '一累', '饭后', '空腹'],
    '加重': ['加重', '更厉害', '变重', '严重了'],
    '缓解': ['缓解', '好一点', '减轻', '休息后', '吃药后好'],
    '频率': ['频率', '次数', '一天', '每天', '一阵', '一会儿', '间断'],
    '性质': ['性质', '钝痛', '刺痛', '隐痛', '胀痛', '绞痛', '烧灼', '酸胀', '闷'],
    '放射': ['放射', '串到', '牵扯到', '窜到'],
    '伴随': ['伴随', '同时', '还有', '伴有'],
}


def _summarize_patient_said(history_entries: list, max_items: int = 8) -> str:
    """
    抽取患者此前在对话中已经说过的要点，用于后续 prompt 的"已说过，请勿重复"摘要。
    末尾追加"已披露维度"汇总，提示模型不要重复同维度。
    """
    bullets = []
    full_text = ''
    for role, content in history_entries:
        if role != '患者':
            continue
        line = (content or '').strip().replace('\n', ' ')
        if not line:
            continue
        full_text += line + ' '
        if len(line) > 60:
            line = line[:57] + '...'
        bullets.append(f'- {line}')
        if len(bullets) >= max_items:
            break
    text_blob = full_text
    disclosed = []
    for dim, kws in _DISCLOSURE_DIMENSION_KEYWORDS.items():
        if any(kw in text_blob for kw in kws):
            disclosed.append(dim)
    summary = '\n'.join(bullets)
    if disclosed:
        summary += '\n[已披露的症状维度]：' + '、'.join(disclosed) + '（这些维度本轮不要重复，可补充其他新维度）'
    return summary


def _format_absent_block(absent_symptoms, absent_signs, absent_labs,
                         absent_imaging, absent_functional) -> str:
    """把 absent_* 五类列表渲染成"该患者无"块。空时返回空串。"""
    def _names(lst):
        if not lst:
            return ''
        names = []
        for it in lst:
            if isinstance(it, str):
                if it.strip():
                    names.append(it.strip())
            elif isinstance(it, (list, tuple)) and it:
                names.append(str(it[0]).strip())
        # 去重保序
        seen = set()
        out = []
        for n in names:
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return '、'.join(out)

    parts = []
    s = _names(absent_symptoms)
    if s:
        parts.append(f'  - 症状：{s}')
    s = _names(absent_signs)
    if s:
        parts.append(f'  - 体征：{s}')
    s = _names(absent_labs)
    if s:
        parts.append(f'  - 实验室检查：{s}')
    s = _names(absent_imaging)
    if s:
        parts.append(f'  - 影像检查：{s}')
    s = _names(absent_functional)
    if s:
        parts.append(f'  - 功能检查：{s}')
    return '\n'.join(parts)


def _summarize_doctor_asked(history_entries: list, max_items: int = 10) -> str:
    """
    抽取医生此前问过的问题，用于提示"不要再问相同问题"。
    """
    bullets = []
    for role, content in history_entries:
        if role != '医生':
            continue
        line = (content or '').strip().replace('\n', ' ')
        if not line:
            continue
        if '[申请' in line or '[结束问诊]' in line or '[诊断]' in line:
            continue
        if len(line) > 60:
            line = line[:57] + '...'
        bullets.append(f'- {line}')
        if len(bullets) >= max_items:
            break
    return '\n'.join(bullets)


def _build_doctor_prompt(history_entries: list, age, gender: str,
                          seed: str = '',
                          force_labs: bool = False, is_first: bool = False) -> str:
    dept = get_department(seed) if seed else '内科'
    triage_note = f'【导诊信息】年龄：{age}岁，性别：{gender}\n【就诊科室】{dept}  — 请以{dept}视角为主诊疗，不过分追问与本科室无关的疾病'
    if is_first:
        history_str = '（问诊刚开始）'
        task_str = '请开始问诊，向患者问第一个问题（主诉/症状）。'
    else:
        history_str = _format_history_for_display(history_entries)
        task_str = '请根据患者的回答继续问诊，或在信息足够时申请查体/化验检查。'
    force_msg = ''
    if force_labs:
        force_msg = (
            '\n\n⚠️ 注意：当前问诊已到最大轮数，你必须立即申请查体或化验检查，'
            '格式：[申请查体]: 要查的具体部位（如"心肺听诊"、"胸部叩诊"）  '
            '或  [申请化验]: 请检测：项目1、项目2、...'
        )
    asked_summary = _summarize_doctor_asked(history_entries) if not is_first else ''
    asked_section = (
        f'\n你此前已经问过的问题（严禁本轮再问相同或同义的问题，请换角度追问新的信息）：\n{asked_summary}\n'
        if asked_summary else ''
    )
    long_duration_hint = ''
    if not is_first and _has_long_duration(history_entries):
        long_duration_hint = (
            '\n⚠️ 患者症状病程较长，请在适当时机追问既往诊治经历'
            '（曾就诊于哪里、做过哪些检查、用过哪些药物及疗效）。\n'
        )
    return f"""你是一位有经验的{dept}医生，正在对一位门诊患者进行问诊。{triage_note}

当前对话记录：
{history_str}
{long_duration_hint}{asked_section}
{task_str}

规则：
1. 每轮严格只问 1 个问题，不能在同一句话里连问多件事
2. 问题必须简洁直接，不加括号、不加解释说明，像医生日常问诊那样说话（例如："您咳嗽多久了？" "咳嗽的时候有没有痰？" "最近有没有发烧？"）
3. 问诊顺序：主诉 → 发病时间 → 症状特点 → 伴随症状 → 既往史/手术史/用药史 → 个人史/家族史等（问诊尽量在 10 轮内完成，最多不超过 15 轮）
   - 既往史（是否有慢性病、手术史、过敏史、长期用药）必须在问诊中询问，不可省略
   - 既往就诊与诊疗信息也要主动追问（可分散在问诊全过程中按需追问，不必一轮问完）：
     · 既往诊断：过去是否曾被诊断过其他疾病、分别是什么；
     · 既往化验/影像/功能检查：过去做过哪些化验、影像、功能学检查，结果如何；
     · 既往治疗：过去接受过哪些药物治疗、手术或非药物干预，疗效如何。
4. **严禁重复上方已经问过的问题**；若患者此前的回答已覆盖某项信息，请直接跳到下一个未询问的维度。
5. 当你认为问诊信息已足够时，可以：
   a) 申请查体（问诊阶段可申请一次），格式——写明要查的具体部位，例如：
      [申请查体]: 心肺听诊
      [申请查体]: 胸部叩诊、肺部听诊
   b) 申请化验检查（问诊阶段只能申请一次）：
      [申请化验]: 请检测：项目1、项目2、...
      申请原则：主诊断和所有合并症相关检查合计最多 8 项；若表现典型、诊断方向已经明确：1-2 项核心检查即可；若表现不典型或需鉴别重要疾病：可申请 3-4 项
      · 合并症监测（重要）：若患者在问诊中陈述了慢性合并症/既往病史，除主诊断相关检查外，应针对每项合并症一并申请其常规监测检查，以评估其当前控制/影响情况。例如：
        - 糖尿病 → 空腹血糖、糖化血红蛋白(HbA1c)
        - 高血压 → 复测血压（可通过查体）、必要时肾功能/电解质
        - 高脂血症/血脂偏高 → 血脂四项(TC、TG、LDL-C、HDL-C)
        - 慢阻肺/哮喘 → 肺功能、血气分析
        - 慢性肝病 → 肝功能(ALT、AST、胆红素、白蛋白)
        - 慢性肾病/肾功能不全 → 肾功能(肌酐、尿素氮、eGFR)、尿常规
        - 冠心病/房颤 → 心电图、心肌标志物
        合并症监测项计入总数，必须与主诊断相关检查合计不超过 8 项；超出时仅保留最能改变诊断或评估风险的项目。
   c) 同时申请查体和化验（合并在一条回复中，分行输出）
6. 直接输出你说的话，不要在最前面加 "[医生]:" 等角色标签
7. 问诊阶段严禁直接索要化验/影像/功能检查的具体数值（如"血常规多少"、"D二聚体是多少"）；你只能申请检查，结果由检验室返回。既往检查可以问"做过哪些检查"，但不要追问具体数值。{force_msg}"""


def _build_patient_prompt(age, gender: str, specific_text: str,
                           diagnosis: str,
                           comorbidities_str: str, history_entries: list,
                           duration_total: str = '',
                           chief_complaint: str = '',
                           symptom_attrs_text: str = '',
                           absent_symptoms: list = None,
                           absent_signs: list = None,
                           absent_labs: list = None,
                           absent_imaging: list = None,
                           absent_functional: list = None,
                           prior_visited: str = '',
                           prior_visit_count: int = 0,
                           prior_visit_history: str = '',
                           patient_view: dict | None = None) -> str:
    specific_text = format_specific_for_prompt(specific_text)
    history_str = _format_history_for_display(history_entries)
    is_first_response = (len(history_entries) <= 1)
    if patient_view is not None:
        view_patient = patient_view.get('patient') if isinstance(patient_view.get('patient'), dict) else {}
        view_course = patient_view.get('course') if isinstance(patient_view.get('course'), dict) else {}
        view_context = patient_view.get('patient_context') if isinstance(patient_view.get('patient_context'), dict) else {}
        prior_visit = view_context.get('prior_visit') if isinstance(view_context.get('prior_visit'), dict) else {}
        symptom_attrs_text = _format_symptom_attributes_for_prompt(
            view_context.get('symptom_attributes') or []
        )
        prior_visited = prior_visit.get('visited') or ''
        prior_visit_count = prior_visit.get('count') or 0
        prior_visit_history = prior_visit.get('history') or ''
        age = view_patient.get('age') or age
        gender = view_patient.get('sex') or gender
        diagnosis = view_patient.get('diagnosis') or diagnosis
        duration_total = view_course.get('current_episode_duration') or ''
        chief_complaint = ''

    # 就诊状态块
    visit_status_block = ''
    if prior_visited:
        if '未就诊' in str(prior_visited) or str(prior_visited).strip() in ('未', '否', '0'):
            visit_status_block = '\n就诊状态：未就诊（此前没有因这次的疾病去医院看过）\n'
        else:
            try:
                cnt = int(prior_visit_count) if prior_visit_count else 0
            except (ValueError, TypeError):
                cnt = 0
            if cnt > 0:
                visit_status_block = f'\n就诊状态：此前已就诊过 {cnt} 次\n'
            else:
                visit_status_block = f'\n就诊状态：{prior_visited}\n'

    # 既往就诊经历块
    prior_history_block = ''
    if prior_visit_history and str(prior_visit_history).strip():
        prior_history_block = (
            f'\n既往就诊经历：\n{str(prior_visit_history).strip()}\n'
            '（如医生问到既往就医情况，可参照上述经历自然作答；不要主动一次性全部倒出）\n'
            '（当医生具体问到既往诊断/既往化验或影像/既往治疗时，请基于上述既往就诊经历如实作答；'
            '不要编造未在上述经历中出现的诊断、检查或治疗）\n'
        )
    else:
        prior_history_block = (
            '\n既往就诊经历：此前未曾就诊\n'
            '（当医生问到既往诊断/既往化验或影像/既往治疗时，请明确回答"未曾就诊，没有做过相关检查/没有接受过相关治疗"；'
            '不要编造既往诊疗经历）\n'
        )

    # 双轨"有/无"块
    absent_text = _format_absent_block(
        absent_symptoms, absent_signs, absent_labs, absent_imaging, absent_functional
    )
    dual_track_block = ''
    if absent_text:
        dual_track_block = (
            '\n=== 该患者有的临床表现（你真实有的，可在被问到时承认） ===\n'
            f'{specific_text}\n'
            '=== 该患者无的临床表现（不要在你的回答中表现出这些） ===\n'
            f'{absent_text}\n'
        )

    if chief_complaint and is_first_response:
        chief_hint = (
            f'\n你此次来看病的主要原因（主诉）：{chief_complaint}\n'
            f'（首次回答时，请围绕这个主诉描述你的主要不适，不要一开口就列出所有症状）'
        )
    else:
        chief_hint = ''
    attrs_section = ''
    if symptom_attrs_text and symptom_attrs_text.strip():
        attrs_section = (
            f'\n你各症状的具体特点（用于医生追问细节时参考，必须转成患者口语，不要照搬这些医学词语）：\n'
            f'{symptom_attrs_text}\n'
            f'（上面未列出的要素——如"部位""程度""缓解方式"——请你结合自身生活情境自洽地即兴补充，'
            f'做到具体、个性化，不同症状/不同患者的诱因与缓解方式尽量各不相同。）\n'
        )
    said_summary = _summarize_patient_said(history_entries) if not is_first_response else ''
    said_section = (
        f'\n你此前已经告诉过医生的内容（以下为要点摘要，**本轮不要重复**同样的描述）：\n{said_summary}\n'
        if said_summary else ''
    )
    duration_line = f'\n患病总时长：{duration_total}\n' if duration_total else ''
    if patient_view is not None:
        ledger_patient_block = _format_patient_view_for_prompt(patient_view)
        return f"""你是一位正在门诊就诊的患者。你的真实情况如下（不要主动说出诊断名称）：
年龄：{age}岁，性别：{gender}{duration_line}{visit_status_block}{prior_history_block}{chief_hint}
你知道自己的疾病诊断：{diagnosis}（这是本次就诊的主要诊断，绝对不能说出这个名称，即使医生直接问"您是不是XX病"也只说"我不清楚"或"医生还没告诉我最终结果"。既往合并症（高血压、糖尿病等其他已知慢性病）被问到时可以如实说。）
你可使用的患者视角事实（来自事实账本投影；不包含化验、影像或功能检查具体结果）：
{ledger_patient_block}{attrs_section}
既往疾病史：{comorbidities_str if comorbidities_str else '无特殊既往病史'}

当前对话记录：
{history_str}
{said_section}
请以真实患者的口语方式，回答医生最后一个问题。
规则：
1. 每次回答只描述 1-2 个你感受到的症状，不要一次倾诉太多
2. 医生问的属于患者视角事实清单 → 如实承认，先说症状名 + 1 个最关键属性
3. 当医生明确追问某个要素时，再依据上方症状特点补充该要素，每轮只透露 1~2 个新维度
4. 未在患者视角事实中出现的检查结果、化验数值、影像结论、功能检查结论一律不能说
5. 化验、影像、功能检查的具体数值和结论一律不说，即使医生直接追问也只说"不记得了"或"医生说偏高/偏低"，不说具体数值
6. 回答简洁自然，1-3 句话
7. 直接输出你说的话，不要在最前面加 "[患者]:" 等角色标签"""
    return f"""你是一位正在门诊就诊的患者。你的真实情况如下（不要主动说出诊断名称）：
年龄：{age}岁，性别：{gender}{duration_line}{visit_status_block}{prior_history_block}{chief_hint}
你知道自己的疾病诊断：{diagnosis}（这是本次就诊的主要诊断，绝对不能说出这个名称，即使医生直接问"您是不是XX病"也只说"我不清楚"或"医生还没告诉我最终结果"。既往合并症（高血压、糖尿病等其他已知慢性病）被问到时可以如实说。）
你实际有的症状/体征/检查结果（仅供你参考，必须用通俗的患者语言表达，不能用医学术语）：
{specific_text}{attrs_section}{dual_track_block}
既往疾病史：{comorbidities_str if comorbidities_str else '无特殊既往病史'}

当前对话记录：
{history_str}
{said_section}
请以真实患者的口语方式，回答医生最后一个问题。
规则：
1. 每次回答只描述 1-2 个你感受到的症状，不要一次倾诉太多
2. 医生问的属于"该患者有"清单 → 如实承认，**先说症状名 + 1 个最关键属性**（最影响生活的、或最早出现的、或最严重的那一个），不要一次性把诱因/加重/缓解/频率/性质/放射/伴随等所有要素一次性倒出
3. 当医生明确追问某个要素（多久了、什么情况下加重、缓解方式、什么样的感觉、有没有放射、还有什么不舒服）时，再依据上方"各症状的具体特点"补充该要素，**每轮只透露 1~2 个新维度**
   - 尤其被问到【诱因/什么情况下加重】和【怎样能缓解】时，请结合你自己的具体生活场景作答（如上班久坐、爬楼、受凉、饭后、夜里平躺、天气变化、干活累着等，或吃了某种药/含片、喝热水、按压、改变体位、开窗通风等），给出**具体、个性化**的描述，避免"活动后加重、休息就好"这类笼统套话，不同患者应答出不同的场景细节。
4. 医生问的属于"该患者无"清单 → 用明确语气否认（如"没有"、"不会"、"这个倒没有"）
5. 医生问的两表都没有的项 → 根据自己的人物设定（年龄/性别/既往就诊经历/合并症）做自洽合理的回答
6. **不要重复上方"已经告诉过医生的内容"**；如果医生问的其实是你此前已回答过的相同问题，请用简短确认一句（如"跟刚才说的一样"），并尽量补充**新的维度**或更具体的细节
7. 化验、影像、功能检查的具体数值和结论一律不说（如血常规数值、D-二聚体、影像描述、病理结果等），即使医生直接追问也只说"不记得了"或"医生说偏高/偏低"，不说具体数值。居家可自测的生命体征（血压、脉搏/心率、血氧饱和度、体温）患者本人可以知道，被问到时可以说出数值。
8. 回答简洁自然，1-3 句话
9. 直接输出你说的话，不要在最前面加 "[患者]:" 等角色标签"""


def _build_physical_exam_prompt(age, gender: str, diagnosis: str,
                                 signs_specific_text: str,
                                 comorbidities_str: str,
                                 exam_request: str = '',
                                 absent_signs: list = None,
                                 patient_said: str = '',
                                 physical_exam_view: dict | None = None) -> str:
    if physical_exam_view is not None:
        projected = _format_fact_list_for_prompt(physical_exam_view.get('facts') or [], include_values=True)
        return f"""你是医院查体室/检诊室。只能依据下方事实账本的查体/生命体征投影作答，不得按诊断补造新阳性体征。

医生申请的查体范围：{exam_request.strip() if exam_request else '围绕当前主诉进行重点查体'}

事实账本查体投影：
{projected}

未知或未列出的项目按正常缺省输出。请按固定五段输出查体结果，不要在最前面加 [查体结果] 角色标签。"""
    absent_signs_text = _format_absent_block(None, absent_signs, None, None, None)
    # 去掉"  - 体征："前缀仅留名称列表
    if absent_signs_text.startswith('  - 体征：'):
        absent_signs_text = absent_signs_text[len('  - 体征：'):]
    comorbidity_note = ''
    if comorbidities_str:
        comorbidity_note = (
            f'\n注意：该患者合并以下伴随疾病：{comorbidities_str}。'
            f'若与这些伴随疾病相关的体征有可能出现，请结合伴随疾病给出符合临床实际的结果。'
        )
    patient_said_note = ''
    if patient_said and patient_said.strip():
        patient_said_note = (
            f'\n=== 患者本人在问诊中已明确陈述的症状/主诉（体征必须与之一致） ===\n'
            f'{patient_said.strip()}\n'
            f'一致性硬约束：凡患者已陈述存在的症状，相应查体体征必须给出与该症状相符的结果，'
            f'不得报"正常/无异常"。例如患者诉发热→体温须偏高；诉脚踝肿胀/下肢水肿→相应部位须见水肿；'
            f'诉咽痛→咽部须见充血；诉气短/呼吸费力→呼吸频率或肺部听诊须有相应异常。'
            f'反之，患者明确否认的症状，相应体征应为正常。\n'
        )
    if exam_request and exam_request.strip():
        scope_instruction = (
            f'医生申请的查体范围：{exam_request.strip()}\n\n'
            f'重要：只报告上述申请范围内的查体结果，不要输出其他未申请的部位；'
            f'但【生命体征】无论是否在申请范围内均须首先给出。'
            f'未申请的子段必须仍输出标题行，其下写一行 "- 无"。'
        )
    else:
        scope_instruction = '医生未指定查体范围，请给出完整查体报告（5 个子段必须全部输出）。'
    return f"""你是医院查体室/检诊室，掌握该患者的全部查体数据。

患者信息：年龄 {age} 岁，性别 {gender}，诊断 {diagnosis}
该患者的已知查体体征（具体数值，"该患者有"清单）：
{signs_specific_text if signs_specific_text else '（无已知异常体征）'}
=== 该患者无的体征（应按正常输出，不得报告异常） ===
{absent_signs_text if absent_signs_text else '（无）'}
伴随疾病：{comorbidities_str if comorbidities_str else '无'}{comorbidity_note}{patient_said_note}

{scope_instruction}

数据规则：
1. 上述"该患者有"清单中有记录的项目 → 直接报告该具体数值（不得修改）
2. 上述"该患者无"清单中的项目 → 必须按正常结果输出
3. 两个清单都没有的项目 → 根据患者底层设定（年龄/性别/诊断/合并症）做自洽合理输出，默认正常

==== 输出格式硬约束（必须严格遵守，便于后续按行数自动统计） ====
请严格按以下结构输出，不得有任何自由叙述、不得合并行；每一条结果**独占一行**，行首一律以 "- " 起始，紧跟"项目: 结果"格式。

【生命体征】
- T: 36.8℃
- P: 75次/分
- R: 16次/分
- BP: 120/80mmHg
- SpO2: 98%
【一般状态】
- 神志: 清楚
- 一般情况: 良好
- 体型: 正常
【胸部查体】
- 视诊: 双侧对称
- 触诊: 触诊无异常
- 叩诊: 清音
- 肺部听诊: 呼吸音清
- 心脏听诊: 心律齐
【腹部查体】
- 视诊: 平坦
- 触诊: 软，无压痛
- 叩诊: 鼓音
- 听诊: 肠鸣音正常
【其他系统】
- 神经系统: 无异常
- 皮肤黏膜: 无异常
- 浅表淋巴结: 未触及肿大

附加规则：
- 5 个子段标题必须全部出现且顺序固定
- 该子段未发现异常时，每项写 "- 项目: 无异常" 或类似具体描述（不得只写"无"或省略）
- 与主诊断相关的重点系统（如呼吸科重点在胸部、心内科重点在心脏）须按「视诊、触诊、叩诊、听诊」四步分别独立成行详细描述，阳性体征给出部位/程度/性质等具体细节，不得笼统一句带过
- 医生申请范围之外的查体子段，仍输出子段标题，其下只写一行 "- 无"
- 不得在子段标题外、行首非 "- " 写任何文本
- 直接输出查体报告内容，不要在最前面加 [查体结果] 角色标签"""


def _build_oracle_prompt(age, gender: str, diagnosis: str, specific_text: str,
                          comorbidities_str: str, lab_request: str,
                          absent_labs: list = None,
                          absent_imaging: list = None,
                          absent_functional: list = None,
                          patient_said: str = '',
                          diagnostic_view: dict | None = None) -> str:
    if diagnostic_view is not None:
        projected = render_diagnostic_view(diagnostic_view)
        return f"""你是医院检验科/影像科/功能检查室。只能返回医生本轮明确申请且事实账本匹配到的检查事实。

医生申请的检查：
{lab_request}

事实账本检查投影：
{projected}

必须按【化验检查】【影像检查】【功能学检查】三段输出；未申请或未匹配到账本事实的子段写 "- 无"。不要补造未在账本中的检查结果。"""
    specific_text = format_specific_for_prompt(specific_text)
    comorbidity_lab_note = ''
    if comorbidities_str:
        comorbidity_lab_note = (
            f'\n注意：该患者合并以下伴随疾病：{comorbidities_str}。'
            f'若医生申请的检查项目与这些伴随疾病相关，请结合伴随疾病给出符合临床实际的异常结果。'
        )
    patient_said_note = ''
    if patient_said and patient_said.strip():
        patient_said_note = (
            f'\n=== 患者本人在问诊中已明确陈述的症状/主诉（检查结果必须与之一致） ===\n'
            f'{patient_said.strip()}\n'
            f'一致性硬约束：凡患者已陈述存在的症状，相应化验/影像检查结果必须与该症状相符，'
            f'不得报正常。例如患者诉咯血/明显失血→血红蛋白应偏低；诉血尿→尿潜血应阳性且尿红细胞升高；'
            f'诉发热→炎症指标可相应升高。同一指标的不同表述（如脉搏与心电图心率）之间数值须一致。\n'
        )
    absent_text = _format_absent_block(None, None, absent_labs, absent_imaging, absent_functional)
    absent_block = ''
    if absent_text:
        absent_block = (
            '\n=== 该患者无的（应正常）检查项目 ===\n'
            f'{absent_text}\n'
        )
    return f"""你是医院检验科/影像科，掌握该患者的全部检查数据。

患者信息：年龄 {age} 岁，性别 {gender}，诊断 {diagnosis}
已知检查结果（该患者实际数据，"该患者有"清单）：
{specific_text}
伴随疾病：{comorbidities_str if comorbidities_str else '无'}{comorbidity_lab_note}{absent_block}{patient_said_note}

医生申请的检查：
{lab_request}

请逐项给出检查结果。

数据规则：
1. 申请的项目在"该患者有"清单（已知检查结果）中有记录 → 直接报告该具体数值（不得修改）
2. 申请的项目在"该患者无"清单中 → 必须报告正常结果，不得输出异常
3. 申请的项目两表都没有，但与伴随疾病相关 → 按伴随疾病的典型异常值报告；例如糖尿病应报空腹血糖/HbA1c升高，高脂血症应报TC/TG/LDL-C升高，慢性肾病应报肌酐/尿素氮升高等，使检查结果能明确体现该伴随疾病的存在，避免全部落在正常范围。
4. 申请的项目两表都没有、也与伴随疾病无关 → 根据患者底层设定（年龄/性别/诊断/合并症）做自洽合理输出，默认在正常参考范围内
5. 影像检查只描述本次单次检查可直接观察到的静态表现，严禁出现游走性、进展性、较前增大/缩小等动态描述
6. 一致性要求：所报结果不得与患者已陈述症状、与"该患者有"清单、以及本次已报告的其它项目相互矛盾；同一生理量的不同表述（如脉搏与心电图心率、SpO2与血气SaO2、RBC与HCT）之间必须数值/方向一致。

==== 输出格式硬约束（必须严格遵守，便于后续按行数自动统计） ====
请严格按以下三个子段输出，**子段标题必须全部出现且顺序固定**；每条结果**独占一行**，行首一律以 "- " 起始，紧跟"项目: 结果"格式。

【化验检查】
- 项目名: 数值 单位 (参考范围或↑↓)
- 项目名: 数值 单位 (参考范围或↑↓)
...
【影像检查】
- 检查名: 影像所见（含所有扫描结构含正常部分的完整描述，不写诊断结论）
...
【功能学检查】
- 检查名: 关键参数与结论（如肺功能必含 VC/FVC/FEV1/FEV1%FVC/DLCO 等，写在同一行内用逗号分隔）
...

附加规则：
- 医生未申请的子段：必须仍输出子段标题，其下只写一行 "- 无"
- 化验项每行只写一项；多项目时每行独立成行（不得用 "、" 或分号合并）
- 数值/结论必须具体，不写"待查/未做"
- 化验检查：申请的每个化验套餐须报**全套参数**，不得只报异常项。即便某项在"已知检查结果"中未列出，也须按该患者底层设定给出正常范围内的具体数值，逐项独立成行。各套餐的必报参数清单如下（申请到哪个套餐就必须至少覆盖对应清单）：
    · 血常规：WBC、RBC、HGB、HCT、MCV、MCH、MCHC、RDW、PLT、MPV，以及白细胞分类（中性粒细胞%/绝对值、淋巴细胞%/绝对值、单核%、嗜酸%、嗜碱%）
    · 肝功能：ALT、AST、ALP、GGT、总胆红素、直接胆红素、总蛋白、白蛋白、球蛋白、A/G
    · 肾功能：尿素、肌酐、尿酸、eGFR、胱抑素C
    · 电解质：Na、K、Cl、Ca、Mg、P、HCO3
    · 血气分析：pH、PaO2、PaCO2、HCO3、BE、SaO2、乳酸
    · 心肌标志物：肌钙蛋白I/T、CK、CK-MB、肌红蛋白、BNP或NT-proBNP
    · 凝血：PT、INR、APTT、TT、纤维蛋白原、D-二聚体
    · 炎症：CRP、hs-CRP、PCT、ESR
    · 尿常规：比重、pH、蛋白、葡萄糖、酮体、隐血、白细胞、红细胞、管型
  已在"已知检查结果"中给出具体数值的项目，必须**原样采用该数值**，其余同套餐参数补全为自洽的正常值。
- 影像检查：须按正规影像报告格式，只写「所见：」（含所有扫描到的结构含正常部分的完整描述，逐个解剖结构分别描述，不少于 5 个结构/征象）；不写诊断结论，正常所见不得省略
- 功能学检查：必须列全该检查的所有关键量化参数（如肺功能须含 VC、FVC、FEV1、FEV1%、FEV1/FVC、PEF、DLCO、TLC、RV 等），逐项给出数值
- 不得在子段标题外、行首非 "- " 写任何文本
- 直接输出检查结果，不要在最前面加 [辅助检查] 等角色标签"""


def _build_doctor_post_lab_prompt(history_entries: list, age, gender: str,
                                   seed: str = '') -> str:
    history_str = _format_history_for_display(history_entries)
    dept = get_department(seed) if seed else '内科'
    triage_note = f'【导诊信息】年龄：{age}岁，性别：{gender}  【就诊科室】{dept}'
    return f"""你是一位{dept}医生。{triage_note}以下是你对患者的完整问诊记录（含查体/化验结果）：

{history_str}

请综合以上信息，从以下三种情况中选择一种作出回应：

【情况一：继续向患者问诊以核实诊断（优先考虑此项）】
适用于以下任一情形：①症状细节尚未充分确认；②存在需要排除的重要鉴别诊断；③结果已提示方向但仍有关键临床特征未经患者亲口确认；④需要进一步了解病程、诱因或既往治疗反应。
仅输出，本回复不要给出诊断：
[继续问诊]: 你要问患者的 1 个具体问题，问题直接简洁，不加括号或注释说明

【情况二：信息已非常充分，直接给出诊断】
仅在临床表现与查体/化验结果均已典型、与诊断高度吻合、无需任何补充确认时才选此项。
直接输出（严格按此格式，不加其他内容）：
[诊断]: （最可能的具体诊断名）
[鉴别诊断]: （2-3 个鉴别诊断及鉴别要点）

【情况三：还需要补充查体或化验检查（不再追问患者，直接申请）】
仅在现有结果确实无法做出诊断、且继续问诊也无法解决问题时才选此项；仅输出，本回复不要给出诊断：
[申请查体]: 要查的具体部位
[申请化验]: 请检测：（不超过 2 项，只申请能改变诊断结论的关键检查；若患者有合并症而此前未查其监测指标，可优先补此项，如糖尿病查血糖/HbA1c、血脂高查血脂四项等）

重要：每次只选择一种情况，不要混用；情况一和三本回复中不包含 [诊断]。
直接输出内容，不要在最前面加 "[医生]:" 标签。"""


def _build_doctor_followup_prompt(history_entries: list, age, gender: str,
                                   seed: str = '',
                                   force_labs: bool = False) -> str:
    dept = get_department(seed) if seed else '内科'
    triage_note = f'【导诊信息】年龄：{age}岁，性别：{gender}  【就诊科室】{dept}'
    history_str = _format_history_for_display(history_entries)
    force_msg = ''
    if force_labs:
        force_msg = (
            '\n\n⚠️ 注意：追加问诊已到最大轮数，请立即结束追问。'
            '若需补充查体，用格式：[申请查体]: 要查的具体部位；'
            '若需补充化验，用格式：[申请化验]: 请检测：项目1、项目2（不超过 2 项）；'
            '若已够，直接回复：[结束问诊]'
        )
    asked_summary = _summarize_doctor_asked(history_entries)
    asked_section = (
        f'\n你此前已经问过的问题（**严禁本轮再问相同或同义的问题**）：\n{asked_summary}\n'
        if asked_summary else ''
    )
    return f"""你是一位{dept}医生，正在对患者进行追加问诊（已收到第一轮查体/化验结果）。{triage_note}

当前完整对话记录（含查体/化验结果）：
{history_str}
{asked_section}
请根据患者的回答，选择下一步行动（三选一）：

1. 继续追问，直接输出你要问的 1 个问题，问题简洁直接，不加括号或注释说明，不加标签
2. 申请补充查体或化验并结束追问，仅在现有结果不足以诊断时才选，格式：
   [申请查体]: 要查的具体部位
   [申请化验]: 请检测：项目1、项目2（不超过 2 项）
3. 已获得足够信息，终止追加问诊，格式：[结束问诊]

直接输出内容，不要在最前面加 "[医生]:" 标签。{force_msg}"""


def _build_doctor_diagnosis_prompt(history_entries: list) -> str:
    history_str = _format_history_for_display(history_entries)
    return f"""你是一位临床医生。以下是你对患者的完整问诊记录（含全部化验结果）：

{history_str}

请根据以上所有信息，给出最终临床判断，严格按以下格式输出：
[诊断]: （最可能的诊断，需写明具体诊断名）
[鉴别诊断]: （列出 2-3 个需要鉴别的疾病，并简述每个的鉴别要点）

直接输出诊断内容，不要在最前面加 "[医生]:" 等角色标签。"""


# ============================================================
# 模块3
# ============================================================

def module_3_interaction(seed_text, age, gender, diagnosis, specific_text,
                          comorbidities_text,
                          signs_specific_text='',
                          chief_complaint='',
                          symptom_attrs_text='',
                          absent_symptoms=None, absent_signs=None,
                          absent_labs=None, absent_imaging=None,
                          absent_functional=None,
                          prior_visited='', prior_visit_count=0,
                          prior_visit_history='',
                          duration_total='',
                          max_turns=None, max_followup_turns=None,
                          on_update=None, tag='模块3',
                          fact_ledger=None, ledger_metadata=None):
    """
    模块3 医生-患者-上帝三方交互问诊（多轮 GPT 调用）

    Returns:
        str: 完整交互历史字符串；失败时返回 None
    """
    if max_turns is None:
        max_turns = DEFAULT_MAX_INTERACTION_TURNS
    if max_followup_turns is None:
        max_followup_turns = DEFAULT_MAX_FOLLOWUP_TURNS

    t0 = time.time()
    print(f'    [{tag}] 开始三方交互问诊（首轮最大 {max_turns} 轮，追加最大 {max_followup_turns} 轮）...')

    comorbidities_list = parse_list_from_response(comorbidities_text)
    comorbidities_str = '、'.join([
        (item if isinstance(item, str) else item[0])
        for item in comorbidities_list if item
    ])

    patient_view = None
    physical_exam_view = None
    if fact_ledger is not None:
        pfl = _ledger_module()
        patient_view = pfl.project_patient_view(fact_ledger)
        physical_exam_view = pfl.project_physical_exam_view(fact_ledger)

    history_entries = []
    exam_done = False
    lab_done = False
    last_lab_request_text = ''

    def _notify():
        if on_update:
            on_update(_format_history_for_display(history_entries))

    def _ensure_request_recorded(marker, request_text):
        if any(
            role == '医生' and _has_marked_section(content, marker)
            for role, content in history_entries
        ):
            return
        content = (request_text or '').strip()
        if not _has_marked_section(content, marker):
            content = f'[{marker}]：{content}'
        history_entries.append(('医生', content))
        _notify()

    def _physical_exam_call(round_tag, doctor_response=''):
        nonlocal exam_done
        exam_request = _extract_marked_section(doctor_response, '申请查体') or '围绕当前主诉进行重点查体'
        _ensure_request_recorded('申请查体', doctor_response or exam_request)
        if physical_exam_view is not None:
            resp = render_physical_exam_result(physical_exam_view, exam_request)
        else:
            ep = _build_physical_exam_prompt(
                age, gender, diagnosis, signs_specific_text, comorbidities_str,
                exam_request=exam_request,
                absent_signs=absent_signs,
                patient_said=_summarize_patient_said(history_entries),
            )
            resp = call_gpt5(ep, tag=round_tag)
        history_entries.append(('查体结果', resp.strip() if resp else '（查体结果获取失败）'))
        exam_done = True
        _notify()
        print(f'    [{tag}] {round_tag} 查体结果已获取 ({round(time.time() - t0, 1)}s)')

    def _oracle_call(lab_request_text, round_tag):
        nonlocal lab_done
        if lab_done:
            return False
        lab_request_text = _cap_laboratory_request(lab_request_text)
        _ensure_request_recorded('申请化验', lab_request_text)
        if fact_ledger is not None:
            diagnostic_view = _ledger_module().project_diagnostic_view(fact_ledger, lab_request_text)
            resp = render_diagnostic_view(diagnostic_view)
        else:
            op = _build_oracle_prompt(
                age, gender, diagnosis, specific_text,
                comorbidities_str, lab_request_text,
                absent_labs=absent_labs,
                absent_imaging=absent_imaging,
                absent_functional=absent_functional,
                patient_said=_summarize_patient_said(history_entries),
            )
            resp = call_gpt5(op, tag=round_tag)
        history_entries.append(('辅助检查', resp.strip() if resp else '（辅助检查结果获取失败）'))
        lab_done = True
        _notify()
        print(f'    [{tag}] {round_tag} 辅助检查结果已获取 ({round(time.time() - t0, 1)}s)')
        return True

    def _final_diagnosis():
        dp = _build_doctor_diagnosis_prompt(history_entries)
        dr = call_gpt5(dp, tag=f'{tag}-医生-诊断')
        history_entries.append(('医生诊断', dr.strip() if dr else '（诊断生成失败）'))
        _notify()

    # 阶段1：医生开场 + 主循环
    doctor_prompt = _build_doctor_prompt([], age, gender, seed=seed_text, is_first=True)
    doctor_response = call_gpt5(doctor_prompt, tag=f'{tag}-医生-开场')
    if not doctor_response:
        print(f'    [{tag}] ❌ 医生开场失败')
        return None
    doctor_response = _cap_laboratory_request(doctor_response)
    history_entries.append(('医生', doctor_response.strip()))
    _notify()

    labs_requested = False
    for turn in range(1, max_turns + 1):
        patient_prompt = _build_patient_prompt(
            age, gender, specific_text, diagnosis, comorbidities_str, history_entries,
            duration_total=duration_total,
            chief_complaint=chief_complaint,
            symptom_attrs_text=symptom_attrs_text,
            absent_symptoms=absent_symptoms, absent_signs=absent_signs,
            absent_labs=absent_labs, absent_imaging=absent_imaging,
            absent_functional=absent_functional,
            prior_visited=prior_visited,
            prior_visit_count=prior_visit_count,
            prior_visit_history=prior_visit_history,
            patient_view=patient_view,
        )
        patient_response = call_gpt5(patient_prompt, tag=f'{tag}-患者-{turn}')
        if not patient_response:
            break
        history_entries.append(('患者', patient_response.strip()))
        _notify()

        force_labs = (turn >= max_turns)
        doctor_prompt = _build_doctor_prompt(history_entries, age, gender,
                                              seed=seed_text,
                                              force_labs=force_labs, is_first=False)
        doctor_response = call_gpt5(doctor_prompt, tag=f'{tag}-医生-{turn + 1}')
        if not doctor_response:
            break
        doctor_response = _cap_laboratory_request(doctor_response)
        history_entries.append(('医生', doctor_response.strip()))
        _notify()

        if _has_marked_section(doctor_response, '申请查体') and not exam_done:
            _physical_exam_call(f'{tag}-查体室-1', doctor_response=doctor_response)
            if not _has_marked_section(doctor_response, '申请化验'):
                post_exam_prompt = _build_doctor_prompt(history_entries, age, gender,
                                                         seed=seed_text,
                                                         force_labs=False, is_first=False)
                post_exam_response = call_gpt5(post_exam_prompt, tag=f'{tag}-医生-查体后')
                if post_exam_response:
                    post_exam_response = _cap_laboratory_request(post_exam_response)
                    history_entries.append(('医生', post_exam_response.strip()))
                    _notify()
                    if _has_marked_section(post_exam_response, '申请化验'):
                        last_lab_request_text = post_exam_response.strip()
                        labs_requested = True
                        break

        if _has_marked_section(doctor_response, '申请化验'):
            last_lab_request_text = doctor_response.strip()
            labs_requested = True
            break

    # 阶段2：统一执行一轮查体和一轮化验/辅助检查
    if not exam_done:
        _physical_exam_call(
            f'{tag}-查体室-1',
            doctor_response='[申请查体]：围绕当前主诉进行重点查体',
        )
    if not lab_done and history_entries:
        lab_request_text = (
            last_lab_request_text
            if labs_requested and last_lab_request_text
            else '围绕当前主诉与鉴别诊断选择必要的化验和辅助检查'
        )
        _oracle_call(lab_request_text, f'{tag}-化验室-1')

    # 阶段3：医生评估后决策
    post_lab_prompt = _build_doctor_post_lab_prompt(history_entries, age, gender, seed=seed_text)
    post_lab_response = call_gpt5(post_lab_prompt, tag=f'{tag}-医生-评估')
    if post_lab_response:
        post_lab_response = _cap_laboratory_request(post_lab_response)

    if not post_lab_response:
        _final_diagnosis()
    elif '[继续问诊]' in post_lab_response:
        if max_followup_turns > 0:
            history_entries.append(('医生', post_lab_response.strip()))
            _notify()

        second_labs_requested = False
        for fturn in range(1, max_followup_turns + 1):
            patient_prompt = _build_patient_prompt(
                age, gender, specific_text, diagnosis, comorbidities_str, history_entries,
                duration_total=duration_total,
                chief_complaint=chief_complaint,
                symptom_attrs_text=symptom_attrs_text,
                absent_symptoms=absent_symptoms, absent_signs=absent_signs,
                absent_labs=absent_labs, absent_imaging=absent_imaging,
                absent_functional=absent_functional,
                prior_visited=prior_visited,
                prior_visit_count=prior_visit_count,
                prior_visit_history=prior_visit_history,
                patient_view=patient_view,
            )
            followup_patient = call_gpt5(patient_prompt, tag=f'{tag}-患者-追-{fturn}')
            if not followup_patient:
                break
            history_entries.append(('患者', followup_patient.strip()))
            _notify()

            force_end = (fturn >= max_followup_turns)
            followup_doctor_prompt = _build_doctor_followup_prompt(
                history_entries, age, gender, seed=seed_text, force_labs=force_end
            )
            followup_doctor = call_gpt5(followup_doctor_prompt, tag=f'{tag}-医生-追-{fturn + 1}')
            if not followup_doctor:
                break
            followup_doctor = _cap_laboratory_request(followup_doctor)
            history_entries.append(('医生', followup_doctor.strip()))
            _notify()

            if _has_marked_section(followup_doctor, '申请查体') and not exam_done:
                _physical_exam_call(f'{tag}-查体室-追', doctor_response=followup_doctor)
                if not _has_marked_section(followup_doctor, '申请化验') and '[结束问诊]' not in followup_doctor:
                    post_exam_prompt = _build_doctor_followup_prompt(
                        history_entries, age, gender, seed=seed_text, force_labs=False
                    )
                    post_exam_response = call_gpt5(post_exam_prompt, tag=f'{tag}-医生-追-查体后')
                    if post_exam_response:
                        post_exam_response = _cap_laboratory_request(post_exam_response)
                        history_entries.append(('医生', post_exam_response.strip()))
                        _notify()
                        if _has_marked_section(post_exam_response, '申请化验'):
                            second_labs_requested = True
                            break
                        if '[结束问诊]' in post_exam_response:
                            break

            if _has_marked_section(followup_doctor, '申请化验'):
                second_labs_requested = True
                break
            if '[结束问诊]' in followup_doctor:
                break

        if second_labs_requested:
            _oracle_call(history_entries[-1][1], f'{tag}-化验室-2')
        _final_diagnosis()

    elif _has_marked_section(post_lab_response, '申请查体') and not exam_done:
        history_entries.append(('医生', post_lab_response.strip()))
        _notify()
        _physical_exam_call(f'{tag}-查体室-2', doctor_response=post_lab_response)
        if _has_marked_section(post_lab_response, '申请化验'):
            _oracle_call(post_lab_response, f'{tag}-化验室-2')
        _final_diagnosis()

    elif _has_marked_section(post_lab_response, '申请化验'):
        history_entries.append(('医生', post_lab_response.strip()))
        _notify()
        _oracle_call(post_lab_response, f'{tag}-化验室-2')
        _final_diagnosis()

    else:
        history_entries.append(('医生诊断', post_lab_response.strip()))
        _notify()

    elapsed = round(time.time() - t0, 1)
    print(f'    [{tag}] 三方交互完成，共 {len(history_entries)} 条记录，总用时 {elapsed}s')
    return _format_history_for_display(history_entries)


# ============================================================
# 模块4.1 / 4.15 / 4.2
# ============================================================

def module_4_1_doctor_treatment(age, gender: str, diagnosis: str,
                                 comorbidities_str: str, interaction_history: str,
                                 tag='模块4.1'):
    """模块4.1 医生给出个体化治疗方案（GPT调用）"""
    prompt = f"""你是一位临床医生，刚刚完成了对患者的问诊和检查，并做出了诊断。
请根据以下信息，以标准医嘱格式为该患者开立当前阶段的个体化医嘱。

患者基本信息：年龄 {age} 岁，性别 {gender}，诊断 {diagnosis}
既往疾病史：{comorbidities_str if comorbidities_str else '无特殊既往病史'}

完整问诊及化验/诊断记录：
{interaction_history}

要求：
1. 只开立该患者现阶段必须用的医嘱，严格控制数量：能口服不静滴，能单药不联合，可以不开的坚决不开
2. 结合患者的年龄、性别、伴随疾病、化验结果和临床表现进行个体化决策
3. 每条医嘱单独一行，编号列出，格式要求：
   - 药物医嘱：药品名 剂量 给药途径 频次 [疗程]
     给药途径缩写：po（口服）、ivgtt（静滴）、iv（静注）、im（肌注）、inh（吸入）
     频次缩写：st（即刻）、qd（每日1次）、bid（每日2次）、tid（每日3次）、qid（每日4次）、q8h/q12h（每8/12小时）、prn（按需）、qn（每晚）
   - 操作/处置医嘱：直接描述操作名称及关键参数
   - 护理/饮食医嘱：如适用
4. 医嘱内容简洁，不加解释、不加机制说明
5. 输出仅为医嘱列表，不加标题、不加总结说明

示例格式：
1. 阿莫西林克拉维酸钾片 875/125mg po bid × 10d
2. 布洛芬片 0.3g po prn（体温>38.5℃时）
3. 生理盐水 100ml + 氨溴索 30mg ivgtt bid
4. 流质饮食
5. 卧床休息"""

    t0 = time.time()
    print(f'    [{tag}] 正在生成治疗方案...')
    response = call_gpt5(prompt, tag=tag)
    if response:
        print(f'    [{tag}] 治疗方案生成完成，用时 {round(time.time()-t0,1)}s')
    else:
        print(f'    [{tag}] ❌ 生成失败，用时 {round(time.time()-t0,1)}s')
    return prompt, response


def module_4_15_sample_adr(treatment_plan, adr_library_text, tag='模块4.15'):
    """模块4.15 不良反应采样（本地伯努利采样，无GPT调用）"""
    t0 = time.time()
    if not treatment_plan or not adr_library_text:
        return []
    adr_list = parse_list_from_response(adr_library_text)
    if not adr_list:
        return []
    treatment_lower = treatment_plan.lower()
    sampled_adrs = []
    for item in adr_list:
        if len(item) < 5:
            continue
        drug_name, adr_name, manifestation, prob, severity = item[0], item[1], item[2], item[3], item[4]
        try:
            prob_val = float(prob)
        except (ValueError, TypeError):
            continue
        if drug_name and drug_name.lower() in treatment_lower:
            if random.random() < prob_val:
                sampled_adrs.append((drug_name, adr_name, manifestation, severity))
    elapsed = round(time.time() - t0, 3)
    print(f'    [{tag}] ADR采样完成: {len(sampled_adrs)}个ADR被触发 (用时{elapsed}s)')
    return sampled_adrs


def module_4_2_god_outcome(age, gender: str, diagnosis: str, specific_text: str,
                            comorbidities_str: str, patient_treatment_factors: str,
                            treatment_plan: str, standard_treatment: str = '',
                            patient_adr_text: str = '',
                            tag='模块4.2'):
    """模块4.2 上帝给出治疗效果评估（GPT调用）"""
    std_tx_section = ''
    if standard_treatment and standard_treatment.strip():
        std_tx_section = f"\n该疾病的标准治疗方案（参考基准）：\n{standard_treatment}\n"
    tf_section = ''
    if patient_treatment_factors and patient_treatment_factors.strip():
        tf_section = f"\n该患者存在以下影响治疗效果的因素：\n{patient_treatment_factors}\n"
    else:
        tf_section = '\n该患者无特殊的治疗影响因素。\n'
    adr_section = ''
    if patient_adr_text and patient_adr_text.strip() and patient_adr_text != '[]':
        adr_section = f"\n该患者在本次治疗期间出现了以下不良反应：\n{patient_adr_text}\n"
    adr_dimension = ''
    if adr_section:
        adr_dimension = """
【维度三：不良反应影响】
评估不良反应对治疗依从性和疗效的影响：
- 轻度ADR通常不影响治疗依从性
- 中度ADR可能导致患者减量或不规律服药，轻度削弱疗效
- 重度ADR可能导致患者自行停药，显著影响治疗效果
"""
    prompt = f"""你是全知模型（上帝视角），掌握该患者的真实病情和所有医学信息。

患者基本信息：年龄 {age} 岁，性别 {gender}，诊断 {diagnosis}
既往疾病史：{comorbidities_str if comorbidities_str else '无'}
实际病史（具体数值化）：
{specific_text}
{std_tx_section}{tf_section}{adr_section}
医生给出的治疗方案：
{treatment_plan}

请从以下维度评估治疗效果，并综合给出最终结论：

【维度一：核心治疗匹配度】
⚠️ 注意：医生医嘱遵循"最小必要原则"，有意省略非关键的支持/对症治疗，评估时不应因此降级。
评估重点仅在于：
- 一线/核心治疗是否覆盖（方向正确、关键药物或操作到位）？
- 是否存在核心治疗的明显错误？
- 对于必须联用的核心方案，是否缺少关键成员？

【维度二：影响因素作用】
结合该患者存在的治疗影响因素，评估对所用治疗的实际影响。
{adr_dimension}
【综合判断标准（核心治疗覆盖率决定基调，影响因素仅作微调）】
- 缓解（默认结论，当核心治疗正确时）
- 无缓解：核心治疗缺失关键环节；或存在重度影响因素/重度ADR导致治疗基本失效
- 恶化：核心治疗方向根本错误或选用了有害治疗

⚠️ 重要原则：不要因为影响因素或ADR"可能"影响疗效就判无缓解——临床中大多数患者即使有影响因素，接受正确核心治疗后仍能获得缓解。

严格按以下格式输出：

[治疗结果]: 缓解 / 无缓解 / 恶化（三选一）

[匹配度分析]: （逐层说明医生治疗与标准治疗的匹配情况）

[影响因素分析]: （说明治疗影响因素和不良反应对疗效的综合影响）

[综合说明]: （结合以上维度，说明最终判断理由及治疗后患者的具体预期表现）

直接输出评估内容，不加角色标签。"""

    t0 = time.time()
    print(f'    [{tag}] 正在生成治疗效果评估...')
    response = call_gpt5(prompt, tag=tag)
    if response:
        print(f'    [{tag}] 治疗效果评估完成，用时 {round(time.time()-t0,1)}s')
    else:
        print(f'    [{tag}] ❌ 生成失败，用时 {round(time.time()-t0,1)}s')
    return prompt, response


# ============================================================
# 模块5辅助函数
# ============================================================

def _parse_treatment_outcome(outcome_text):
    """从模块4.2/5.5的治疗效果文本中解析 [治疗结果] 标签的值。"""
    if not outcome_text:
        return ''
    m = re.search(r'\[治疗结果\]\s*[:：]\s*(缓解|无缓解|恶化)', outcome_text)
    return m.group(1) if m else ''



# ============================================================
# 单患者 Worker（模块3-4全流程）
# ============================================================

# 模块3/5.3 三方交互完成的标志（诊断阶段产物）
_M3_COMPLETION_MARKERS = (
    '[鉴别诊断]',
    '[医生诊断]',
    '[诊断]',
    '鉴别诊断：',
    '诊断：',
    '（诊断生成失败）',
)


def _is_interaction_complete(text: str) -> bool:
    """
    判定交互历史是否"完整结束"。
    判定规则：文本末尾 2000 字符内出现模块3/5.3 诊断阶段产物之一。
    - 如果为空字符串 → 未开始（False）
    - 如果仅有医生/患者半截对话 → 截断（False）
    - 如果包含诊断标签 → 完整（True）
    这是为了避免续跑时把"前 N 轮对话"当作已完成，导致下游模块4/5
    基于残缺对话生成错误数据。
    """
    if not text or not text.strip():
        return False
    tail = text[-2000:]
    return any(marker in tail for marker in _M3_COMPLETION_MARKERS)


def _m3_row_has_current_ledger_metadata(csv_path, patient_idx, row, staging_system) -> bool:
    ledger_context = _load_m3_ledger_context(csv_path, patient_idx, row, staging_system)
    if not _is_interaction_complete(row.get(COL_INTERACTION, '') or ''):
        return False
    return _m3_ledger_metadata_current(csv_path, patient_idx, ledger_context['metadata'])


def _generate_single_patient_m345(args):
    """线程池Worker：为单个患者运行模块3-4全流程"""
    (csv_path, patient_idx, row, row_1,
     staging_system,
     max_interaction_turns, max_followup_turns,
     update_queue) = args

    tag_prefix = f"患者{patient_idx}"
    worker_start = time.time()
    print(f"\n  --- 模拟{tag_prefix}（模块3-4）---")

    def _push_update():
        try:
            update_queue.put((patient_idx, dict(row)))
        except Exception:
            pass

    try:
        ledger_context = _load_m3_ledger_context(
            csv_path, patient_idx, row, staging_system
        )
        fact_ledger = ledger_context['ledger']
        ledger_metadata = ledger_context['metadata']

        # 判定模块3 交互历史是否"完整结束"：只有完整的三方交互
        # 才能作为下游模块4 的输入。实时保存机制会在每轮对话
        # 后落盘半截数据，若进程被异常中断，续跑时必须识别为未完成并重跑。
        has_3 = _is_interaction_complete(row.get(COL_INTERACTION, '') or '')
        has_current_ledger_metadata = _m3_ledger_metadata_current(
            csv_path, patient_idx, ledger_metadata
        )

        seed_text = row.get(COL_SEED, '')
        age_val = row.get(COL_AGE, '') or ''
        gender_val = row.get(COL_GENDER, '') or ''
        diag_val = row.get(COL_DIAGNOSIS, '') or ''
        spec_val = row.get(COL_SPECIFIC, '') or ''
        patient_lat = row.get(COL_PATIENT_LATERALITY, '') or ''
        if patient_lat:
            spec_val = spec_val + f'\n（本患者侧别：{patient_lat}）'
        como_val = row.get(COL_COMORBIDITIES, '') or ''
        age_int = int(age_val) if age_val.isdigit() else age_val
        chief_complaint_val = row.get(COL_CHIEF_COMPLAINT, '') or ''

        comorbidities_display = '、'.join([
            (item if isinstance(item, str) else item[0])
            for item in parse_list_from_response(como_val) if item
        ])

        all_specific_items = parse_list_from_response(spec_val)

        def _item_category(it):
            if not isinstance(it, (list, tuple)):
                return ''
            cats = {'症状', '体征', '实验室检查', '化验检查', '影像检查', '功能检查'}
            if len(it) >= 2 and str(it[1]).strip() in cats:
                return str(it[1]).strip()
            return ''

        signs_items = [
            item for item in all_specific_items
            if _item_category(item) == '体征'
        ]
        if signs_items:
            signs_specific_text = '\n'.join([str(item[0]) for item in signs_items])
        else:
            signs_raw = parse_list_from_response(row.get(COL_SIGNS, '') or '')
            if signs_raw:
                signs_specific_text = '\n'.join([
                    (item[0] if isinstance(item, (list, tuple)) and item else str(item))
                    for item in signs_raw
                ])
            else:
                signs_specific_text = ''

        # 从采样症状中抽取属性（持续时间/诱发因素/性质），供患者上帝模型细节回答
        # 新 5 元组: (name, stage, duration, trigger, nature)；旧 2 元组会被自动跳过
        symptom_attrs_items = parse_list_from_response(row.get(COL_SYMPTOMS, '') or '')
        symptom_attrs_lines = []
        for it in symptom_attrs_items:
            if isinstance(it, (list, tuple)) and len(it) >= 5:
                name = it[0]
                duration = str(it[2]).strip() if it[2] else ''
                trigger = str(it[3]).strip() if it[3] else ''
                nature = str(it[4]).strip() if it[4] else ''
                bits = []
                if duration:
                    bits.append(f'持续时间 {duration}')
                if trigger:
                    bits.append(f'诱发/加重 {trigger}')
                if nature:
                    bits.append(f'性质 {nature}')
                if bits:
                    symptom_attrs_lines.append(f'- {name}：' + '；'.join(bits))
        symptom_attrs_text = '\n'.join(symptom_attrs_lines)

        # ---- v4 新增：双轨 absent + 就诊状态 + 既往就诊经历 ----
        absent_symptoms = parse_list_from_response(row.get(COL_ABSENT_SYMPTOMS, '') or '[]')
        absent_signs = parse_list_from_response(row.get(COL_ABSENT_SIGNS, '') or '[]')
        absent_labs = parse_list_from_response(row.get(COL_ABSENT_LAB_TESTS, '') or '[]')
        absent_imaging = parse_list_from_response(row.get(COL_ABSENT_IMAGING, '') or '[]')
        absent_functional = parse_list_from_response(row.get(COL_ABSENT_FUNCTIONAL, '') or '[]')
        prior_visited = row.get(COL_PRIOR_VISITED, '') or ''
        try:
            prior_visit_count = int(row.get(COL_PRIOR_VISIT_COUNT, 0) or 0)
        except (ValueError, TypeError):
            prior_visit_count = 0
        prior_visit_history = row.get(COL_PRIOR_VISIT_HISTORY, '') or ''
        duration_total = row.get(COL_DURATION_TOTAL, '') or ''
        if fact_ledger is not None:
            symptom_attrs_text = ''
            prior_visited = ''
            prior_visit_count = 0
            prior_visit_history = ''

        # ---- 模块3 ----
        if has_3 and has_current_ledger_metadata:
            print(f"    [{tag_prefix}-3] ⏭️  已有当前账本交互历史数据，跳过")
        else:
            if has_3 and not has_current_ledger_metadata:
                print(f"    [{tag_prefix}-3] 旧交互历史缺少当前账本 metadata，重新生成")
            def _on_interaction_update(hist_str):
                row[COL_INTERACTION] = hist_str
                _push_update()
            interaction_result = _retry_module_call(
                module_3_interaction,
                kwargs={
                    'seed_text': seed_text, 'age': age_int, 'gender': gender_val,
                    'diagnosis': diag_val, 'specific_text': spec_val,
                    'comorbidities_text': como_val,
                    'signs_specific_text': signs_specific_text,
                    'chief_complaint': chief_complaint_val,
                    'symptom_attrs_text': symptom_attrs_text,
                    'absent_symptoms': absent_symptoms,
                    'absent_signs': absent_signs,
                    'absent_labs': absent_labs,
                    'absent_imaging': absent_imaging,
                    'absent_functional': absent_functional,
                    'prior_visited': prior_visited,
                    'prior_visit_count': prior_visit_count,
                    'prior_visit_history': prior_visit_history,
                    'duration_total': duration_total,
                    'max_turns': max_interaction_turns,
                    'max_followup_turns': max_followup_turns,
                    'on_update': _on_interaction_update,
                    'tag': f"{tag_prefix}-3",
                    'fact_ledger': fact_ledger,
                    'ledger_metadata': ledger_metadata,
                },
                module_name=f"{tag_prefix}-3"
            )
            row[COL_INTERACTION] = interaction_result
            if interaction_result:
                _write_m3_ledger_metadata(csv_path, patient_idx, ledger_metadata)
            _push_update()

        # ---- 模块4.1 / 4.15 / 4.2 已去掉，不比较治疗效果 ----

        worker_elapsed = round(time.time() - worker_start, 1)
        print(f"    [{tag_prefix}] ✅ 模块3完成，总用时 {worker_elapsed}s")
        _push_update()

    except Exception as e:
        worker_elapsed = round(time.time() - worker_start, 1)
        print(f"    [{tag_prefix}] ❌ 模块3-4失败（{worker_elapsed}s）: {e}")
        row[_M345_ERROR_ROW_KEY] = f'{type(e).__name__}: {e}'
        traceback.print_exc()
        _push_update()

    return row




def _initial_patient_latest(existing_patients):
    return {
        patient_idx: dict(row)
        for patient_idx, row in sorted((existing_patients or {}).items())
    }


def _rows_for_expected_patient_indices(row_1, patient_latest, expected_indices):
    indices = sorted(set(expected_indices or []) | set(patient_latest.keys()))
    rows = [row_1]
    for patient_idx in indices:
        row = patient_latest.get(patient_idx)
        rows.append(dict(row) if isinstance(row, dict) else _empty_row())
    return rows


# ============================================================
# 单CSV文件处理函数
# ============================================================

def process_csv_module345(csv_path, max_interaction_turns=None, max_followup_turns=None,
                           num_workers=5):
    """
    对单个 CSV 文件运行模块3-4：
    - 读取 row_1（概率库）和患者行（需有 COL_SPECIFIC）
    - 为每个患者运行模块3-4
    - 结果写回同一 CSV
    """
    if max_interaction_turns is None:
        max_interaction_turns = DEFAULT_MAX_INTERACTION_TURNS
    if max_followup_turns is None:
        max_followup_turns = DEFAULT_MAX_FOLLOWUP_TURNS

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    print(f"\n{'='*60}")
    print(f"[模块3-4] 处理: {csv_path}")
    print(f"{'='*60}")

    try:
        row_1, existing_patients = _load_existing_csv(csv_path)

        if row_1 is None:
            print(f"  ⚠️ CSV不存在或为空，跳过: {csv_path}")
            return {'status': 'skip', 'file': csv_path}

        if not row_1.get(COL_STAGING_SYSTEM, '').strip():
            print(f"  ⚠️ row_1 缺少分级系统数据（模块1未完成），跳过")
            return {'status': 'skip', 'file': csv_path}

        if not existing_patients:
            print(f"  ⚠️ 没有患者行（模块2未运行），跳过: {csv_path}")
            return {'status': 'skip', 'file': csv_path}

        staging_system = parse_staging_system(row_1.get(COL_STAGING_SYSTEM, ''))
        if not staging_system:
            staging_system = {
                'name': '默认分级',
                'levels': [
                    {'level': 1, 'name': '轻度', 'description': '', 'proportion': 0.6},
                    {'level': 2, 'name': '重度', 'description': '', 'proportion': 0.4},
                ]
            }

        tasks = []
        skip_count = 0
        expected_patient_indices = sorted(existing_patients.keys())
        patient_latest = _initial_patient_latest(existing_patients)
        update_queue = _queue_module.Queue()

        for patient_i, existing_row in sorted(existing_patients.items()):
            if not existing_row.get(COL_SPECIFIC, '').strip():
                print(f"  ⚠️ 患者{patient_i} 缺少 COL_SPECIFIC（模块2未完成），跳过")
                patient_latest[patient_i] = existing_row
                skip_count += 1
                continue

            try:
                has_current_ledger_metadata = _m3_row_has_current_ledger_metadata(
                    csv_path, patient_i, existing_row, staging_system
                )
            except Exception as exc:
                print(f"  ❌ 患者{patient_i} M3 fact ledger invalid: {exc}")
                return {'status': 'error', 'file': csv_path, 'error': str(exc)}

            if has_current_ledger_metadata:
                patient_latest[patient_i] = existing_row
                skip_count += 1
                print(f"  ⏭️  患者{patient_i} 模块3已完成且账本 metadata 当前，跳过")
                continue
            if _is_interaction_complete(existing_row.get(COL_INTERACTION, '') or ''):
                print(f"  🔁 患者{patient_i} 旧模块3缺少当前账本 metadata，重新生成")

            tasks.append((
                csv_path, patient_i, dict(existing_row), row_1,
                staging_system,
                max_interaction_turns, max_followup_turns,
                update_queue
            ))

        if skip_count > 0:
            print(f"  ⏭️  已跳过 {skip_count} 个已完成的患者")

        if not tasks:
            print(f"  ⏭️  所有患者的模块3-4已完成")
        else:
            print(f"  🔄 需要运行模块3-4 的患者: {len(tasks)} 个")

            with ThreadPoolExecutor(max_workers=min(len(tasks), num_workers)) as executor:
                futures = {
                    executor.submit(_generate_single_patient_m345, task): task[1]
                    for task in tasks
                }
                all_done = False
                while not all_done:
                    all_done = all(f.done() for f in futures)
                    drained = False
                    while not drained:
                        try:
                            patient_idx, row_data = update_queue.get(
                                timeout=2 if not all_done else 0.1
                            )
                            patient_latest[patient_idx] = row_data
                            _save_csv(
                                csv_path,
                                _rows_for_expected_patient_indices(
                                    row_1, patient_latest, expected_patient_indices
                                ),
                            )
                        except _queue_module.Empty:
                            drained = True


                worker_errors = []
                for future, patient_idx in futures.items():
                    try:
                        row_result = future.result()
                    except Exception as exc:
                        worker_errors.append(f'patient {patient_idx}: {type(exc).__name__}: {exc}')
                        continue
                    if isinstance(row_result, dict) and row_result.get(_M345_ERROR_ROW_KEY):
                        worker_errors.append(f'patient {patient_idx}: {row_result[_M345_ERROR_ROW_KEY]}')
                if worker_errors:
                    return {'status': 'error', 'file': csv_path, 'error': '; '.join(worker_errors)}

        while True:
            try:
                patient_idx, row_data = update_queue.get_nowait()
                patient_latest[patient_idx] = row_data
            except _queue_module.Empty:
                break

        _save_csv(
            csv_path,
            _rows_for_expected_patient_indices(
                row_1, patient_latest, expected_patient_indices
            ),
        )
        print(f"  ✅ 模块3-4完成，保存至: {csv_path}")
        print(f"{'='*60}")

        return {'status': 'success', 'file': csv_path}

    except Exception as e:
        print(f"  ❌ 处理CSV [{csv_path}] 时出错: {e}")
        traceback.print_exc()
        return {'status': 'error', 'file': csv_path, 'error': str(e)}


# ============================================================
# 批量处理多个 CSV
# ============================================================

class _CsvScopedQueue:
    """
    将 worker 的 (patient_idx, row_data) 转发到全局队列，
    并附加所属 csv_path，以实现跨 CSV 统一 drain。
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


def _prepare_csv_m345(csv_path, max_interaction_turns, max_followup_turns, global_queue):
    """CSV 级预处理：读取 → 校验 → 构造 tasks。返回 (state, tasks)。"""
    print(f"\n[模块3-4 预处理] {csv_path}")
    try:
        row_1, existing_patients = _load_existing_csv(csv_path)
        if row_1 is None:
            print(f"  ⚠️ CSV不存在或为空，跳过: {csv_path}")
            return None, []
        if not row_1.get(COL_STAGING_SYSTEM, '').strip():
            print(f"  ⚠️ row_1 缺少分级系统数据（模块1未完成），跳过: {csv_path}")
            return None, []
        if not existing_patients:
            print(f"  ⚠️ 没有患者行（模块2未运行），跳过: {csv_path}")
            return None, []

        staging_system = parse_staging_system(row_1.get(COL_STAGING_SYSTEM, ''))
        if not staging_system:
            staging_system = {
                'name': '默认分级',
                'levels': [
                    {'level': 1, 'name': '轻度', 'description': '', 'proportion': 0.6},
                    {'level': 2, 'name': '重度', 'description': '', 'proportion': 0.4},
                ]
            }

        scoped_queue = _CsvScopedQueue(csv_path, global_queue)

        tasks = []
        skip_count = 0
        expected_patient_indices = sorted(existing_patients.keys())
        patient_latest = _initial_patient_latest(existing_patients)
        for patient_i, existing_row in sorted(existing_patients.items()):
            if not existing_row.get(COL_SPECIFIC, '').strip():
                print(f"  ⚠️ [{os.path.basename(csv_path)}] 患者{patient_i} 缺少 COL_SPECIFIC（模块2未完成），跳过")
                patient_latest[patient_i] = existing_row
                skip_count += 1
                continue

            try:
                has_current_ledger_metadata = _m3_row_has_current_ledger_metadata(
                    csv_path, patient_i, existing_row, staging_system
                )
            except Exception as exc:
                print(f"  ❌ [{os.path.basename(csv_path)}] 患者{patient_i} M3 fact ledger invalid: {exc}")
                return {'csv_path': csv_path, 'error': str(exc)}, []

            if has_current_ledger_metadata:
                patient_latest[patient_i] = existing_row
                skip_count += 1
                continue
            if _is_interaction_complete(existing_row.get(COL_INTERACTION, '') or ''):
                print(f"  🔁 [{os.path.basename(csv_path)}] 患者{patient_i} 旧模块3缺少当前账本 metadata，重新生成")

            tasks.append((
                csv_path, patient_i, dict(existing_row), row_1,
                staging_system,
                max_interaction_turns, max_followup_turns,
                scoped_queue
            ))

        if skip_count > 0:
            print(f"  ⏭️  [{os.path.basename(csv_path)}] 已跳过 {skip_count} 个已完成/未就绪患者")
        if tasks:
            print(f"  🔄 [{os.path.basename(csv_path)}] 待运行模块3-4 患者: {len(tasks)} 个")
        else:
            print(f"  ✅ [{os.path.basename(csv_path)}] 所有患者的模块3-4已完成")

        state = {
            'csv_path': csv_path,
            'row_1': row_1,
            'patient_latest': patient_latest,
            'expected_patient_indices': expected_patient_indices,
            'lock': threading.Lock(),
        }
        return state, tasks
    except Exception as e:
        print(f"  ❌ 预处理出错 [{csv_path}]: {e}")
        traceback.print_exc()
        return None, []


def _apply_update_m345(state, patient_idx, row_data):
    with state['lock']:
        state['patient_latest'][patient_idx] = row_data
        rows = _rows_for_expected_patient_indices(
            state['row_1'],
            state['patient_latest'],
            state.get('expected_patient_indices', []),
        )
        _save_csv(state['csv_path'], rows)


def _finalize_csv_m345(state):
    with state['lock']:
        rows = _rows_for_expected_patient_indices(
            state['row_1'],
            state['patient_latest'],
            state.get('expected_patient_indices', []),
        )
        _save_csv(state['csv_path'], rows)
    print(f"  ✅ [模块3-4] 最终保存: {state['csv_path']}")


def run_module345(csv_dir=None, csv_file=None,
                  max_interaction_turns=None, max_followup_turns=None,
                  num_workers=50):
    """
    跨 CSV 全局并发：num_workers 个线程同时服务来自不同 CSV 的患者任务。
    """
    if max_interaction_turns is None:
        max_interaction_turns = DEFAULT_MAX_INTERACTION_TURNS
    if max_followup_turns is None:
        max_followup_turns = DEFAULT_MAX_FOLLOWUP_TURNS

    if csv_file:
        csv_files = [csv_file]
    elif csv_dir:
        csv_files = sorted(glob(os.path.join(csv_dir, '**', '*.csv'), recursive=True))
        csv_files = [f for f in csv_files if '处理汇总' not in os.path.basename(f)
                     and '循证调用记录' not in f]
    else:
        print("错误：请指定 --csv_dir 或 --csv_file")
        return

    if not csv_files:
        print(f"未找到CSV文件: {csv_dir or csv_file}")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，开始模块3-4处理（全局并发 {num_workers}）...")
    start_time = time.time()

    global_queue = _queue_module.Queue()
    csv_states = {}
    all_tasks = []
    prep_skipped = 0
    prep_errors = []

    for csv_path in csv_files:
        state, tasks = _prepare_csv_m345(
            csv_path, max_interaction_turns, max_followup_turns, global_queue
        )
        if state is None:
            prep_skipped += 1
            continue
        if state.get('error'):
            prep_skipped += 1
            prep_errors.append(f"{csv_path}: {state['error']}")
            continue
        csv_states[csv_path] = state
        for t in tasks:
            all_tasks.append((csv_path, t))

    if not all_tasks:
        print("\n所有 CSV 的模块3-4均已完成，无需运行。")
    else:
        total = len(all_tasks)
        print(f"\n{'='*60}")
        print(f"共 {total} 个患者任务将通过全局并发池（workers={num_workers}）处理")
        print(f"{'='*60}")

        effective_workers = min(total, num_workers)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_generate_single_patient_m345, task): (csv_path, task[1])
                for (csv_path, task) in all_tasks
            }
            all_done = False
            while not all_done:
                all_done = all(f.done() for f in futures)
                drained = False
                while not drained:
                    try:
                        csv_path, patient_idx, row_data = global_queue.get(
                            timeout=2 if not all_done else 0.1
                        )
                        state = csv_states.get(csv_path)
                        if state is not None:
                            _apply_update_m345(state, patient_idx, row_data)
                    except _queue_module.Empty:
                        drained = True

        worker_errors = []
        for future, (csv_path, patient_idx) in futures.items():
            try:
                row_result = future.result()
            except Exception as exc:
                worker_errors.append(f'{csv_path} patient {patient_idx}: {type(exc).__name__}: {exc}')
                continue
            if isinstance(row_result, dict) and row_result.get(_M345_ERROR_ROW_KEY):
                worker_errors.append(f'{csv_path} patient {patient_idx}: {row_result[_M345_ERROR_ROW_KEY]}')
        if worker_errors:
            raise RuntimeError('; '.join(worker_errors))

        while True:
            try:
                csv_path, patient_idx, row_data = global_queue.get_nowait()
                state = csv_states.get(csv_path)
                if state is not None:
                    _apply_update_m345(state, patient_idx, row_data)
            except _queue_module.Empty:
                break

    if prep_errors:
        raise RuntimeError('; '.join(prep_errors))

    for state in csv_states.values():
        _finalize_csv_m345(state)

    total_time = round(time.time() - start_time, 2)
    print(f"\n{'='*60}")
    print(f"模块3-4全部处理完成")
    print(f"处理 CSV: {len(csv_states)}  预处理跳过: {prep_skipped}")
    print(f"总耗时: {total_time} 秒")
    print(f"{'='*60}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='virtual_clinical_interaction.py：模块3-4 - 虚拟临床交互与治疗效果评估'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv_dir', type=str,
                       help='扫描目录中所有由 atlas_based_patient_generation.py 输出的 CSV')
    group.add_argument('--csv_file', type=str,
                       help='处理单个 CSV 文件')
    parser.add_argument('--max_interaction_turns', type=int, default=DEFAULT_MAX_INTERACTION_TURNS,
                        help=f'每轮问诊最大轮数（默认: {DEFAULT_MAX_INTERACTION_TURNS}）')
    parser.add_argument('--max_followup_turns', type=int, default=DEFAULT_MAX_FOLLOWUP_TURNS,
                        help=f'追加问诊最大轮数（默认: {DEFAULT_MAX_FOLLOWUP_TURNS}）')
    parser.add_argument('--num_workers', type=int, default=50,
                        help='跨 CSV 全局并发线程数，同时服务不同 CSV 的不同患者（默认: 50）')
    args = parser.parse_args()

    run_module345(
        csv_dir=args.csv_dir,
        csv_file=args.csv_file,
        max_interaction_turns=args.max_interaction_turns,
        max_followup_turns=args.max_followup_turns,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    main()
