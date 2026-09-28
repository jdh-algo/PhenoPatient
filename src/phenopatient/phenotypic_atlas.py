# -*- coding: utf-8 -*-
"""
phenotypic_atlas.py
模块1：表型概率库生成（表型图谱构建）

功能：读取患者种子文件，为每个种子生成问诊、查体和检验检查模拟所需的
五类表型概率库、伴随疾病库及疾病分级系统 CSV（row_1）。
支持循证检索增强（--enable_evidence 开关）。

输出：{output_dir}/{seed_name}.csv（每个种子一个文件，仅写入 row_1）

下游：atlas_based_patient_generation.py 读取此 CSV 生成患者行
"""

import sys
import os
import ast
import json
import math
import re
import time
import multiprocessing as mp
import argparse
import traceback

# 从同目录 utils 导入所有公共工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DEFAULT_SEED_FILE, DEFAULT_OUTPUT_BASE, DEFAULT_NUM_SEED_WORKERS,
    DEFAULT_EVIDENCE_ENV, DEFAULT_EVIDENCE_WORKERS,
    DEFAULT_EVIDENCE_TIMEOUT, DEFAULT_EVIDENCE_MAX_RETRIES,
    DEFAULT_MODULE_MAX_RETRIES, DEFAULT_MODULE_RETRY_SLEEP,
    EvidenceLogger, _get_evidence_api,
    call_evidence_api, batch_evidence_queries,
    call_gpt54 as call_gpt5, _retry_module_call,
    parse_list_from_response, parse_staging_system,
    _build_staging_levels_text, _get_num_levels, _get_level_names,
    COL_SEED, COL_STAGING_SYSTEM,
    COL_DIFFERENTIAL_DIAGNOSIS,
    COL_LATERALITY_TYPE,
    COL_SYMPTOMS, COL_SIGNS,
    COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS,
    COL_KEY_LAB_TESTS, COL_DIFF_KEY_LAB_TESTS,
    COL_KEY_SYMPTOMS, COL_DIFF_KEY_SYMPTOMS,
    COL_KEY_SIGNS, COL_DIFF_KEY_SIGNS,
    COL_DIAGNOSIS_COMPONENTS,
    COL_STANDARD_TREATMENT, COL_ADR_LIBRARY,
    COL_COMORBIDITIES, COL_COMPLICATION_PHENOTYPES, COL_TREATMENT_FACTORS,
    COL_M10_INPUT, COL_M10_OUTPUT,
    COL_M101_INPUT, COL_M101_OUTPUT,
    COL_M102_INPUT, COL_M102_OUTPUT,
    COL_M111_INPUT, COL_M111_OUTPUT,
    COL_M112_INPUT, COL_M112_OUTPUT,
    COL_M121_INPUT, COL_M121_OUTPUT,
    COL_M122_INPUT, COL_M122_OUTPUT,
    COL_M123_INPUT, COL_M123_OUTPUT,
    COL_M211_INPUT, COL_M211_OUTPUT,
    COL_M221_INPUT, COL_M221_OUTPUT,
    COL_M231_INPUT, COL_M231_OUTPUT,
    COL_M124_INPUT, COL_M124_OUTPUT,
    COL_M125_INPUT, COL_M125_OUTPUT,
    COL_M126_INPUT, COL_M126_OUTPUT,
    COL_M131_INPUT, COL_M131_OUTPUT,
    COL_M132_INPUT, COL_M132_OUTPUT,
    COL_M15_INPUT, COL_M15_OUTPUT,
    COL_M151_INPUT, COL_M151_OUTPUT,
    COL_M16_INPUT, COL_M16_OUTPUT,
    _empty_row, _load_existing_csv, _save_csv,
)


# ============================================================
# 内部辅助函数
# ============================================================

# A2：轻/重分档概率硬约束（适用于所有表型类型的概率生成阶段）
_A2_SEVERITY_RULES = """
【A2 轻症/重症 概率分档硬约束（必须严格遵守）】
在为各分级（如轻度/中度/重度、早期/进展期/终末期）打概率时必须遵守以下规则：

1. 属于【重度/危重表现】的条目（例如：发绀、意识障碍、休克体征、明显三凹征、SpO2<90%、严重酸中毒、
   脓毒症相关心肌损伤、DIC、急性肾损伤、昏迷、抽搐、循环衰竭、ICU 级别监护参数，以及
   MRI、PET-CT、CTA/CTPA、支气管镜、纤支镜活检等侵入性/高级影像/有创检查）：
   - 在【最轻档（例如"轻度"/"早期"）】概率**必须严格为 0**（不是 0.01，不是 0.05，必须是精确的 0）；
   - 在【中间档】概率只能为 0 或极低值（≤ 0.05）；
   - 在【最重档（例如"重度"/"危重"）】概率应显著高于中间档，至少为 0.3 及以上；
   - 绝不允许出现"最轻档概率 > 0 的重度表现"。

   ⚠️ 严格执行：若你在最轻档给出任何 > 0 的概率给上述重度/危重表现，该输出视为不合格。
   错误示例（禁止）：('发绀', 0.05, 0.20, 0.60)  ← 最轻档0.05不为0，不允许
   正确示例：        ('发绀', 0.0,  0.0,  0.65)   ← 最轻档和中间档均为0，正确

2. 属于【轻度典型/常见表现】的条目（例如：低热、轻度咳嗽、局灶干啰音、白细胞轻度升高、CRP 轻度升高）：
   - 在【最轻档】概率应 ≥ 常见水平（如 0.4 及以上）；
   - 在【最重档】概率可以持平或略低（因重症患者往往表现更重而非更轻）；
   - 不要把常见轻症表现的最轻档概率压得过低。

3. 定性方向需与病情相符：同一表型的"升高/降低"方向在不同档之间不应反转；数量级应随病情加重递增。

总原则：如有疑问，宁可把最轻档概率设为 0，也不要给出任何非零值给重度/危重表现。
"""

# C1：核心/常规/少见 重要性分层（针对化验/影像/功能检查）
_C1_IMPORTANCE_RULES = """
【C1 重要性分层 概率硬约束（仅适用于检查类：实验室检查/影像检查/功能检查）】
请在生成概率时先判断每个条目的"重要性分层"，并按以下区间给概率：

1. 核心（金标准/确诊性/对本诊断决定性的检查，例如病原培养/PCR、特异性抗原抗体、
   关键影像征象、金标准功能检查）：
   - 最轻档概率 ≥ 0.60；
   - 中间档概率 ≥ 0.75；
   - 仅最重档概率 = 1.0；较轻分期仍按实际敏感度和检查时机设定，不得一律设为 1.0。
2. 常规（对该诊断有支持性意义的常规检查，如 CRP/PCT/血常规等通用炎症指标）：
   - 概率按实际临床发生率设置，最轻档通常 0.3–0.5，最重档 0.6–0.85。
3. 少见（偶尔出现异常的检查）：
   - 概率按临床发生率设置，不必刻意抬高；同时必须遵守 A2 规则（若为重度表现请在轻档置 0）。

所有被标注为该诊断"重要化验检查"/"鉴别诊断重要化验检查"的条目，自动视为【核心】档。
"""


def _severity_rules_for(phenotype_type: str) -> str:
    """返回针对指定表型类型应附加在 prompt 中的 A2/C1 硬约束文本。"""
    lines = [_A2_SEVERITY_RULES]
    if phenotype_type in ('实验室检查', '影像检查', '功能检查'):
        lines.append(_C1_IMPORTANCE_RULES)
    return '\n'.join(lines)


def _extract_diagnosis_from_seed(seed_text):
    """从seed_text中提取诊断名称（去掉ICD编码和括号修饰）用于循证查询。"""
    # V5 种子一行就是"疾病名  # ICD编码"，直接切 # 取前半段即可，
    # 不用走 _parse_seed（那个会触发 demographics 查询）。
    diagnosis = str(seed_text).split('#', 1)[0].strip()
    diag = re.sub(r'\[.*?\]', '', diagnosis).strip()
    diag = re.sub(r'[（(][^（()）]*?ICD[^（()）]*?[）)]', '', diag).strip()
    return diag


# ============================================================
# 多疾病诊断拆分（v4 新增 - 主题2.5）
# ============================================================

# 启发式拆分表：键为完整诊断名（或子串），值为拆分后的子疾病列表。
# 命中时取最长匹配的 key 对应的列表返回。
_COMPOUND_DIAGNOSIS_HEURISTIC = {
    "糖尿病肾病": ["糖尿病", "糖尿病肾病"],
    "糖尿病视网膜病变": ["糖尿病", "糖尿病视网膜病变"],
    "糖尿病周围神经病变": ["糖尿病", "糖尿病周围神经病变"],
    "糖尿病足": ["糖尿病", "糖尿病足"],
    "糖尿病酮症酸中毒": ["糖尿病", "糖尿病酮症酸中毒"],
    "高血压性心脏病": ["高血压", "高血压性心脏病"],
    "高血压心脏病": ["高血压", "高血压性心脏病"],
    "高血压肾病": ["高血压", "高血压肾病"],
    "高血压性肾病": ["高血压", "高血压肾病"],
    "肺源性心脏病": ["慢性阻塞性肺疾病", "肺源性心脏病"],
    "慢性肺源性心脏病": ["慢性阻塞性肺疾病", "慢性肺源性心脏病"],
    "风湿性心脏病": ["风湿热", "风湿性心脏病"],
    "冠状动脉粥样硬化性心脏病": ["动脉粥样硬化", "冠状动脉粥样硬化性心脏病"],
    "缺血性心肌病": ["冠心病", "缺血性心肌病"],
    "酒精性肝硬化": ["酒精性肝病", "酒精性肝硬化"],
    "乙型肝炎肝硬化": ["慢性乙型肝炎", "乙型肝炎肝硬化"],
    "丙型肝炎肝硬化": ["慢性丙型肝炎", "丙型肝炎肝硬化"],
    "肝硬化合并肝性脑病": ["肝硬化", "肝性脑病"],
}


def _split_compound_diagnosis(diagnosis: str) -> list:
    """启发式：把复合诊断拆为子疾病名列表。
    单一疾病时返回 [diagnosis]。仅基于命名启发式，不做 LLM 兜底。"""
    if not diagnosis:
        return [diagnosis]
    diag = str(diagnosis).strip()
    if not diag:
        return [diag]
    # 优先做精确匹配
    if diag in _COMPOUND_DIAGNOSIS_HEURISTIC:
        return list(_COMPOUND_DIAGNOSIS_HEURISTIC[diag])
    # 子串匹配（最长优先）
    matched_keys = [k for k in _COMPOUND_DIAGNOSIS_HEURISTIC.keys() if k in diag]
    if matched_keys:
        best = max(matched_keys, key=len)
        return list(_COMPOUND_DIAGNOSIS_HEURISTIC[best])
    return [diag]


def _replace_diagnosis_in_seed(seed_text: str, new_diagnosis: str) -> str:
    """构造一份临时 seed_text：把诊断字段替换为 new_diagnosis（保留 # 后的 ICD 部分）。"""
    if not seed_text:
        return new_diagnosis
    s = str(seed_text)
    if '#' in s:
        head, tail = s.split('#', 1)
        return f"{new_diagnosis}  #{tail}"
    return new_diagnosis


def _merge_probability_tuples(parsed_lists: list) -> list:
    """合并多个 (name, p1, p2, ...) 元组列表：同名项目保留概率最大者（按各档逐位取 max）。"""
    merged = {}
    for lst in parsed_lists:
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            name = str(item[0]).strip()
            if not name:
                continue
            probs = list(item[1:])
            if name not in merged:
                merged[name] = probs
            else:
                old = merged[name]
                new_probs = []
                for i in range(max(len(old), len(probs))):
                    a = old[i] if i < len(old) else 0
                    b = probs[i] if i < len(probs) else 0
                    try:
                        new_probs.append(max(float(a), float(b)))
                    except Exception:
                        new_probs.append(a if a else b)
                merged[name] = new_probs
    return [tuple([name] + probs) for name, probs in merged.items()]


def _strip_code_fence(text):
    s = str(text or '').strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    m = re.fullmatch(r'```(?:python|json)?\s*\n?(.*?)\n?```', s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s


def _parse_strict_list_literal(response_text, module_name='模块1'):
    """严格解析完整列表字面量；拒绝列表前后包装文本。"""
    s = _strip_code_fence(response_text)
    if not s.startswith('[') or not s.endswith(']'):
        raise ValueError(f'{module_name} 输出必须是纯 Python/JSON 列表')
    for loader in (ast.literal_eval, json.loads):
        try:
            parsed = loader(s)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    raise ValueError(f'{module_name} 输出不是可解析列表')


def _coerce_probability(value, module_name):
    try:
        prob = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{module_name} 概率不是数字: {value!r}')
    if not 0 <= prob <= 1:
        raise ValueError(f'{module_name} 概率超出[0,1]: {prob!r}')
    return prob


_FLAT_COMPOSITE_CONNECTOR_RE = re.compile(
    r'(?:且|同时|并伴|合并|以及|、|或|'
    r'(?<!饱)(?<!中)(?<!亲)和(?!度|力|抗体)|'
    r'(?<!触)(?<!闻)(?<!累)及(?!以上|以下)|并)'
)
_FLAT_CJK_SLASH_RE = re.compile(
    r'(?<=[\u4e00-\u9fff])\s*[/／]\s*(?=[\u4e00-\u9fff])'
)
_FLAT_RATIO_TERM_RE = re.compile(r'(?:比值|比率|比例|指数)')
_FLAT_CJK_UNIT_SLASH_RE = re.compile(
    r'(?:次|毫升|升|[千毫微纳]?[克米瓦]|单位|个)\s*[/／]\s*'
    r'(?:分(?:钟)?|秒|小时|天|升|分升|平方米|立方米|视野|高倍视野)'
)
_FLAT_ABNORMAL_TERM_RE = re.compile(
    r'(?:升高|增高|降低|下降|减少|缩小|增大|扩张|狭窄|延长|缩短|阳性|阴性|异常|受限|减退|增强|减弱|[<>＜＞≤≥])'
)
_FUNCTIONAL_IMAGING_TERM_RE = re.compile(
    r'(?:超声|心动图|CT|MRI|PET|X线|磁共振|血管成像|造影|影像)'
)


def _looks_like_composite_measure_name(name):
    """拒绝明显把两个可量化指标/异常发现合成一个表型名的平面库条目。"""
    if _FLAT_CJK_SLASH_RE.search(name) and \
            not _FLAT_RATIO_TERM_RE.search(name) and \
            not _FLAT_CJK_UNIT_SLASH_RE.search(name):
        return True
    if not isinstance(name, str) or not _FLAT_COMPOSITE_CONNECTOR_RE.search(name):
        return False
    parts = [p.strip() for p in _FLAT_COMPOSITE_CONNECTOR_RE.split(name) if p.strip()]
    if len(parts) < 2:
        return False
    abnormal_parts = [p for p in parts if _FLAT_ABNORMAL_TERM_RE.search(p)]
    if len(abnormal_parts) >= 2:
        return True
    # 拒绝“A及B升高”这类主名称共享异常后缀的复合指标，
    # 但允许括号内的解剖部位/观察方式备选描述。
    head = re.split(r'[（(]', name, maxsplit=1)[0]
    if not _FLAT_COMPOSITE_CONNECTOR_RE.search(head):
        return False
    head_parts = [
        part.strip() for part in _FLAT_COMPOSITE_CONNECTOR_RE.split(head)
        if part.strip()
    ]
    return len(head_parts) >= 2 and any(
        _FLAT_ABNORMAL_TERM_RE.search(part) for part in head_parts
    )


def validate_m1_flat_probability_schema(response_text, staging_system,
                                        module_name='模块1', expected_names=None):
    n_levels = _get_num_levels(staging_system)
    parsed = _parse_strict_list_literal(response_text, module_name=module_name)
    canonical = []
    seen = set()
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f'{module_name} 第{idx}项不是元组/列表')
        if len(item) != n_levels + 1:
            raise ValueError(f'{module_name} 第{idx}项长度应为{n_levels + 1}，实际为{len(item)}')
        name = str(item[0]).strip()
        if not name:
            raise ValueError(f'{module_name} 第{idx}项名称为空')
        if module_name == '模块1.23' and _FUNCTIONAL_IMAGING_TERM_RE.search(name):
            raise ValueError(f'{module_name} 功能检查混入影像学项目: {name}')
        if _looks_like_composite_measure_name(name):
            raise ValueError(f'{module_name} 第{idx}项不是单一表型/单一可量化指标: {name}')
        if name in seen:
            raise ValueError(f'{module_name} 出现重复条目: {name}')
        seen.add(name)
        probs = tuple(_coerce_probability(v, module_name) for v in item[1:])
        canonical.append((name,) + probs)
    if expected_names is not None:
        expected = [str(x).strip() for x in expected_names if str(x).strip()]
        if set(seen) != set(expected):
            raise ValueError(f'{module_name} 名称不守恒: expected={expected!r}, actual={sorted(seen)!r}')
    return repr(canonical)


def _split_expected_symptom_names(names):
    result = []
    seen = set()
    for name in names or []:
        for sub in _split_compound_symptom_name(str(name)):
            sub = str(sub).strip()
            if sub and sub not in seen:
                result.append(sub)
                seen.add(sub)
    return result


def validate_m1_symptom_schema(response_text, staging_system, expected_names=None):
    n_levels = _get_num_levels(staging_system)
    parsed = _parse_strict_list_literal(response_text, module_name='模块1.11')
    canonical = []
    seen = set()
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, (list, tuple)):
            raise ValueError(f'模块1.11 第{idx}项不是元组/列表')
        if len(item) != n_levels + 1:
            raise ValueError(f'模块1.11 第{idx}项长度应为{n_levels + 1}，实际为{len(item)}')
        name = str(item[0]).strip()
        if not name:
            raise ValueError(f'模块1.11 第{idx}项症状名为空')
        if name in seen:
            raise ValueError(f'模块1.11 出现重复症状: {name}')
        seen.add(name)
        staged = []
        for level_idx, sub in enumerate(item[1:], start=1):
            if not isinstance(sub, (list, tuple)) or len(sub) != 4:
                raise ValueError(f'模块1.11 第{idx}项第{level_idx}级不是4元属性子元组')
            duration = str(sub[0]).strip()
            trigger = str(sub[1]).strip()
            nature = str(sub[2]).strip()
            prob = _coerce_probability(sub[3], '模块1.11')
            staged.append((duration, trigger, nature, prob))
        canonical.append((name, *staged))
    if expected_names is not None:
        expected = _split_expected_symptom_names(expected_names)
        if set(seen) != set(expected):
            raise ValueError(f'模块1.11 症状名称不守恒: expected={expected!r}, actual={sorted(seen)!r}')
    return repr(canonical)


def _validate_or_none(response_text, validator, module_name):
    if response_text is None:
        return None
    try:
        return validator(response_text)
    except ValueError as exc:
        print(f"  [{module_name}] ⚠️ schema校验失败: {exc}")
        return None


def _run_with_subdiagnoses(seed_text, sub_diagnoses, single_fn, tag=''):
    """对每个子诊断分别调用 single_fn(sub_seed_text)，返回 (合并后的 prompt, 合并后的 response_text)。
    single_fn 接收 sub_seed_text，返回 (prompt, response_text)。
    response_text 应是 Python 元组列表字符串。合并时按表型项目名去重、概率取 max。"""
    all_prompts = []
    all_parsed = []
    raw_responses = []
    for sub in sub_diagnoses:
        sub_seed = _replace_diagnosis_in_seed(seed_text, sub)
        print(f"  [{tag}] 子诊断 [{sub}] 单独建库...")
        sub_prompt, sub_response = _retry_module_call(
            single_fn,
            args=(sub_seed,),
            module_name=f'{tag}-{sub}',
        )
        all_prompts.append(f"\n===== 子诊断: {sub} =====\n{sub_prompt}")
        raw_responses.append(f"\n===== 子诊断: {sub} =====\n{sub_response or ''}")
        if sub_response:
            try:
                parsed = parse_list_from_response(sub_response)
                if isinstance(parsed, list):
                    all_parsed.append(parsed)
            except Exception:
                pass
    merged = _merge_probability_tuples(all_parsed)
    merged_str = repr(merged)
    audit = (
        f"\n\n[复合诊断合并审计]\n"
        f"[合并后表型项目数: {len(merged)}]\n"
        f"[原始子诊断响应]\n{''.join(raw_responses)}"
    )
    combined_prompt = '\n'.join(all_prompts) + audit
    return combined_prompt, merged_str


def _phase1_generate_names(seed_text, phenotype_type, extra_constraints='', tag=''):
    """阶段1：GPT生成初始表型名称列表（不含概率）。"""
    type_examples = {
        '症状': '咳嗽、气短、胸痛、发热感、乏力、盗汗、食欲减退、咯血',
        '体征': '体温升高、心动过速、呼吸急促、低血压、双肺湿啰音',
        '实验室检查': '白细胞升高、CRP升高、D-二聚体升高、痰培养阳性',
        '影像检查': '胸部CT示肺实变影、胸部X线示浸润影、超声示胸腔积液',
        '功能检查': '肺功能示阻塞性通气功能障碍、心电图示窦性心动过速',
    }
    examples = type_examples.get(phenotype_type, '')

    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地列出该患者可能出现的所有{phenotype_type}名称。

患者信息：{seed_text}

要求：
1. 以Python列表格式输出，每个元素为字符串（{phenotype_type}名称）
2. 名称应包含必要的性质、部位、特征描述，具体且无歧义
3. 尽可能覆盖全面，包括常见和少见的{phenotype_type}
4. **每个名称必须是单一语义原子**，禁止用顿号"、"、斜杠"/"、"和"、"或"、"及"等把多个表型合并为一条；若有多个表型须拆为多个独立元素。正确示例：`'易疲劳'`、`'体力下降'` 各占一项；错误示例：`'易疲劳、体力下降'`。
5. 对检查类条目，每个名称还必须是单一可量化指标或单一异常发现，禁止把多个指标用"和/或/及/并/、"合成一项；错误示例：`'平均跨瓣压差升高并瓣口面积缩小'`。
{extra_constraints}6. 只输出Python列表，不要输出其他内容

参考（仅举例）：{examples}

输出格式示例：
['表现1', '表现2', '表现3', ...]"""

    print(f"    [{tag}] 阶段1: GPT生成初始{phenotype_type}列表...")
    t0 = time.time()
    response = call_gpt5(prompt, tag=f"{tag}-P1")
    names = []
    if response:
        try:
            cleaned = re.sub(r'```python\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            names = ast.literal_eval(cleaned)
            if not isinstance(names, list):
                names = []
        except Exception:
            for line in response.splitlines():
                line = line.strip().strip('-').strip('*').strip()
                if line and not line.startswith('[') and not line.startswith('#'):
                    name = re.sub(r'^\d+[\.\、\)]\s*', '', line).strip().strip("'\"")
                    if name and len(name) > 1:
                        names.append(name)
        print(f"    [{tag}] 阶段1完成: {len(names)}个{phenotype_type} ({round(time.time()-t0,1)}s)")
    else:
        print(f"    [{tag}] 阶段1失败 ({round(time.time()-t0,1)}s)")
    return names, prompt, response


def _phase2_evidence_supplement(seed_text, phenotype_type, initial_names,
                                 evidence_logger=None, evidence_env='yufa', tag=''):
    """阶段2：循证检索补充表型列表。"""
    diagnosis = _extract_diagnosis_from_seed(seed_text)
    type_query_map = {
        '症状': '有哪些症状',
        '体征': '有哪些体征',
        '实验室检查': '有哪些化验检查异常',
        '影像检查': '有哪些影像检查异常',
        '功能检查': '有哪些功能检查异常',
    }
    query_suffix = type_query_map.get(phenotype_type, f'有哪些{phenotype_type}')
    query = f"{diagnosis} {query_suffix}"

    print(f"    [{tag}] 阶段2: 循证检索 '{query}'...")
    evidence_text = call_evidence_api(query, evidence_logger=evidence_logger,
                                       tag=f"{tag}-P2", env=evidence_env)

    if not evidence_text or len(evidence_text.strip()) < 50:
        print(f"    [{tag}] 阶段2: 循证无有效返回，保留阶段1结果")
        return initial_names, '', None

    initial_list_str = '\n'.join([f'  - {n}' for n in initial_names])
    merge_prompt = f"""你是一位资深临床医学专家。请根据循证医学文献和已有列表，补充可能遗漏的{phenotype_type}。

患者信息：{seed_text}

已有{phenotype_type}列表（GPT初步生成）：
{initial_list_str}

循证医学文献参考：
{evidence_text[:3000]}

要求：
1. 根据循证文献，补充上方列表中遗漏的{phenotype_type}
2. 严格去除语义重复项（如"咳嗽"和"干咳"只保留更具体的描述）
3. 保留原列表中所有合理项，仅新增遗漏项
4. 以Python列表格式输出完整的{phenotype_type}名称列表
5. 只输出Python列表，不要输出其他内容

输出格式：
['表现1', '表现2', ...]"""

    print(f"    [{tag}] 阶段2: GPT融合补充...")
    t0 = time.time()
    merge_response = call_gpt5(merge_prompt, tag=f"{tag}-P2-merge")
    merged_names = []
    if merge_response:
        try:
            cleaned = re.sub(r'```python\s*', '', merge_response)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            merged_names = ast.literal_eval(cleaned)
            if not isinstance(merged_names, list):
                merged_names = initial_names
        except Exception:
            merged_names = initial_names
        print(f"    [{tag}] 阶段2完成: {len(initial_names)}→{len(merged_names)}个{phenotype_type} ({round(time.time()-t0,1)}s)")
    else:
        merged_names = initial_names
        print(f"    [{tag}] 阶段2 GPT融合失败，保留阶段1结果")
    return merged_names, merge_prompt, merge_response


def _phase3_evidence_probabilities(seed_text, phenotype_type, final_names, staging_system,
                                    extra_prob_rules='', evidence_logger=None,
                                    evidence_env='yufa', evidence_workers=5, tag=''):
    """阶段3：逐表型循证检索概率，然后一次性GPT生成概率元组列表。"""
    diagnosis = _extract_diagnosis_from_seed(seed_text)
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)

    queries = [f"{diagnosis} {name} 概率" for name in final_names]
    print(f"    [{tag}] 阶段3: 并发循证检索 {len(queries)} 个{phenotype_type}概率...")
    t0 = time.time()
    evidence_map = batch_evidence_queries(
        queries, evidence_logger=evidence_logger,
        tag=f"{tag}-P3", env=evidence_env, max_workers=evidence_workers
    )
    print(f"    [{tag}] 阶段3: 循证检索完成 ({round(time.time()-t0,1)}s)")

    evidence_sections = []
    for name in final_names:
        q = f"{diagnosis} {name} 概率"
        resp = evidence_map.get(q, '')
        snippet = resp[:500].strip() if resp else '无循证数据'
        evidence_sections.append(f"- {name}: {snippet}")
    evidence_block = '\n'.join(evidence_sections)

    prob_fields = '、'.join([f'{name}概率' for name in level_names])
    example_probs = ', '.join([f'0.{2 + i*2}' for i in range(n_levels)])

    severity_block = _severity_rules_for(phenotype_type)

    prob_prompt = f"""你是一位资深临床医学专家。请根据你的医学知识和以下循证文献参考，为每个{phenotype_type}生成在各疾病级别下的出现概率。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

需要生成概率的{phenotype_type}列表及循证参考：
{evidence_block}

要求：
1. 以Python列表格式输出，每个元素为({n_levels+1})元组：({phenotype_type}, {prob_fields})
2. 每个概率为0到1之间的浮点数
3. 概率应综合你的医学知识和循证文献中的数据，若循证文献提供了具体概率数据则优先参考
4. 严重级别的概率通常≥轻度级别的概率
{extra_prob_rules}5. 每个条目必须是单一表型/单一可量化指标，禁止把多个指标或发现用"和/或/及/并/、"合成一项；例如"平均跨瓣压差升高并瓣口面积缩小"必须拆成两个条目
6. 必须为上方列表中的每个{phenotype_type}都生成概率元组，不得遗漏
7. 只输出Python列表，不要输出其他内容

{severity_block}

示例格式：
[('表现1', {example_probs}), ...]"""

    print(f"    [{tag}] 阶段3: GPT生成{len(final_names)}个概率元组...")
    t1 = time.time()
    prob_response = call_gpt5(prob_prompt, tag=f"{tag}-P3-prob")
    if prob_response:
        prob_response = _validate_or_none(
            prob_response,
            lambda text: validate_m1_flat_probability_schema(
                text, staging_system, module_name=tag or f'模块1-{phenotype_type}',
                expected_names=final_names,
            ),
            tag or f'模块1-{phenotype_type}',
        )
        if prob_response:
            parsed = parse_list_from_response(prob_response)
            print(f"    [{tag}] 阶段3完成: {len(parsed)}个概率元组 ({round(time.time()-t1,1)}s)")
    else:
        print(f"    [{tag}] 阶段3 GPT生成失败 ({round(time.time()-t1,1)}s)")
    return prob_prompt, prob_response


def _phase3_symptoms_with_attributes(seed_text, final_names, staging_system,
                                      evidence_logger=None, evidence_env='yufa',
                                      evidence_workers=5, tag=''):
    """阶段3变体：为症状生成嵌套元组（含持续时间、诱发因素、性质、概率）。"""
    diagnosis = _extract_diagnosis_from_seed(seed_text)
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)

    queries = [f"{diagnosis} {name} 概率" for name in final_names]
    print(f"    [{tag}] 阶段3: 并发循证检索 {len(queries)} 个症状概率...")
    t0 = time.time()
    evidence_map = batch_evidence_queries(
        queries, evidence_logger=evidence_logger,
        tag=f"{tag}-P3", env=evidence_env, max_workers=evidence_workers
    )
    print(f"    [{tag}] 阶段3: 循证检索完成 ({round(time.time()-t0,1)}s)")

    evidence_sections = []
    for name in final_names:
        q = f"{diagnosis} {name} 概率"
        resp = evidence_map.get(q, '')
        snippet = resp[:500].strip() if resp else '无循证数据'
        evidence_sections.append(f"- {name}: {snippet}")
    evidence_block = '\n'.join(evidence_sections)

    sub_tuple_desc = '、'.join([
        f'({name}_持续时间, {name}_诱发因素, {name}_性质, {name}_概率)'
        for name in level_names
    ])
    example_subs = ', '.join([
        f"('{['数天', '1-2周', '持续数周', '持续'][min(i,3)]}', "
        f"'{['受凉后', '受凉/吸入刺激物', '无明显诱因', '无明显诱因'][min(i,3)]}', "
        f"'{['间断、轻度', '频繁、中度', '持续、剧烈', '持续、剧烈'][min(i,3)]}', "
        f"0.{2 + i*2})"
        for i in range(n_levels)
    ])

    prob_prompt = f"""你是一位资深临床医学专家。请根据你的医学知识和以下循证文献参考，为每个症状生成在各疾病级别下的嵌套元组（含持续时间、诱发因素、性质、概率）。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

需要生成的症状列表及循证参考：
{evidence_block}

要求：
1. 以Python列表格式输出，每个元素为嵌套元组：(症状名, {sub_tuple_desc})
2. 外层元组第1个元素为症状名称，后续每个元素为该级别的子元组
3. 每个子元组包含4个字段：(持续时间, 诱发因素, 性质, 概率)
   - 持续时间：该症状在此级别下的典型持续时间
   - 诱发因素：该症状在此级别下的典型诱发因素
   - 性质：该症状在此级别下的典型性质特征（间断/持续、钝/锐/烧灼等）
   - 概率：0到1之间的浮点数
4. 不同级别的持续时间、诱发因素、性质可以不同
5. 概率应综合医学知识和循证文献，若循证文献提供了具体数据则优先参考
6. 只包含患者主观能感受并描述的症状，不含客观体征
7. 必须为上方列表中的每个症状都生成元组，不得遗漏
8. 只输出Python列表，不要输出其他内容

{_severity_rules_for('症状')}

示例格式：
[('表现1', {example_subs}), ...]"""

    print(f"    [{tag}] 阶段3: GPT生成{len(final_names)}个症状嵌套元组...")
    t1 = time.time()
    prob_response = call_gpt5(prob_prompt, tag=f"{tag}-P3-prob")
    if prob_response:
        prob_response = _split_compound_symptoms_in_response(prob_response)
        prob_response = _validate_or_none(
            prob_response,
            lambda text: validate_m1_symptom_schema(text, staging_system, expected_names=final_names),
            tag or '模块1.11',
        )
        if prob_response:
            parsed = parse_list_from_response(prob_response)
            print(f"    [{tag}] 阶段3完成: {len(parsed)}个症状嵌套元组 ({round(time.time()-t1,1)}s)")
    else:
        print(f"    [{tag}] 阶段3 GPT生成失败 ({round(time.time()-t1,1)}s)")
    return prob_prompt, prob_response


# ============================================================
# 模块1.0
# ============================================================

_DIAGNOSTIC_CERTAINTY_STAGE_RE = re.compile(
    r'(?:明确诊断|确诊|可能诊断|疑似|排除|不支持诊断)'
)
_RISK_AXIS_RE = re.compile(
    r'(?:风险|危险|预后|死亡|复发|出血|栓塞|卒中|事件).{0,8}'
    r'(?:分层|评分|等级|级别|评估|预测)|(?:低危|中危|高危|极高危)'
)
_RISK_SCORE_NAME_RE = re.compile(
    r'(?:ABCD(?:2|²|3(?:-I)?)|s?PESI|Bova|Wells|(?:改良)?Geneva|GRACE|TIMI|HEART)'
    r'(?:评分|分级|指数)?|'
    r'ESC.{0,16}(?:肺栓塞|PE)',
    re.IGNORECASE,
)
_RISK_OUTCOME_DESCRIPTION_RE = re.compile(
    r'(?:诊断前|患病|阳性).{0,8}(?:概率|可能性)|'
    r'(?:死亡|病死|卒中|出血|复发|事件).{0,8}(?:风险|概率|率)'
)
_CURRENT_SEVERITY_SCORE_RE = re.compile(r'(?:CURB-?65|SOFA|APACHE)', re.IGNORECASE)
_ETIOLOGY_AXIS_RE = re.compile(
    r'(?:病因|病原|亚型|分型|病理类型|组织学类型|分子类型)'
)
_ANATOMICAL_AXIS_RE = re.compile(
    r'(?:Stanford|DeBakey)|'
    r'(?:解剖|部位).{0,8}(?:分型|分类)|'
    r'(?:分型|分类).{0,8}(?:解剖|部位)',
    re.IGNORECASE,
)
_ANATOMICAL_LEVEL_RE = re.compile(
    r'^(?:(?:Stanford|DeBakey)\s*)?(?:[ABCⅠⅡⅢⅣIV]+型|中央型|外周型|周围型|近端型|远端型)$',
    re.IGNORECASE,
)
_GENERIC_SEVERITY_NAME_RE = re.compile(
    r'^(?:临床)?(?:病情)?严重(?:程度|度)?(?:分级|分期)$'
)
_GENERIC_SEVERITY_LEVEL_RE = re.compile(
    r'^(?:轻度|中度|中重度|重度|危重(?:度|型|症)?|极重度|轻症|中症|重症)'
    r'(?:[（(][^）)]+[）)])?$'
)
_SEVERITY_PREFIX_RE = re.compile(r'(?:轻度|中度|重度|危重|极重|轻症|中症|重症|危重症)')
_SEVERITY_AXIS_RE = re.compile(
    r'(?:严重|轻度|中度|重度|危重|极重|早期|晚期|进展|终末|'
    r'代偿|失代偿|器官功能|活动受限|分期|分级|'
    r'\bI{1,3}V?\b|[ⅠⅡⅢⅣⅤ]|[1-4][期级]|'
    r'KDIGO|NYHA|GOLD|TNM|CURB|SOFA|APACHE|Child-Pugh|Killip|Rai|Binet|Ann Arbor|FIGO)'
)
_REVERSED_SEVERITY_FIRST_RE = re.compile(r'(?:重度|危重|极重|晚期|终末|IV级|Ⅳ级|4期|4级)')
_REVERSED_SEVERITY_LAST_RE = re.compile(r'(?:轻度|早期|I级|Ⅰ级|1期|1级)')


def _severity_prefix(level_name):
    match = _SEVERITY_PREFIX_RE.search(str(level_name or ''))
    if not match:
        return ''
    return str(level_name or '')[:match.start()].strip(' -_，,、:：；;（）()[]【】')


def _ensure_single_severity_axis(staging_system):
    """Reject diagnostic, risk, etiology, or subtype axes masquerading as M1 severity."""
    name = str(staging_system.get('name', '') or '').strip()
    levels = staging_system.get('levels', []) or []
    level_names = [str(level.get('name', '') or '').strip() for level in levels]
    descriptions = [str(level.get('description', '') or '').strip() for level in levels]
    combined = ' '.join([name] + level_names + descriptions)

    if _DIAGNOSTIC_CERTAINTY_STAGE_RE.search(' '.join(level_names)):
        raise ValueError('严重程度分级不得使用明确/可能/排除等诊断确定性分级')
    if (_RISK_AXIS_RE.search(name) or _RISK_SCORE_NAME_RE.search(name)
            or all(_RISK_AXIS_RE.search(x) for x in level_names)
            or (not _CURRENT_SEVERITY_SCORE_RE.search(name)
                and all(_RISK_OUTCOME_DESCRIPTION_RE.search(x) for x in descriptions))):
        raise ValueError('严重程度分级不得使用风险分层、预后风险或事件风险轴')
    if _ETIOLOGY_AXIS_RE.search(name):
        raise ValueError('严重程度分级不得使用病因、亚型、分型或类型轴')
    if (_ANATOMICAL_AXIS_RE.search(name)
            or all(_ANATOMICAL_LEVEL_RE.fullmatch(x) for x in level_names)):
        raise ValueError('严重程度分级不得使用解剖部位或解剖分类轴')
    prefixes = [_severity_prefix(level_name) for level_name in level_names]
    non_empty_prefixes = {prefix for prefix in prefixes if prefix}
    generic_level_names = all(
        _GENERIC_SEVERITY_LEVEL_RE.fullmatch(x) for x in level_names
    )
    shared_disease_prefix = (
        all(prefixes) and len(non_empty_prefixes) == 1
    )
    if (_GENERIC_SEVERITY_NAME_RE.fullmatch(name)
            and not (generic_level_names or shared_disease_prefix)):
        raise ValueError('临床严重程度分级的 level name 必须体现递增严重度，不得隐藏病因或亚型')
    if len(non_empty_prefixes) >= 2:
        raise ValueError('严重程度分级不得混合不同疾病、病因、亚型或组织学前缀')
    if not _SEVERITY_AXIS_RE.search(combined):
        raise ValueError('严重程度分级必须体现单一病情严重度或疾病进展轴')
    if levels:
        first_text = f"{level_names[0]} {descriptions[0]}"
        last_text = f"{level_names[-1]} {descriptions[-1]}"
        if _REVERSED_SEVERITY_FIRST_RE.search(first_text) and _REVERSED_SEVERITY_LAST_RE.search(last_text):
            raise ValueError('严重程度分级必须随 level 递增，不得倒序')


def validate_m1_staging_system(response_text):
    """校验目标疾病的严重程度分级，禁止把诊断确定性当作患者分期。"""
    parsed = response_text if isinstance(response_text, dict) else parse_staging_system(response_text)
    if not parsed:
        raise ValueError('输出不是有效的分期字典')
    levels = parsed.get('levels')
    if not isinstance(levels, list) or not 2 <= len(levels) <= 4:
        raise ValueError('严重程度分级必须包含2-4级')
    proportions = []
    seen_names = set()
    for expected_level, level in enumerate(levels, start=1):
        if not isinstance(level, dict):
            raise ValueError(f'第{expected_level}级不是字典')
        if level.get('level') != expected_level:
            raise ValueError('分级 level 必须从1连续递增')
        level_name = str(level.get('name', '')).strip()
        if not level_name:
            raise ValueError(f'第{expected_level}级名称为空')
        if level_name in seen_names:
            raise ValueError(f'分级 level name 重复: {level_name}')
        seen_names.add(level_name)
        try:
            proportion = float(level.get('proportion'))
        except (TypeError, ValueError):
            raise ValueError(f'第{expected_level}级 proportion 不是数字')
        if not 0 < proportion <= 1:
            raise ValueError(f'第{expected_level}级 proportion 超出(0,1]')
        proportions.append(proportion)
    if not math.isclose(sum(proportions), 1.0, abs_tol=1e-6):
        raise ValueError('proportion 之和必须为1.0')
    _ensure_single_severity_axis(parsed)
    return parsed


def _build_m10_staging_prompt(seed_text, previous_response=None, validation_error=None):
    """Build the M1.0 prompt, including concrete feedback after a rejected answer."""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，为该疾病生成一个单一、递增的当前临床严重程度分级。

患者信息：{seed_text}

要求：
1. 只使用一个当前临床严重程度/疾病进展轴（如GOLD、NYHA、KDIGO或基于当前症状、器官功能、血流动力学和并发症的临床严重度分级）。各级必须都是已患该疾病的患者
2. 将该分级系统归纳为最多4个级别。若原系统超过4级则合并相近级别；若不足4级则保持原样（2-3级均可）
3. 为每个级别指定在该类患者中的分布比例（proportion），所有比例之和必须等于1.0
4. 严禁使用诊断确定性、未来风险/预后风险、病因、解剖部位、分子或组织学亚型作为分级轴。即使该疾病最常用的是风险分层或解剖分型，也不要输出该系统；改按当前症状负担、器官功能、血流动力学及并发症划分轻度/中度/重度
5. level 必须从1开始连续递增，且严重度/疾病进展程度随 level 递增，不得倒序
6. 以Python字典格式输出，方便程序解析
7. 只输出Python字典，不要输出其他内容

输出格式示例（以社区获得性肺炎为例）：
{{"name": "临床严重度分级", "levels": [{{"level": 1, "name": "轻度", "description": "症状较轻，无重要器官功能障碍或血流动力学不稳定", "proportion": 0.5}}, {{"level": 2, "name": "中度", "description": "症状明显或出现局部/单一器官功能受损，但无持续性血流动力学不稳定", "proportion": 0.3}}, {{"level": 3, "name": "重度", "description": "严重症状，出现重要器官功能障碍、血流动力学不稳定或危及生命并发症", "proportion": 0.2}}]}}"""

    if validation_error:
        prior = str(previous_response or '')[:6000]
        error = str(validation_error)[:1000]
        prompt += f"""

上一轮输出未通过程序校验。请只修正分级轴和格式，不要重复同类错误。
<上一轮输出>
{prior}
</上一轮输出>
<校验错误>
{error}
</校验错误>

本轮必须完全放弃上一轮中的风险、预后、病因或解剖分型轴。若没有合适的公认严重度系统，name 使用“临床严重度分级”，level name 使用“轻度/中度/重度”，description 仅依据当前症状、器官功能、血流动力学和并发症，并保持严重度递增。"""
    return prompt


def _call_m10_staging_once(seed_text, previous_response=None, validation_error=None):
    """Run one M1.0 attempt while preserving rejected output and its reason."""
    prompt = _build_m10_staging_prompt(seed_text, previous_response, validation_error)

    print(f"  [模块1.0] 正在生成疾病分级系统...")
    t0 = time.time()
    raw_response = call_gpt5(prompt, tag="模块1.0")
    if raw_response:
        try:
            parsed = validate_m1_staging_system(raw_response)
            n_levels = len(parsed.get('levels', []))
            print(f"  [模块1.0] 生成了 {n_levels} 级分级系统: {parsed.get('name', '未知')}，用时 {round(time.time()-t0,1)}s")
            return prompt, raw_response, raw_response, None
        except ValueError as exc:
            print(f"  [模块1.0] ⚠️ 分级系统校验失败: {exc}，用时 {round(time.time()-t0,1)}s")
            return prompt, None, raw_response, str(exc)
    else:
        print(f"  [模块1.0] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
        return prompt, None, raw_response, '模型返回为空'


def module_1_0_generate_staging_system(seed_text):
    """Module 1.0: Generate and validate one disease-severity axis."""
    prompt, response, _raw_response, _error = _call_m10_staging_once(seed_text)
    return prompt, response


def _retry_m10_generate_staging_system(
        seed_text,
        max_retries=DEFAULT_MODULE_MAX_RETRIES,
        retry_sleep=DEFAULT_MODULE_RETRY_SLEEP):
    """Retry M1.0 with the prior invalid answer and validation error as feedback."""
    previous_response = None
    validation_error = None
    for attempt in range(1, max_retries + 1):
        prompt, response, raw_response, error = _call_m10_staging_once(
            seed_text,
            previous_response=previous_response,
            validation_error=validation_error,
        )
        if response is not None:
            return prompt, response
        previous_response = raw_response
        validation_error = error
        if attempt < max_retries:
            print(f"  [模块1.0] ⚠️ 第 {attempt} 次模块调用失败，"
                  f"{retry_sleep}s 后进行第 {attempt+1} 次带反馈重试...")
            time.sleep(retry_sleep)
        else:
            print(f"  [模块1.0] ❌ 经过 {max_retries} 次带反馈重试仍失败，终止后续流程")
            raise RuntimeError(f"模块1.0 经过 {max_retries} 次带反馈重试仍失败")


# ============================================================
# 模块1.01
# ============================================================

def module_1_01_generate_differential_diagnosis(seed_text):
    """模块1.01 鉴别诊断生成（紧随1.0之后）"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，生成该疾病需要鉴别的病因亚型和其他疾病。

患者信息：{seed_text}

要求：
1. 以Python字典格式输出，包含两个键：'病因亚型' 和 '鉴别疾病'
2. '病因亚型'：该疾病需要鉴别的不同病因或亚型（如社区获得性肺炎需鉴别阻塞性肺炎、吸入性肺炎等），最多4个
   - 若该疾病无需细分病因亚型，可以为空列表 []
3. '鉴别疾病'：临床上需要与该疾病鉴别的其他不同疾病（如肺炎需鉴别肺结核、肺脓肿等），最多4个
   - 【强制要求】'鉴别疾病' 列表不得为空，必须至少包含1个鉴别疾病
4. 鉴别项应选择临床上最常见、最容易混淆的病因/疾病
5. 只输出Python字典，不要输出其他内容

输出格式示例（以社区获得性肺炎为例）：
{{'病因亚型': ['阻塞性肺炎', '吸入性肺炎', '真菌性肺炎'], '鉴别疾病': ['肺结核', '肺脓肿', '肺癌', '肺栓塞']}}"""

    print(f"  [模块1.01] 正在生成鉴别诊断...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.01")
    if response:
        try:
            cleaned = re.sub(r'```python\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, dict):
                diff_diseases = parsed.get('鉴别疾病', [])
                n_subtypes = len(parsed.get('病因亚型', []))
                n_diseases = len(diff_diseases)
                if not diff_diseases:
                    print(f"  [模块1.01] ⚠️ 鉴别疾病为空，需重试")
                    return prompt, None
                print(f"  [模块1.01] 生成了 {n_subtypes} 个病因亚型 + {n_diseases} 个鉴别疾病，用时 {round(time.time()-t0,1)}s")
            else:
                print(f"  [模块1.01] ⚠️ 返回非字典格式，用时 {round(time.time()-t0,1)}s")
                return prompt, None
        except Exception:
            print(f"  [模块1.01] ⚠️ 解析失败，用时 {round(time.time()-t0,1)}s")
            return prompt, None
    else:
        print(f"  [模块1.01] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


# ============================================================
# 模块1.02
# ============================================================

def module_1_02_generate_laterality(seed_text):
    """模块1.02 侧别判断：判断该疾病是单侧/双侧/可以是两者。"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，判断该疾病在临床上的发病侧别特征。

患者信息：{seed_text}

请从以下三种情况中选择一种，只输出对应的英文字符串，不输出任何解释：
- "unilateral"：该疾病通常只能单侧发病（如单侧自发性气胸、单侧睾丸扭转、单侧肺叶病变等）
- "bilateral"：该疾病通常只能双侧或全身性发病（如双侧间质性肺炎、系统性疾病等）
- "either"：该疾病可以单侧发病，也可以双侧发病（如大多数感染性疾病、肿瘤等）

只输出以上三个字符串之一，不要输出其他内容。"""

    print(f"  [模块1.02] 正在判断侧别类型...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.02")
    raw = (response or '').strip().strip('"\'').lower()
    if raw not in ('unilateral', 'bilateral', 'either'):
        print(f"  [模块1.02] ⚠️ 无法解析侧别类型: {raw!r}，默认 'either'，用时 {round(time.time()-t0,1)}s")
        return prompt, 'either'
    print(f"  [模块1.02] 侧别类型: {raw}，用时 {round(time.time()-t0,1)}s")
    return prompt, raw


# ============================================================
# 模块1.11 / 1.12
# ============================================================

_COMPOUND_SPLIT_RE = re.compile(r'[、/／,，]|(?:\s和\s)|(?:\s或\s)|(?:\s及\s)')


def _split_compound_symptom_name(name):
    """将复合症状名按顿号/斜杠/和/或/及等分隔符拆分为单一语义原子列表。

    - 每个子名至少保留 2 个字符（避免将"3/4 级"这类单位误拆）。
    - 若拆分结果只剩 1 项，直接返回 [原名]。
    """
    if not isinstance(name, str):
        return [name]
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(name) if p and p.strip()]
    parts = [p for p in parts if len(p) >= 2]
    if len(parts) <= 1:
        return [name.strip()] if name.strip() else []
    return parts


def _split_compound_symptoms_in_response(response_text):
    """对 1.11 的文本响应（Python 列表字面量），解析后拆分复合症状名，
    保留每个子名对应的属性元组，最后重新序列化回字符串。

    若解析失败或不是列表，原样返回。
    """
    if not response_text:
        return response_text
    try:
        parsed = _parse_strict_list_literal(response_text, module_name='模块1.11')
    except ValueError:
        return response_text
    if not isinstance(parsed, list) or not parsed:
        return response_text
    new_items = []
    seen_names = set()
    for item in parsed:
        if not isinstance(item, (list, tuple)) or not item:
            new_items.append(item)
            continue
        name = item[0]
        rest = tuple(item[1:])
        sub_names = _split_compound_symptom_name(name)
        if len(sub_names) <= 1:
            new_items.append(item)
            if isinstance(name, str):
                seen_names.add(name)
            continue
        for sn in sub_names:
            base = sn
            final = base
            suffix = 2
            while final in seen_names:
                final = f'{base}({suffix})'
                suffix += 1
            seen_names.add(final)
            new_items.append((final,) + rest)
    try:
        return repr(new_items)
    except Exception:
        return response_text


def module_1_11_generate_symptoms(seed_text, staging_system=None,
                                   evidence_logger=None, enable_evidence=False,
                                   evidence_env='yufa', evidence_workers=5):
    """模块1.11 患者症状生成（主观症状）- V3循证增强版"""
    tag = '模块1.11'
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)

    if enable_evidence:
        t_total = time.time()
        extra_constraints = """5. 只生成患者主观能感受并描述的症状
6. 不包含需要医生查体才能发现的客观体征
7. 症状名称应包含必要的性质、部位、特征描述（如"吸气性呼吸困难"而非仅"呼吸困难"）
8. **每个症状必须是单一语义原子**，禁止用顿号"、"、斜杠"/"、"和"、"或"、"及"拼接多个症状；有多个症状时必须拆为多个独立条目。错误示例：`'易疲劳、体力耐受下降'`、`'恶心、食欲差'`、`'注意力不集中/记忆力下降'`。
"""
        names, p1_prompt, p1_resp = _phase1_generate_names(
            seed_text, '症状', extra_constraints=extra_constraints, tag=tag)
        if not names:
            print(f"  [{tag}] 阶段1无结果，回退V2流程")
            return _module_1_11_v2_fallback(seed_text, staging_system)
        merged_names, p2_prompt, p2_resp = _phase2_evidence_supplement(
            seed_text, '症状', names, evidence_logger=evidence_logger,
            evidence_env=evidence_env, tag=tag)
        prob_prompt, prob_response = _phase3_symptoms_with_attributes(
            seed_text, merged_names, staging_system,
            evidence_logger=evidence_logger,
            evidence_env=evidence_env, evidence_workers=evidence_workers, tag=tag)
        prob_response = _split_compound_symptoms_in_response(prob_response)
        all_prompts = f"[阶段1]\n{p1_prompt}\n\n[阶段2]\n{p2_prompt or '(跳过)'}\n\n[阶段3]\n{prob_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, prob_response

    return _module_1_11_v2_fallback(seed_text, staging_system)


def _module_1_11_v2_fallback(seed_text, staging_system):
    """模块1.11 V2原始逻辑（无循证）- 嵌套元组版"""
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)

    sub_tuple_desc = '、'.join([
        f'({name}_持续时间, {name}_诱发因素, {name}_性质, {name}_概率)'
        for name in level_names
    ])
    example_subs = ', '.join([
        f"('{['数天', '1-2周', '持续数周', '持续'][min(i,3)]}', "
        f"'{['受凉后', '受凉/吸入刺激物', '无明显诱因', '无明显诱因'][min(i,3)]}', "
        f"'{['间断、轻度', '频繁、中度', '持续、剧烈', '持续、剧烈'][min(i,3)]}', "
        f"0.{2 + i*2})"
        for i in range(n_levels)
    ])
    example_tuple = f"('干咳', {example_subs})"

    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地生成该患者在确诊前所有可能出现的主观症状（患者自己能感受并描述的不适）。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

要求：
1. 以Python列表格式输出，每个元素为嵌套元组：(症状名, {sub_tuple_desc})
2. 外层元组第1个元素为症状名称（字符串），后续每个元素为该级别的子元组
3. 每个子元组包含4个字段：(持续时间, 诱发因素, 性质, 概率)
   - 持续时间：该症状在此级别下的典型持续时间，如"数分钟"、"数小时"、"数天"、"持续数周"等
   - 诱发因素：该症状在此级别下的典型诱发因素，如"活动后"、"受凉后"、"无明显诱因"等
   - 性质：该症状在此级别下的典型性质特征，如"间断"、"持续"、"钝痛"、"锐痛"、"烧灼样"、"阵发性"等
   - 概率：0到1之间的浮点数，该症状在此级别下出现的概率
4. 不同级别的持续时间、诱发因素、性质可以不同（通常严重级别持续更久、诱因更不明显、性质更剧烈）
5. 只生成患者主观能感受并描述的症状，不包含需要医生查体才能发现的客观体征
6. 症状名称应包含必要的部位、特征描述（如"吸气性呼吸困难"而非仅"呼吸困难"）
7. 尽可能覆盖全面，包括常见和少见的症状
8. **每个症状必须是单一语义原子**，禁止用顿号"、"、斜杠"/"、"和"、"或"、"及"拼接多个症状；有多个症状时必须拆为多个独立元组。错误示例：`'易疲劳、体力耐受下降'`、`'恶心、食欲差'`。
8b. 对于该疾病最具特征性的必发症状（几乎所有患者都会出现的核心主诉）：其全分期概率均应设为 1.0。
9. 只输出Python列表，不要输出其他内容

{_severity_rules_for('症状')}

示例格式：
[{example_tuple}, ...]"""

    print(f"  [模块1.11] 正在生成主观症状（{n_levels}级嵌套元组）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.11", max_tokens=16384)
    if response:
        response = _split_compound_symptoms_in_response(response)
        response = _validate_or_none(
            response,
            lambda text: validate_m1_symptom_schema(text, staging_system),
            "模块1.11",
        )
        if response:
            parsed = parse_list_from_response(response)
            print(f"  [模块1.11] 生成了 {len(parsed)} 个主观症状，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.11] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def module_1_12_generate_signs(seed_text, staging_system=None,
                                evidence_logger=None, enable_evidence=False,
                                evidence_env='yufa', evidence_workers=5):
    """模块1.12 患者异常体征生成（查体才能获得的客观体征）- V3循证增强版"""
    tag = '模块1.12'
    if enable_evidence:
        t_total = time.time()
        extra_constraints = """5. 只生成需要医生查体才能获得的客观体征（生命体征、视触叩听）
6. 【强制要求】必须包含五项生命体征的异常改变：体温、心率、呼吸频率、血压、血氧饱和度
7. 体征名称应描述具体发现并注明异常阈值（如"心动过速（心率>100次/分）"）
"""
        names, p1_prompt, p1_resp = _phase1_generate_names(
            seed_text, '体征', extra_constraints=extra_constraints, tag=tag)
        if not names:
            return _module_1_12_v2_fallback(seed_text, staging_system)
        merged_names, p2_prompt, p2_resp = _phase2_evidence_supplement(
            seed_text, '体征', names, evidence_logger=evidence_logger,
            evidence_env=evidence_env, tag=tag)
        extra_prob_rules = """5. 只包含需要医生查体才能获得的客观体征
6. 【强制要求】必须包含五项生命体征异常：体温、心率、呼吸频率、血压、血氧饱和度
7. 体征名称应描述具体发现并注明异常阈值
"""
        prob_prompt, prob_response = _phase3_evidence_probabilities(
            seed_text, '体征', merged_names, staging_system,
            extra_prob_rules=extra_prob_rules, evidence_logger=evidence_logger,
            evidence_env=evidence_env, evidence_workers=evidence_workers, tag=tag)
        all_prompts = f"[阶段1]\n{p1_prompt}\n\n[阶段2]\n{p2_prompt or '(跳过)'}\n\n[阶段3]\n{prob_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, prob_response
    return _module_1_12_v2_fallback(seed_text, staging_system)


def _module_1_12_v2_fallback(seed_text, staging_system):
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)
    prob_fields = '、'.join([f'{name}出现的概率' for name in level_names])
    example_probs = ', '.join([f'0.{2 + i*3}' for i in range(n_levels)])
    example_tuple = f"('体温升高（>38.0℃）', {example_probs})"

    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地生成该患者在确诊前医生查体时所有可能发现的异常体征。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

要求：
1. 以Python列表格式输出，每个元素为({n_levels+1})元组：(体征, {prob_fields})
2. 每个概率为0到1之间的浮点数，分别对应该体征在各级别下出现的概率
3. 只生成需要医生查体才能获得的客观体征，包括：
   - 生命体征（体温、心率、呼吸频率、血压、血氧饱和度）
   - 视诊（一般外观、呼吸形态、皮肤黏膜等）
   - 触诊（胸廓动度、语颤等）
   - 叩诊（肺部叩诊音等）
   - 听诊（呼吸音、啰音、心音等）
4. 【强制要求】必须包含以下五项生命体征的异常改变（若该疾病可能导致以下任一异常，则必须列出，不得遗漏）：
   - 体温升高/降低（如发热、超高热、低体温）
   - 心率增快/减慢（如心动过速、心动过缓）
   - 呼吸频率增快（如呼吸急促、呼吸频率增快）
   - 血压升高/降低（如血压升高、低血压）
   - 血氧饱和度下降（SpO2降低）
5. 体征名称应描述具体发现，并在括号中注明异常阈值（如"心动过速（心率>100次/分）"）
6. 尽可能覆盖全面，包括常见和少见的体征
7b. 对于该疾病最具特征性的必有体征（几乎所有阶段必然存在的金标准体征）：其全分期概率均应设为 1.0。
7. 每个条目必须是单一表型/单一可量化指标，禁止把多个指标或发现用"和/或/及/并/、"合成一项；例如"平均跨瓣压差升高并瓣口面积缩小"必须拆成两个条目
8. 只输出Python列表，不要输出其他内容

{_severity_rules_for('体征')}

示例格式：
[{example_tuple}, ...]"""

    print(f"  [模块1.12] 正在生成查体体征（{n_levels}级概率）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.12")
    if response:
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.12"),
            "模块1.12",
        )
        if response:
            parsed = parse_list_from_response(response)
            print(f"  [模块1.12] 生成了 {len(parsed)} 个查体体征，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.12] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


# ============================================================
# 模块1.21 / 1.22 / 1.23
# ============================================================

def _build_gold_std_desc(staging_system):
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    parts = []
    if n_levels >= 3:
        parts.append(f'{level_names[0]}概率0.6-0.9')
        for mid_name in level_names[1:-1]:
            parts.append(f'{mid_name}概率0.75-1.0')
        parts.append(f'{level_names[-1]}概率1.0')
    elif n_levels == 2:
        parts.append(f'{level_names[0]}概率0.6-1.0')
        parts.append(f'{level_names[-1]}概率1.0')
    else:
        parts.append('概率1.0')
    return '、'.join(parts)


def module_1_21_generate_lab_tests(seed_text, staging_system=None,
                                    evidence_logger=None, enable_evidence=False,
                                    evidence_env='yufa', evidence_workers=5):
    """模块1.21 实验室检查生成 - V3循证增强版（v4 支持多疾病拆分）"""
    tag = '模块1.21'
    diag_full = _extract_diagnosis_from_seed(seed_text)
    sub_list = _split_compound_diagnosis(diag_full)
    if len(sub_list) > 1:
        print(f"  [{tag}] 检测到复合诊断 {sub_list}，对每个子诊断分别建库后合并")
        def _single(sub_seed):
            return _module_1_21_single(sub_seed, staging_system, evidence_logger,
                                        enable_evidence, evidence_env, evidence_workers)
        prompt, response = _run_with_subdiagnoses(seed_text, sub_list, _single, tag=tag)
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name=tag),
            tag,
        )
        return prompt, response
    return _module_1_21_single(seed_text, staging_system, evidence_logger,
                                enable_evidence, evidence_env, evidence_workers)


def _module_1_21_single(seed_text, staging_system, evidence_logger,
                         enable_evidence, evidence_env, evidence_workers):
    tag = '模块1.21'
    gold_std_desc = _build_gold_std_desc(staging_system)
    if enable_evidence:
        t_total = time.time()
        extra_constraints = """5. 只生成实验室检查（血液、尿液、粪便、培养、免疫学、肿瘤标志物、生化等），不包含影像和功能检查
6. 只生成可能出现异常的检查项，数值型结果只需定性方向（升高/降低）
7. 必须覆盖：血常规、肝肾功、电解质、凝血、炎症指标
"""
        names, p1_prompt, p1_resp = _phase1_generate_names(
            seed_text, '实验室检查', extra_constraints=extra_constraints, tag=tag)
        if not names:
            return _module_1_21_v2_fallback(seed_text, staging_system)
        merged_names, p2_prompt, p2_resp = _phase2_evidence_supplement(
            seed_text, '实验室检查', names, evidence_logger=evidence_logger,
            evidence_env=evidence_env, tag=tag)
        extra_prob_rules = f"""5. 只包含实验室检查，不含影像和功能检查
6. 概率规则：金标准/确诊性检查：{gold_std_desc}；非金标准按临床发生率
"""
        prob_prompt, prob_response = _phase3_evidence_probabilities(
            seed_text, '实验室检查', merged_names, staging_system,
            extra_prob_rules=extra_prob_rules, evidence_logger=evidence_logger,
            evidence_env=evidence_env, evidence_workers=evidence_workers, tag=tag)
        all_prompts = f"[阶段1]\n{p1_prompt}\n\n[阶段2]\n{p2_prompt or '(跳过)'}\n\n[阶段3]\n{prob_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, prob_response
    return _module_1_21_v2_fallback(seed_text, staging_system)


def _module_1_21_v2_fallback(seed_text, staging_system):
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)
    prob_fields = '、'.join([f'{name}出现异常的概率' for name in level_names])
    gold_std_desc = _build_gold_std_desc(staging_system)
    example_probs = ', '.join([f'0.{3 + i*2}' for i in range(n_levels)])
    example_gold_probs = ', '.join(
        [f'0.{4 + i*2}' if i < n_levels - 1 else '0.9' for i in range(n_levels)]
    )
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地生成该患者在确诊前所有可能出现异常的实验室检查结果。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

要求：
1. 以Python列表格式输出，每个元素为({n_levels+1})元组：(检查结果, {prob_fields})
2. 只生成实验室检查（血液检查、尿液检查、粪便检查、培养、免疫学检测、肿瘤标志物、生化检查等），不包含影像检查和功能检查
3. 只生成可能出现异常的检查项，不要列出结果为正常的检查项
4. 数值型检查结果只需输出定性方向（升高或降低），不需要具体数值
5. 每个概率为0到1之间的浮点数，分别对应该检查在各级别下出现异常的概率
6. 必须考虑该疾病对以下常见检查的影响，若会导致异常则必须包含：
   - 血常规（白细胞、中性粒细胞、淋巴细胞、血红蛋白、血小板等）
   - 肝功能（ALT、AST、胆红素、白蛋白等）
   - 肾功能（肌酐、尿素氮等）
   - 电解质（钠、钾、氯、钙等）
   - 凝血功能（PT、APTT、INR、D-二聚体等）
   - 炎症指标（CRP、PCT、ESR等）
7. 概率规则（重要）：
   - 金标准/确诊性检查（特异性病原培养/PCR阳性、特异性抗原/抗体检测、组织活检病理等）：{gold_std_desc}
   - 非金标准的支持性检查：按实际临床发生率设置概率
8. 每个条目必须是单一表型/单一可量化指标，禁止把多个指标用"和/或/及/并/、"合成一项；例如"平均跨瓣压差升高并瓣口面积缩小"必须拆成两个条目
9. 只输出Python列表，不要输出其他内容

{_severity_rules_for('实验室检查')}

示例格式：
[('白细胞升高', {example_probs}), ('咽拭子培养阳性（A组β溶血性链球菌）', {example_gold_probs})]"""

    print(f"  [模块1.21] 正在生成实验室检查（{n_levels}级概率）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.21")
    if response:
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.21"),
            "模块1.21",
        )
        if response:
            parsed = parse_list_from_response(response)
            print(f"  [模块1.21] 生成了 {len(parsed)} 个实验室检查项，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.21] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def module_1_22_generate_imaging(seed_text, staging_system=None,
                                  evidence_logger=None, enable_evidence=False,
                                  evidence_env='yufa', evidence_workers=5):
    """模块1.22 影像检查生成 - V3循证增强版（v4 支持多疾病拆分）"""
    tag = '模块1.22'
    diag_full = _extract_diagnosis_from_seed(seed_text)
    sub_list = _split_compound_diagnosis(diag_full)
    if len(sub_list) > 1:
        print(f"  [{tag}] 检测到复合诊断 {sub_list}，对每个子诊断分别建库后合并")
        def _single(sub_seed):
            return _module_1_22_single(sub_seed, staging_system, evidence_logger,
                                        enable_evidence, evidence_env, evidence_workers)
        prompt, response = _run_with_subdiagnoses(seed_text, sub_list, _single, tag=tag)
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name=tag),
            tag,
        )
        return prompt, response
    return _module_1_22_single(seed_text, staging_system, evidence_logger,
                                enable_evidence, evidence_env, evidence_workers)


def _module_1_22_single(seed_text, staging_system, evidence_logger,
                         enable_evidence, evidence_env, evidence_workers):
    tag = '模块1.22'
    if enable_evidence:
        t_total = time.time()
        extra_constraints = """5. 只生成主要影像学大类别（X线、CT、MRI、超声、PET-CT等），不将同一检查的子类型作为独立条目分别列出
6. 禁止同时列出CT与HRCT/薄层CT等同类变体；每种影像学检查方式只能出现一次
7. 每项应描述具体影像征象/发现，而非完整报告
8. 影像发现名称应包含检查方式和具体征象描述
"""
        names, p1_prompt, p1_resp = _phase1_generate_names(
            seed_text, '影像检查', extra_constraints=extra_constraints, tag=tag)
        if not names:
            return _module_1_22_v2_fallback(seed_text, staging_system)
        merged_names, p2_prompt, p2_resp = _phase2_evidence_supplement(
            seed_text, '影像检查', names, evidence_logger=evidence_logger,
            evidence_env=evidence_env, tag=tag)
        extra_prob_rules = """5. 只包含影像学检查，不含实验室和功能检查
6. 金标准/确诊性影像发现的最高级别概率设为1.0
7. 禁止同时列出同一检查方式的不同子类型变体（如CT与HRCT/薄层CT不能都列出）
"""
        prob_prompt, prob_response = _phase3_evidence_probabilities(
            seed_text, '影像检查', merged_names, staging_system,
            extra_prob_rules=extra_prob_rules, evidence_logger=evidence_logger,
            evidence_env=evidence_env, evidence_workers=evidence_workers, tag=tag)
        all_prompts = f"[阶段1]\n{p1_prompt}\n\n[阶段2]\n{p2_prompt or '(跳过)'}\n\n[阶段3]\n{prob_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, prob_response
    return _module_1_22_v2_fallback(seed_text, staging_system)


def _module_1_22_v2_fallback(seed_text, staging_system):
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)
    prob_fields = '、'.join([f'{name}出现的概率' for name in level_names])
    example_probs = ', '.join([f'0.{3 + i*2}' for i in range(n_levels)])
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地生成该患者在确诊前所有可能出现异常的影像学检查发现。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

要求：
1. 以Python列表格式输出，每个元素为({n_levels+1})元组：(影像检查发现, {prob_fields})
2. 只生成主要影像学大类别（胸部X线、胸部CT、MRI、超声、PET-CT等），不将同一检查的子类型作为独立条目分别列出。禁止示例：不得同时列出"胸部CT"和"HRCT"/"薄层CT"；只保留最适合的一个大类。
3. 每项应描述具体的影像征象/发现（如"胸部CT示双肺磨玻璃影"），而非完整的影像报告
4. 只生成可能出现异常的影像发现，不要列出正常的检查结果
5. 每个概率为0到1之间的浮点数，分别对应该影像发现在各级别下出现的概率
6. 影像发现名称应包含检查方式和具体征象描述
7. 金标准/确诊性影像发现的最高级别概率设为1.0
8. 每种影像学检查方式只能出现一次，描述其最重要的阳性发现；禁止同时列出同一检查方式的不同子类型变体
9. 每个条目必须是单一表型/单一可量化指标，禁止把多个指标或发现用"和/或/及/并/、"合成一项；例如"平均跨瓣压差升高并瓣口面积缩小"必须拆成两个条目
10. 只输出Python列表，不要输出其他内容

{_severity_rules_for('影像检查')}

示例格式：
[('胸部X线示肺部浸润影', {example_probs}), ('胸部CT示肺实变伴支气管充气征', {example_probs})]"""

    print(f"  [模块1.22] 正在生成影像检查（{n_levels}级概率）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.22")
    if response:
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.22"),
            "模块1.22",
        )
        if response:
            parsed = parse_list_from_response(response)
            print(f"  [模块1.22] 生成了 {len(parsed)} 个影像检查项，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.22] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def module_1_23_generate_functional_tests(seed_text, staging_system=None,
                                           evidence_logger=None, enable_evidence=False,
                                           evidence_env='yufa', evidence_workers=5):
    """模块1.23 功能检查生成 - V3循证增强版（v4 支持多疾病拆分）"""
    tag = '模块1.23'
    diag_full = _extract_diagnosis_from_seed(seed_text)
    sub_list = _split_compound_diagnosis(diag_full)
    if len(sub_list) > 1:
        print(f"  [{tag}] 检测到复合诊断 {sub_list}，对每个子诊断分别建库后合并")
        def _single(sub_seed):
            return _module_1_23_single(sub_seed, staging_system, evidence_logger,
                                        enable_evidence, evidence_env, evidence_workers)
        prompt, response = _run_with_subdiagnoses(seed_text, sub_list, _single, tag=tag)
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name=tag),
            tag,
        )
        return prompt, response
    return _module_1_23_single(seed_text, staging_system, evidence_logger,
                                enable_evidence, evidence_env, evidence_workers)


def _module_1_23_single(seed_text, staging_system, evidence_logger,
                         enable_evidence, evidence_env, evidence_workers):
    tag = '模块1.23'
    if enable_evidence:
        t_total = time.time()
        extra_constraints = """5. 只生成功能性检查（肺功能、心电图、6分钟步行试验、睡眠多导图等），不含实验室和影像检查；超声心动图及其结构/血流动力学发现统一归入影像检查
6. 若该疾病通常不涉及功能检查，可输出较短列表
"""
        names, p1_prompt, p1_resp = _phase1_generate_names(
            seed_text, '功能检查', extra_constraints=extra_constraints, tag=tag)
        if not names:
            return _module_1_23_v2_fallback(seed_text, staging_system)
        merged_names, p2_prompt, p2_resp = _phase2_evidence_supplement(
            seed_text, '功能检查', names, evidence_logger=evidence_logger,
            evidence_env=evidence_env, tag=tag)
        extra_prob_rules = """5. 只包含功能性检查，不含实验室和影像检查
6. 金标准/确诊性功能检查的最高级别概率设为1.0
"""
        prob_prompt, prob_response = _phase3_evidence_probabilities(
            seed_text, '功能检查', merged_names, staging_system,
            extra_prob_rules=extra_prob_rules, evidence_logger=evidence_logger,
            evidence_env=evidence_env, evidence_workers=evidence_workers, tag=tag)
        all_prompts = f"[阶段1]\n{p1_prompt}\n\n[阶段2]\n{p2_prompt or '(跳过)'}\n\n[阶段3]\n{prob_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, prob_response
    return _module_1_23_v2_fallback(seed_text, staging_system)


def _module_1_23_v2_fallback(seed_text, staging_system):
    n_levels = _get_num_levels(staging_system)
    level_names = _get_level_names(staging_system)
    staging_levels_text = _build_staging_levels_text(staging_system)
    prob_fields = '、'.join([f'{name}出现异常的概率' for name in level_names])
    example_probs = ', '.join([f'0.{2 + i*2}' for i in range(n_levels)])
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，尽可能全面地生成该患者在确诊前所有可能出现异常的功能检查结果。

患者信息：{seed_text}

该疾病的分级系统：
{staging_levels_text}

要求：
1. 以Python列表格式输出，每个元素为({n_levels+1})元组：(功能检查结果, {prob_fields})
2. 只生成功能性检查（肺功能检查、心电图、6分钟步行试验、睡眠多导图、支气管激发试验、运动心肺功能等），不包含实验室检查和影像检查；超声心动图及其结构/血流动力学发现统一归入影像检查，本模块不得输出超声、CT、MRI、PET、X线或造影项目
3. 只生成可能出现异常的功能检查项，不要列出结果为正常的检查项
4. 功能检查结果描述应为定性异常发现（如"肺功能示阻塞性通气功能障碍"），不需要具体数值
5. 每个概率为0到1之间的浮点数，分别对应该检查在各级别下出现异常的概率
6. 若该疾病通常不涉及功能检查或功能检查不太可能异常，可输出较短列表甚至空列表 []
7. 金标准/确诊性功能检查的最高级别概率设为1.0
8. 每个条目必须是单一表型/单一可量化指标，禁止把多个指标或发现用"和/或/及/并/、"合成一项；例如"平均跨瓣压差升高并瓣口面积缩小"必须拆成两个条目
9. 只输出Python列表，不要输出其他内容

{_severity_rules_for('功能检查')}

示例格式（以COPD为例）：
[('肺功能示阻塞性通气功能障碍（FEV1/FVC<70%）', {example_probs}), ('支气管舒张试验阴性', {example_probs})]"""

    print(f"  [模块1.23] 正在生成功能检查（{n_levels}级概率）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.23")
    if response:
        response = _validate_or_none(
            response,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.23"),
            "模块1.23",
        )
        if response:
            parsed = parse_list_from_response(response)
            print(f"  [模块1.23] 生成了 {len(parsed)} 个功能检查项，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.23] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def _audit_gold_standard_probs(tuples_str, seed_text, phenotype_type, staging_system, tag=''):
    """查核：LLM识别金标准表型 → 仅将最高严重级概率校正为1.0。
    返回 (audit_prompt, audit_response, corrected_tuples_str, names_corrected)。"""
    tuples = parse_list_from_response(tuples_str)
    if not tuples:
        return '', '', tuples_str, []

    n_levels = _get_num_levels(staging_system)
    items_text = '\n'.join(
        f'{i+1}. {t[0]}  [当前概率: {list(t[1:])}]'
        for i, t in enumerate(tuples) if len(t) >= 2
    )
    diagnosis = _extract_diagnosis_from_seed(seed_text)
    prompt = (
        f'以下是为「{diagnosis}」生成的{phenotype_type}列表，'
        f'请识别其中对该诊断是金标准/定义性诊断证据，且在最高严重级患者中几乎必然异常的条目。'
        f'若检查可能未实施、受采样时机或治疗影响而阴性、存在替代证据，或属于多项择一的证据族，均不得选入。'
        f'输出这些条目的名称（仅第1列），每行一个，不输出其他内容。'
        f'如果没有符合条件的条目，必须输出 []，不得返回空内容。\n\n'
        f'{items_text}'
    )
    response = call_gpt5(prompt, tag=tag or f'查核-{phenotype_type}', max_retries=5)
    if not response:
        raise RuntimeError(f"{tag or f'查核-{phenotype_type}'} 空响应，金标准查核失败")

    if response.strip() in {'[]', '无', '没有', '无符合条目', '__NONE__'}:
        response = '[]'
        gold_names = set()
    elif _strip_code_fence(response).startswith('['):
        listed_names = _parse_strict_list_literal(
            response, module_name=tag or f'查核-{phenotype_type}'
        )
        if not all(isinstance(name, str) and name.strip() for name in listed_names):
            raise ValueError(f"{tag or f'查核-{phenotype_type}'} 名称列表格式无效")
        gold_names = {name.strip() for name in listed_names}
    else:
        gold_names = {line.strip() for line in response.strip().splitlines() if line.strip()}

    candidate_names = {str(t[0]).strip() for t in tuples if t}
    unknown_names = gold_names - candidate_names
    if unknown_names:
        raise ValueError(
            f"{tag or f'查核-{phenotype_type}'} 返回了原列表之外的名称: {sorted(unknown_names)!r}"
        )

    corrected = []
    names_corrected = []
    for t in tuples:
        if not t or len(t) < 2:
            corrected.append(t)
            continue
        name = t[0]
        matched = name in gold_names
        if matched and t[-1] != 1.0:
            corrected.append(tuple(t[:-1]) + (1.0,))
            names_corrected.append(name)
        else:
            corrected.append(t)

    return prompt, response, repr(corrected), names_corrected


def module_1_211_audit_lab_tests(tuples_str, seed_text, staging_system, tag='模块1.211'):
    return _audit_gold_standard_probs(tuples_str, seed_text, '实验室检查', staging_system, tag=tag)


def module_1_221_audit_imaging(tuples_str, seed_text, staging_system, tag='模块1.221'):
    return _audit_gold_standard_probs(tuples_str, seed_text, '影像检查', staging_system, tag=tag)


def module_1_231_audit_functional(tuples_str, seed_text, staging_system, tag='模块1.231'):
    return _audit_gold_standard_probs(tuples_str, seed_text, '功能检查', staging_system, tag=tag)


# ============================================================
# 模块1.24 / 1.25 / 1.26（v4 重构：合并诊断与鉴别诊断关键化验；新增关键症状/体征）
# ============================================================

def _safe_parse_json_obj(response_text):
    """从 LLM 响应中提取 JSON 对象。失败返回 {}。"""
    if not response_text:
        return {}
    text = str(response_text).strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # 截取第一个 { 到最后一个 }
    l = text.find('{')
    r = text.rfind('}')
    if l != -1 and r != -1 and r > l:
        text = text[l:r+1]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def module_1_24_extract_key_lab_tests(seed_text, differential_diagnosis_text,
                                       lab_text, imaging_text, functional_text,
                                       tag='模块1.24'):
    """模块1.24（v4 合并版）：单次调用 LLM，同时输出"诊断关键化验"和"鉴别诊断关键化验"。
    返回 (prompt, response, parsed_dict)。
    parsed_dict 形如 {"诊断关键化验":[...], "鉴别诊断关键化验":[...]}。"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息及已生成的检查表型库，分别给出：
（A）支持/确诊该诊断最关键的检查项目
（B）用于鉴别诊断（特别是病因鉴别）所必需、但不在 A 中的额外关键检查项目

患者信息：{seed_text}

鉴别诊断（来自模块1.01）：
{differential_diagnosis_text if differential_diagnosis_text else '(无)'}

已生成的实验室检查概率库：
{(lab_text or '(无)')[:2500]}

已生成的影像检查概率库：
{(imaging_text or '(无)')[:2500]}

已生成的功能检查概率库：
{(functional_text or '(无)')[:2500]}

【强约束 — 必须使用"整体检查项目名"，禁止写单一指标或单一异常征象】
正确示例（整体项目名）：✓血常规  ✓呼吸道病原检测  ✓肺功能+激发试验  ✓动脉血气  ✓胸部CT
错误示例（单一指标）：✗血红蛋白  ✗鼻病毒检测  ✗肺功能激发试验  ✗PaO2  ✗胸部CT-肺纹理

要求：
1. 严格输出 JSON（不要任何注释、不要 Markdown 代码块、不要其他文字），结构如下：
{{
  "诊断关键化验": ["...", "..."],
  "鉴别诊断关键化验": ["...", "..."]
}}
2. "诊断关键化验" 用于支持/确诊该诊断（含分型、分期、严重程度评估）
3. "鉴别诊断关键化验" 用于排除/区分上述鉴别诊断（特别是病因鉴别），不要与第一项重复
4. 每类最多 5 项；如果不足 5 项不要硬凑
5. 两类内部不得重复，两类之间也不得重复
6. 检查项目名必须为"整体项目名"（参见上方正反例），不要写单一指标
7. 这里的"化验"含义放宽：含实验室、影像、功能检查；只要是项目级别即可
"""
    print(f"  [{tag}] 单次调用合并生成 诊断关键化验 + 鉴别诊断关键化验 ...")
    t0 = time.time()
    response = call_gpt5(prompt, tag=tag)
    parsed = _safe_parse_json_obj(response) if response else {}
    if not parsed:
        print(f"  [{tag}] ⚠️ JSON 解析失败/为空（{round(time.time()-t0,1)}s）")
        parsed = {}
    else:
        for k in ("诊断关键化验", "鉴别诊断关键化验"):
            v = parsed.get(k, [])
            if not isinstance(v, list):
                parsed[k] = []
        # 截断到 5
        parsed["诊断关键化验"] = [str(x).strip() for x in parsed.get("诊断关键化验", []) if str(x).strip()][:5]
        parsed["鉴别诊断关键化验"] = [str(x).strip() for x in parsed.get("鉴别诊断关键化验", []) if str(x).strip()][:5]
        # 去除两类间重复
        diag_set = set(parsed["诊断关键化验"])
        parsed["鉴别诊断关键化验"] = [x for x in parsed["鉴别诊断关键化验"] if x not in diag_set]
        print(f"  [{tag}] 完成: 诊断 {len(parsed['诊断关键化验'])} / 鉴别 {len(parsed['鉴别诊断关键化验'])} ({round(time.time()-t0,1)}s)")
    return prompt, response, parsed


def module_1_25_generate_key_symptoms(seed_text, differential_diagnosis_text,
                                       symptoms_text, tag='模块1.25'):
    """模块1.25（v4 新语义）：单次调用 LLM，输出"诊断关键症状"+"鉴别诊断关键症状"。
    可精确到性质/部位/诱因/放射等多维度（同一症状的不同维度算多项）。
    返回 (prompt, response, parsed_dict)。"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息和已生成的症状库，分别给出：
（A）支持/确诊该诊断最关键的症状要点
（B）用于鉴别诊断（特别是病因鉴别）所必需、但不在 A 中的额外关键症状要点

患者信息：{seed_text}

鉴别诊断（来自模块1.01）：
{differential_diagnosis_text if differential_diagnosis_text else '(无)'}

已生成的症状概率库：
{(symptoms_text or '(无)')[:3500]}

【粒度说明】
- 症状要点可精确到具体维度，如：性质、部位、诱因、放射、加重/缓解、伴随、起病速度等
- 同一症状的不同维度算作多项（例如"胸痛-压榨性"、"胸痛-向左肩放射"、"胸痛-劳累诱发"可同时入选）
- 写法风格示例：「胸痛-压榨性」「咳嗽-夜间为主」「呼吸困难-平卧加重」「腹痛-餐后2小时」

要求：
1. 严格输出 JSON（不要 Markdown、不要解释），结构：
{{
  "诊断关键症状": ["...", "..."],
  "鉴别诊断关键症状": ["...", "..."]
}}
2. 每类最多 10 项；不够 10 不要硬凑
3. 两类内部不得重复，两类之间也不得重复
4. 表述要具体到维度，不要只写笼统的"咳嗽""胸痛"
"""
    print(f"  [{tag}] 单次调用生成 诊断关键症状 + 鉴别诊断关键症状 ...")
    t0 = time.time()
    response = call_gpt5(prompt, tag=tag)
    parsed = _safe_parse_json_obj(response) if response else {}
    if not parsed:
        print(f"  [{tag}] ⚠️ JSON 解析失败/为空（{round(time.time()-t0,1)}s）")
        parsed = {}
    else:
        for k in ("诊断关键症状", "鉴别诊断关键症状"):
            v = parsed.get(k, [])
            if not isinstance(v, list):
                parsed[k] = []
        parsed["诊断关键症状"] = [str(x).strip() for x in parsed.get("诊断关键症状", []) if str(x).strip()][:10]
        parsed["鉴别诊断关键症状"] = [str(x).strip() for x in parsed.get("鉴别诊断关键症状", []) if str(x).strip()][:10]
        diag_set = set(parsed["诊断关键症状"])
        parsed["鉴别诊断关键症状"] = [x for x in parsed["鉴别诊断关键症状"] if x not in diag_set]
        print(f"  [{tag}] 完成: 诊断 {len(parsed['诊断关键症状'])} / 鉴别 {len(parsed['鉴别诊断关键症状'])} ({round(time.time()-t0,1)}s)")
    return prompt, response, parsed


def module_1_26_generate_key_signs(seed_text, differential_diagnosis_text,
                                    signs_text, tag='模块1.26'):
    """模块1.26（v4 新增）：单次调用 LLM，输出"诊断关键体征"+"鉴别诊断关键体征"。
    精确到检查项目维度（如"肺部听诊"、"颈静脉视诊"）。
    返回 (prompt, response, parsed_dict)。"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息和已生成的体征库，分别给出：
（A）支持/确诊该诊断最关键的查体项目（按"检查项目维度"组织，如"肺部听诊"、"心脏听诊"、"颈静脉视诊"、"腹部触诊"、"周围水肿检查"）
（B）用于鉴别诊断（特别是病因鉴别）所必需、但不在 A 中的额外关键查体项目

患者信息：{seed_text}

鉴别诊断（来自模块1.01）：
{differential_diagnosis_text if differential_diagnosis_text else '(无)'}

已生成的体征概率库：
{(signs_text or '(无)')[:3500]}

要求：
1. 严格输出 JSON（不要 Markdown、不要解释），结构：
{{
  "诊断关键体征": ["...", "..."],
  "鉴别诊断关键体征": ["...", "..."]
}}
2. 每类最多 5 项；不够 5 不要硬凑
3. 必须按"检查项目维度"组织（如"肺部听诊"、"颈静脉视诊"），不要写单一异常体征（如不要写"双肺湿啰音"，应写"肺部听诊"）
4. 两类内部不得重复，两类之间也不得重复

写法示例：✓肺部听诊  ✓颈静脉视诊  ✓心尖搏动触诊  ✓周围水肿检查  ✓腹部触诊
错误示例：✗双肺湿啰音  ✗颈静脉怒张  ✗下肢凹陷性水肿
"""
    print(f"  [{tag}] 单次调用生成 诊断关键体征 + 鉴别诊断关键体征 ...")
    t0 = time.time()
    response = call_gpt5(prompt, tag=tag)
    parsed = _safe_parse_json_obj(response) if response else {}
    if not parsed:
        print(f"  [{tag}] ⚠️ JSON 解析失败/为空（{round(time.time()-t0,1)}s）")
        parsed = {}
    else:
        for k in ("诊断关键体征", "鉴别诊断关键体征"):
            v = parsed.get(k, [])
            if not isinstance(v, list):
                parsed[k] = []
        parsed["诊断关键体征"] = [str(x).strip() for x in parsed.get("诊断关键体征", []) if str(x).strip()][:5]
        parsed["鉴别诊断关键体征"] = [str(x).strip() for x in parsed.get("鉴别诊断关键体征", []) if str(x).strip()][:5]
        diag_set = set(parsed["诊断关键体征"])
        parsed["鉴别诊断关键体征"] = [x for x in parsed["鉴别诊断关键体征"] if x not in diag_set]
        print(f"  [{tag}] 完成: 诊断 {len(parsed['诊断关键体征'])} / 鉴别 {len(parsed['鉴别诊断关键体征'])} ({round(time.time()-t0,1)}s)")
    return prompt, response, parsed


# ============================================================
# 模块1.31 / 1.32 / 1.5 / 1.6
# ============================================================

def _parse_staged_treatment(treatment_text, staging_system):
    """解析按分级分期格式化的治疗文本为字典 {级别名: 治疗文本}。"""
    if not treatment_text or not staging_system:
        return {}
    level_names = _get_level_names(staging_system)
    result = {}
    current_level = None
    current_lines = []
    for line in treatment_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched_level = None
        for lname in level_names:
            if stripped.startswith(f'【{lname}】'):
                matched_level = lname
                break
        if matched_level:
            if current_level and current_lines:
                result[current_level] = '\n'.join(current_lines)
            current_level = matched_level
            remainder = stripped[len(f'【{matched_level}】'):].strip()
            current_lines = [remainder] if remainder else []
        else:
            if current_level is not None:
                current_lines.append(stripped)
    if current_level and current_lines:
        result[current_level] = '\n'.join(current_lines)
    if not result:
        for lname in level_names:
            result[lname] = treatment_text
    return result


def module_1_31_generate_standard_treatments(seed_text, staging_system=None,
                                              evidence_logger=None, enable_evidence=False,
                                              evidence_env='yufa', evidence_workers=5):
    """模块1.31 标准治疗方案生成 - V3循证增强版（按分级分期逐级生成）"""
    tag = '模块1.31'
    if enable_evidence and staging_system:
        t_total = time.time()
        diagnosis = _extract_diagnosis_from_seed(seed_text)
        level_names = _get_level_names(staging_system)
        levels = staging_system.get('levels', [])
        all_prompts = []
        all_sections = []
        for lvl in levels:
            lname = lvl.get('name', '')
            ldesc = lvl.get('description', '')
            ev_query = f"{diagnosis} {lname}({ldesc}) 治疗"
            print(f"  [{tag}] 循证检索: '{ev_query}'...")
            ev_text = call_evidence_api(ev_query, evidence_logger=evidence_logger,
                                         tag=f"{tag}-{lname}", env=evidence_env)
            ev_section = ''
            if ev_text and len(ev_text.strip()) >= 50:
                ev_section = f"\n循证医学文献参考：\n{ev_text[:3000]}\n"
            level_prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，为该疾病在特定严重度级别下列出标准治疗方案。

患者信息：{seed_text}
当前级别：{lname} - {ldesc}
{ev_section}
要求：
1. 仅列出该级别（{lname}）患者适用的标准治疗方案
2. 按治疗层级分类列出，格式为"层级名称：治疗1、治疗2、..."
3. 只写治疗名称，不加剂量、频次、机制说明
4. 对同类治疗概括列出，不逐一罗列
5. 只列与该诊断直接相关的治疗
6. 若有循证文献参考，优先参考循证数据
7. 直接输出分层文本，不加标题

示例输出：
一线治疗：xxx
支持/对症治疗：xxx"""
            print(f"  [{tag}] GPT生成 {lname} 治疗方案...")
            t0 = time.time()
            level_response = call_gpt5(level_prompt, tag=f"{tag}-{lname}")
            if level_response:
                all_sections.append(f"【{lname}】\n{level_response.strip()}")
                print(f"  [{tag}] {lname} 治疗生成完成 ({round(time.time()-t0,1)}s)")
            else:
                all_sections.append(f"【{lname}】\n(生成失败)")
                print(f"  [{tag}] {lname} 治疗生成失败 ({round(time.time()-t0,1)}s)")
            all_prompts.append(f"[{lname}]\n{level_prompt}")
        combined_response = '\n\n'.join(all_sections)
        combined_prompt = '\n\n'.join(all_prompts)
        print(f"  [{tag}] 分级治疗方案生成完成: {len(levels)}个级别 ({round(time.time()-t_total,1)}s)")
        return combined_prompt, combined_response
    return _module_1_31_v2_fallback(seed_text, staging_system)


def _module_1_31_v2_fallback(seed_text, staging_system=None):
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，列出该疾病当前的标准治疗方案。

患者信息：{seed_text}

要求：
1. 按照治疗层级分类列出，格式为"层级名称：治疗1、治疗2、..."
2. 只写治疗名称，不加剂量、频次、机制说明或其他解释
3. 层级划分示例（根据疾病实际情况选用合适的层级）：
   - 一线治疗：最优先使用的标准方案
   - 二线/难治性治疗：一线无效或不耐受时的替代方案
   - 支持/对症治疗：辅助缓解症状的措施
   - 手术/有创操作：需要手术或操作的治疗（若适用）
   - 并发症处理：针对常见并发症的处理措施（若适用）
4. 对于同类/相似的治疗方案概括列出即可
5. 只列与该诊断直接相关的治疗，不列预防性措施
6. 直接输出分层文本，不加标题或额外说明

示例输出（以社区获得性肺炎为例）：
一线治疗：β-内酰胺类抗生素（阿莫西林克拉维酸钾/头孢菌素）、氟喹诺酮类抗生素
支持/对症治疗：退热（对乙酰氨基酚/布洛芬）、止咳化痰、补液
重症/并发症处理：吸氧、机械通气（重度呼吸衰竭）、ICU监护"""

    print(f"  [模块1.31] 正在生成标准治疗方案...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.31")
    if response:
        lines = [l for l in response.strip().splitlines() if l.strip()]
        print(f"  [模块1.31] 生成了 {len(lines)} 个治疗层级，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.31] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def module_1_32_generate_adr_library(seed_text, standard_treatment_text=''):
    """模块1.32 不良反应（ADR）概率库生成"""
    std_tx_section = ''
    if standard_treatment_text and standard_treatment_text.strip():
        std_tx_section = f"\n该疾病的标准治疗方案（来自模块1.31）：\n{standard_treatment_text}\n"

    prompt = f"""你是一位资深临床药学专家。请根据以下患者信息和标准治疗方案，生成各治疗可能引起的不良反应（ADR）概率库。

患者信息：{seed_text}{std_tx_section}
要求：
1. 以Python列表格式输出，每个元素为 (治疗名称, 不良反应名称, 临床表现, 发生概率, 严重程度) 五元组
2. 治疗名称：直接引用标准治疗中的具体治疗/药物名称
3. 不良反应名称：该治疗已知的不良反应名称
4. 临床表现：该不良反应的具体临床表现描述（患者可能的症状/体征）
5. 发生概率：该不良反应在接受该治疗的患者中的发生概率（0到1之间的浮点数），应符合药品说明书或文献报道的实际发生率
6. 严重程度：分为"轻度"、"中度"、"重度"三级
   - 轻度：不影响日常活动，通常可自行缓解或简单对症处理
   - 中度：影响日常活动，需要药物干预或调整治疗方案
   - 重度：危及生命或导致严重后果，需要紧急处理或停药
7. 列出常见不良反应（发生率≥5%）以及虽然罕见但严重的不良反应（如过敏性休克、肝衰竭等）
8. 只输出Python列表，不要输出其他内容

示例格式（以抗生素治疗为例）：
[('阿莫西林克拉维酸钾', '胃肠道反应', '恶心、呕吐、腹泻', 0.15, '轻度'), ('阿莫西林克拉维酸钾', '过敏性休克', '血压骤降、呼吸困难、意识丧失', 0.001, '重度')]"""

    print(f"  [模块1.32] 正在生成不良反应库...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.32")
    if response:
        parsed = parse_list_from_response(response)
        print(f"  [模块1.32] 生成了 {len(parsed)} 个不良反应条目，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.32] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


def module_1_5_generate_comorbidities(seed_text):
    """模块1.5 患者伴随疾病生成（含人口学适用范围 A9）"""
    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，生成该患者在确诊当前疾病之前可能已经患有的其他疾病（合并症/基础疾病）。

患者信息：{seed_text}

要求：
1. 以 Python 列表格式输出，每个元素为 `(疾病名称, 概率, 适用最小年龄, 适用最大年龄, 适用性别)` **5 元组**。
   - 概率：0–1 之间的浮点数，表示此类患者在确诊当前疾病时已患有该基础疾病的流行病学概率。
   - 适用最小年龄 / 适用最大年龄：该伴随疾病**合理出现**的年龄区间（整数）。
     对没有明确年龄限制的疾病填 `0, 120`；对儿童病填如 `0, 14`；对老年病填如 `50, 120`；对妊娠相关填 `15, 49`（育龄）。
   - 适用性别：`"男"` / `"女"` / `"通用"`；妊娠、前列腺增生、子宫内膜异位等请给出对应性别，其余通用疾病写 `"通用"`。
2. 只列出在确诊当前疾病之前就已存在的慢性病、基础疾病或常见合并症，不包括当前疾病本身。
3. 考虑患者的年龄、性别特点，以及当前诊断疾病常见的基础病。
4. 尽量覆盖常见的合并症，概率应符合流行病学实际。
5. **严禁输出明显矛盾的条目**，如对老年男性输出"妊娠"、对 20 岁女性输出"前列腺增生"、同时出现"1 型糖尿病"和"2 型糖尿病"等。
6. 只输出 Python 列表，不要输出其它内容。

示例格式（为 65 岁男性肺炎患者示意，数值仅为占位）：
[('高血压', 0.45, 30, 120, '通用'),
 ('2型糖尿病', 0.25, 30, 120, '通用'),
 ('慢性阻塞性肺疾病', 0.20, 40, 120, '通用'),
 ('冠心病', 0.15, 40, 120, '通用'),
 ('前列腺增生', 0.30, 50, 120, '男')]"""

    print(f"  [模块1.5] 正在生成伴随疾病...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.5")
    if response:
        parsed = parse_list_from_response(response)
        print(f"  [模块1.5] 生成了 {len(parsed)} 个伴随疾病，模块用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.5] ❌ 生成失败，模块用时 {round(time.time()-t0,1)}s")
    return prompt, response


# ============================================================
# 模块1.51
# ============================================================

def module_1_51_generate_complication_phenotypes(seed_text, comorbidities_text):
    """模块1.51 为每种伴随疾病生成异常表型库（体征/化验/影像/功能）。"""
    comorbidities_list = parse_list_from_response(comorbidities_text)
    comorbidity_names = []
    for item in comorbidities_list:
        if isinstance(item, (list, tuple)) and item:
            comorbidity_names.append(str(item[0]))
        elif isinstance(item, str) and item.strip():
            comorbidity_names.append(item.strip())
    if not comorbidity_names:
        print(f"  [模块1.51] 无伴随疾病，跳过")
        return '', '[]'

    comorbidities_str = '\n'.join([f'- {n}' for n in comorbidity_names])
    prompt = f"""你是一位资深临床医学专家。以下是一位患者（{seed_text}）可能合并的伴随疾病。
请为每种伴随疾病生成其常见的异常表型（体征、化验、影像、功能检查），
以便在患者模拟中，当该伴随疾病被采样时，相关表型异常也被同步纳入。

伴随疾病列表：
{comorbidities_str}

要求：
1. 以Python列表格式输出，每个元素为4元组：(伴随疾病名称, 表型类型, 表型名称, 出现概率)
   - 伴随疾病名称：与输入列表完全一致
   - 表型类型：从 ['体征', '实验室检查', '影像检查', '功能检查'] 中选一个
   - 表型名称：具体表现（如"血压>140/90mmHg"、"空腹血糖升高"、"左室壁轻度增厚"）
   - 出现概率：0-1之间的浮点数
2. 每种伴随疾病至少生成 2 个表型、最多 6 个
3. 只输出体征/化验/影像/功能表型，不输出症状（症状在问诊阶段自然涉及）
4. 只输出Python列表，不要输出其他内容

示例（高血压患者）：
[('高血压', '体征', '血压升高(>140/90mmHg)', 0.95),
 ('高血压', '实验室检查', '血清肌酐轻度升高', 0.20),
 ('高血压', '影像检查', '超声示左室壁轻度增厚', 0.25)]"""

    print(f"  [模块1.51] 正在生成并发症表型库（{len(comorbidity_names)}种伴随疾病）...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.51")
    if response:
        parsed = parse_list_from_response(response)
        print(f"  [模块1.51] 生成了 {len(parsed)} 个表型条目，用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.51] ❌ 生成失败，用时 {round(time.time()-t0,1)}s")
    return prompt, response


# ============================================================
# 模块1.6
def module_1_6_generate_treatment_factors(seed_text, standard_treatment_text='',
                                           evidence_logger=None, enable_evidence=False,
                                           evidence_env='yufa', evidence_workers=5):
    """模块1.6 治疗影响因素概率库生成 - V3循证增强版（GPT归纳治疗类别 → 逐类循证 → GPT融合）"""
    tag = '模块1.6'
    if enable_evidence and standard_treatment_text and standard_treatment_text.strip():
        t_total = time.time()
        diagnosis = _extract_diagnosis_from_seed(seed_text)
        cat_prompt = f"""你是一位资深临床医学专家。请将以下治疗方案归纳为几种主要治疗类别（如抗感染治疗、支持治疗、手术治疗等）。

治疗方案：
{standard_treatment_text}

要求：
1. 以Python列表格式输出，每个元素为字符串（治疗类别名称）
2. 归纳为3-8个类别即可
3. 只输出Python列表

输出示例：['抗感染治疗', '支持治疗', '手术治疗']"""

        print(f"  [{tag}] 阶段1: GPT归纳治疗类别...")
        t0 = time.time()
        cat_response = call_gpt5(cat_prompt, tag=f"{tag}-P1")
        categories = []
        if cat_response:
            try:
                cleaned = re.sub(r'```python\s*', '', cat_response)
                cleaned = re.sub(r'```\s*', '', cleaned).strip()
                categories = ast.literal_eval(cleaned)
                if not isinstance(categories, list):
                    categories = []
            except Exception:
                categories = []
            print(f"  [{tag}] 阶段1完成: {len(categories)}个治疗类别 ({round(time.time()-t0,1)}s)")
        else:
            print(f"  [{tag}] 阶段1失败，回退V2流程")
            return _module_1_6_v2_fallback(seed_text, standard_treatment_text)
        if not categories:
            return _module_1_6_v2_fallback(seed_text, standard_treatment_text)

        ev_queries = [f"{diagnosis} {cat} 疗效影响因素" for cat in categories]
        print(f"  [{tag}] 阶段2: 并发循证检索 {len(ev_queries)} 个治疗类别...")
        ev_map = batch_evidence_queries(
            ev_queries, evidence_logger=evidence_logger,
            tag=f"{tag}-P2", env=evidence_env, max_workers=evidence_workers
        )
        ev_sections = []
        for cat in categories:
            q = f"{diagnosis} {cat} 疗效影响因素"
            resp = ev_map.get(q, '')
            snippet = resp[:1500].strip() if resp else '无循证数据'
            ev_sections.append(f"[{cat}]\n{snippet}")
        evidence_block = '\n\n'.join(ev_sections)

        std_tx_section = f"\n该疾病的标准治疗方案（'影响什么治疗'字段应直接引用以下治疗名称）：\n{standard_treatment_text}\n"
        final_prompt = f"""你是一位资深临床医学专家。请根据以下患者信息、标准治疗方案和循证医学文献参考，生成可能影响治疗效果的因素。

患者信息：{seed_text}{std_tx_section}
循证医学文献参考（按治疗类别分类）：
{evidence_block}

要求：
1. 以Python列表格式输出，每个元素为 (因素, 影响什么治疗, 具体影响, 概率) 四元组
2. 主要输出患者本身可能潜在存在的因素（作为患者的隐藏设置），如合并未控制的基础疾病、不良生活习惯、特殊生理状态等
3. 只关注"疗效"层面：哪些因素会导致治疗反应变差、疾病难以控制、或容易复发
4. 不要包含治疗决策层面的内容（如过敏禁忌、药物选择调整等）
5. 影响什么治疗：直接引用标准治疗中的具体治疗名称
6. 具体影响：说明该因素导致疗效下降的机制，要有临床解释
7. 概率：该因素在该类患者中出现的概率（0-1），应符合流行病学实际
8. 若循证文献提供了相关数据则优先参考
9. 只输出Python列表，不要输出其他内容

正确示例：
[('合并未控制的胃食管反流', '吸入激素', '反流物刺激气道维持高反应性', 0.3), ('长期吸烟', '所有治疗', '损伤气道上皮降低激素敏感性', 0.2)]"""

        print(f"  [{tag}] 阶段3: GPT融合生成治疗影响因素...")
        t1 = time.time()
        final_response = call_gpt5(final_prompt, tag=f"{tag}-P3")
        if final_response:
            parsed = parse_list_from_response(final_response)
            print(f"  [{tag}] 阶段3完成: {len(parsed)}个因素 ({round(time.time()-t1,1)}s)")
        else:
            print(f"  [{tag}] 阶段3失败 ({round(time.time()-t1,1)}s)")
        all_prompts = f"[阶段1-归纳]\n{cat_prompt}\n\n[阶段3-融合]\n{final_prompt}"
        print(f"  [{tag}] 三阶段循证增强完成 ({round(time.time()-t_total,1)}s)")
        return all_prompts, final_response
    return _module_1_6_v2_fallback(seed_text, standard_treatment_text)


def _module_1_6_v2_fallback(seed_text, standard_treatment_text=''):
    std_tx_section = ''
    if standard_treatment_text and standard_treatment_text.strip():
        std_tx_section = f"\n该疾病的标准治疗方案（来自模块1.31，'影响什么治疗'字段应直接引用以下治疗名称）：\n{standard_treatment_text}\n"

    prompt = f"""你是一位资深临床医学专家。请根据以下患者信息，生成在该患者接受针对所患疾病的标准治疗时，可能影响治疗效果的因素及其具体影响。

患者信息：{seed_text}{std_tx_section}
要求：
1. 以Python列表格式输出，每个元素为 (因素, 影响什么治疗, 具体影响, 概率) 四元组
2. 主要输出患者本身可能潜在存在的因素（作为患者的隐藏设置），如合并未控制的基础疾病、不良生活习惯、特殊生理状态等，这些因素在就诊时不一定被医生发现
3. 只关注"疗效"层面：即患者已经在接受该疾病的标准治疗，哪些因素会导致治疗反应变差、疾病难以控制、或容易复发
4. 不要包含治疗决策层面的内容（如过敏禁忌、药物选择调整、剂量调整等）
5. 影响什么治疗：直接引用上方标准治疗中的具体治疗名称；如果同一种因素对所有治疗都有影响，可以写"所有治疗"
6. 具体影响：说明该因素通过什么机制导致疗效下降，要有临床解释
7. 概率：该因素在该类患者中出现的概率，0到1之间的浮点数，应符合流行病学实际
8. 只输出Python列表，不要输出其他内容

正确示例（以支气管哮喘为例）：
[('合并未控制的胃食管反流', '吸入激素及支气管扩张剂', '反流物反复刺激气道，维持气道高反应性，即使规律用药哮喘仍难以控制', 0.3), ('长期吸烟', '所有治疗', '烟草烟雾持续损伤气道上皮并降低激素敏感性，同时加重气道重塑，使所有控制性治疗疗效降低', 0.2)]"""

    print(f"  [模块1.6] 正在生成治疗影响因素概率库...")
    t0 = time.time()
    response = call_gpt5(prompt, tag="模块1.6")
    if response:
        parsed = parse_list_from_response(response)
        print(f"  [模块1.6] 生成了 {len(parsed)} 个治疗影响因素，模块用时 {round(time.time()-t0,1)}s")
    else:
        print(f"  [模块1.6] ❌ 生成失败，模块用时 {round(time.time()-t0,1)}s")
    return prompt, response


# ============================================================
# 单种子处理函数（模块1串行流程）
# ============================================================

def _existing_m1_library_is_valid(row, value_col, input_col, output_col,
                                  validator, audit_cols=()):
    """已有 M1 库只有通过当前 schema 校验才可续跑跳过；否则清空并触发重跑。"""
    value = row.get(value_col, '')
    if not str(value or '').strip():
        return False
    try:
        row[value_col] = validator(value)
        return True
    except ValueError as exc:
        print(f"  ⚠️ 已有 {value_col} 无效，将重跑对应M1模块: {exc}")
        for col in (value_col, input_col, output_col, *audit_cols):
            row[col] = ''
        return False


def process_seed_module1(seed_text, output_dir,
                          enable_evidence=False, evidence_env='yufa',
                          evidence_workers=DEFAULT_EVIDENCE_WORKERS):
    """
    处理单个种子的模块1部分：
    1.0 → 1.02 → 1.11 → 1.12 → 1.21 → 1.22 → 1.23 → 1.5 → 1.51

    当前问诊模拟不消费 1.01、1.24、1.25、1.26、1.31、1.32、1.6，
    因此这些模块不进入执行链，也不写入 CSV。
    生成/续接 CSV row_1（概率库行），仅写入 row_1，不生成患者行。

    Returns:
        dict: {'status': 'success'/'error', 'seed': ..., 'file': ...}
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', seed_text.split('#')[0].strip())
    csv_path = os.path.join(output_dir, f'{safe_name}.csv')

    evidence_logger = None
    if enable_evidence:
        evidence_logger = EvidenceLogger(output_dir, safe_name)
        _get_evidence_api(env=evidence_env, timeout=DEFAULT_EVIDENCE_TIMEOUT,
                          max_retries=DEFAULT_EVIDENCE_MAX_RETRIES)

    print(f"\n{'='*60}")
    print(f"[模块1] 处理种子: {seed_text}")
    print(f"输出文件: {csv_path}")
    if enable_evidence:
        print(f"循证检索: 已启用 (环境={evidence_env}, 并发={evidence_workers})")
    print(f"{'='*60}")

    try:
        existing_row_1, existing_patients = _load_existing_csv(csv_path)

        if existing_row_1 is not None:
            row_1 = existing_row_1
            print(f"  📂 加载已有CSV（{len(existing_patients)}个患者行），续接运行")
        else:
            row_1 = _empty_row()
            row_1[COL_SEED] = seed_text

        def _save():
            patient_rows = [existing_patients[k] for k in sorted(existing_patients.keys())]
            _save_csv(csv_path, [row_1] + patient_rows)

        t_module1 = time.time()
        print(f"\n--- 模块1: 表型概率库生成 ---")

        # 模块1.0
        staging_system = None
        if row_1.get(COL_STAGING_SYSTEM, '').strip():
            try:
                staging_system = validate_m1_staging_system(row_1[COL_STAGING_SYSTEM])
                print(f"  ⏭️  模块1.0 已有数据，跳过")
            except ValueError as exc:
                print(f"  ⚠️ 已有模块1.0无效，将清空分级及依赖库后重跑: {exc}")
                for col in (
                    COL_STAGING_SYSTEM, COL_M10_INPUT, COL_M10_OUTPUT,
                    COL_SYMPTOMS, COL_M111_INPUT, COL_M111_OUTPUT,
                    COL_SIGNS, COL_M112_INPUT, COL_M112_OUTPUT,
                    COL_LAB_TESTS, COL_M121_INPUT, COL_M121_OUTPUT,
                    COL_IMAGING, COL_M122_INPUT, COL_M122_OUTPUT,
                    COL_FUNCTIONAL_TESTS, COL_M123_INPUT, COL_M123_OUTPUT,
                    COL_M211_INPUT, COL_M211_OUTPUT,
                    COL_M221_INPUT, COL_M221_OUTPUT,
                    COL_M231_INPUT, COL_M231_OUTPUT,
                ):
                    row_1[col] = ''
                _save()
        else:
            pass

        if staging_system is None:
            p10, r10 = _retry_m10_generate_staging_system(seed_text)
            row_1[COL_M10_INPUT] = p10 or ''
            row_1[COL_M10_OUTPUT] = r10 or ''
            row_1[COL_STAGING_SYSTEM] = r10 or ''
            staging_system = validate_m1_staging_system(r10) if r10 else None
            _save()
            print(f"  📝 模块1.0 已保存到CSV")

        if not staging_system:
            raise RuntimeError('模块1.0 分级系统生成失败，停止后续M1/M2流程')

        level_names = _get_level_names(staging_system)
        num_levels = _get_num_levels(staging_system)
        print(f"  分级系统: {staging_system.get('name','')} ({num_levels}级: {', '.join(level_names)})")

        _ev_kwargs = {
            'evidence_logger': evidence_logger,
            'enable_evidence': enable_evidence,
            'evidence_env': evidence_env,
            'evidence_workers': evidence_workers,
        }

        # 模块1.02
        if row_1.get(COL_LATERALITY_TYPE, '').strip():
            print(f"  ⏭️  模块1.02 已有数据，跳过")
        else:
            p102, r102 = _retry_module_call(
                module_1_02_generate_laterality,
                args=(seed_text,),
                module_name="模块1.02"
            )
            row_1[COL_M102_INPUT] = p102 or ''
            row_1[COL_M102_OUTPUT] = r102 or ''
            row_1[COL_LATERALITY_TYPE] = r102 or 'either'
            _save()
            print(f"  📝 模块1.02 已保存到CSV")

        # 诊断拆分（v4 主题2.5：多疾病同时建库）
        if not row_1.get(COL_DIAGNOSIS_COMPONENTS, '').strip():
            try:
                _diag_full = _extract_diagnosis_from_seed(seed_text)
                _sub_list = _split_compound_diagnosis(_diag_full)
                row_1[COL_DIAGNOSIS_COMPONENTS] = json.dumps(_sub_list, ensure_ascii=False)
                if len(_sub_list) > 1:
                    print(f"  📝 诊断拆分: {_diag_full} -> {_sub_list}")
                else:
                    print(f"  📝 诊断拆分: {_diag_full} (单一疾病)")
                _save()
            except Exception as _e:
                print(f"  ⚠️ 诊断拆分失败: {_e}")
        else:
            print(f"  ⏭️  诊断拆分 已有数据，跳过")

        # 模块1.11
        if _existing_m1_library_is_valid(
            row_1, COL_SYMPTOMS, COL_M111_INPUT, COL_M111_OUTPUT,
            lambda text: validate_m1_symptom_schema(text, staging_system),
        ):
            print(f"  ⏭️  模块1.11 已有数据，跳过")
        else:
            p111, r111 = _retry_module_call(
                module_1_11_generate_symptoms,
                args=(seed_text,), kwargs={'staging_system': staging_system, **_ev_kwargs},
                module_name="模块1.11"
            )
            row_1[COL_M111_INPUT] = p111 or ''
            row_1[COL_M111_OUTPUT] = r111 or ''
            row_1[COL_SYMPTOMS] = r111 or ''
            _save()
            print(f"  📝 模块1.11 已保存到CSV")

        # 模块1.12
        if _existing_m1_library_is_valid(
            row_1, COL_SIGNS, COL_M112_INPUT, COL_M112_OUTPUT,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.12"),
        ):
            print(f"  ⏭️  模块1.12 已有数据，跳过")
        else:
            p112, r112 = _retry_module_call(
                module_1_12_generate_signs,
                args=(seed_text,), kwargs={'staging_system': staging_system, **_ev_kwargs},
                module_name="模块1.12"
            )
            row_1[COL_M112_INPUT] = p112 or ''
            row_1[COL_M112_OUTPUT] = r112 or ''
            row_1[COL_SIGNS] = r112 or ''
            _save()
            print(f"  📝 模块1.12 已保存到CSV")

        # 模块1.21
        if _existing_m1_library_is_valid(
            row_1, COL_LAB_TESTS, COL_M121_INPUT, COL_M121_OUTPUT,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.21"),
            audit_cols=(COL_M211_INPUT, COL_M211_OUTPUT),
        ):
            print(f"  ⏭️  模块1.21 已有数据，跳过")
        else:
            p121, r121 = _retry_module_call(
                module_1_21_generate_lab_tests,
                args=(seed_text,), kwargs={'staging_system': staging_system, **_ev_kwargs},
                module_name="模块1.21"
            )
            row_1[COL_M121_INPUT] = p121 or ''
            row_1[COL_M121_OUTPUT] = r121 or ''
            row_1[COL_LAB_TESTS] = r121 or ''
            _save()
            print(f"  📝 模块1.21 已保存到CSV")

        # 模块1.211 查核（实验室检查金标准概率）
        if row_1.get(COL_M211_OUTPUT, '').strip():
            print(f"  ⏭️  模块1.211 已有数据，跳过")
        else:
            a211_prompt, a211_resp, corrected_lab, names_lab = module_1_211_audit_lab_tests(
                row_1[COL_LAB_TESTS], seed_text, staging_system
            )
            row_1[COL_M211_INPUT]  = a211_prompt or ''
            row_1[COL_M211_OUTPUT] = a211_resp or ''
            if names_lab:
                row_1[COL_LAB_TESTS] = corrected_lab
                print(f"  ✅ 模块1.211 修正了 {len(names_lab)} 个实验室检查金标准概率: {names_lab}")
            else:
                print(f"  ✅ 模块1.211 无需修正（已全部满足或无金标准条目）")
            _save()

        # 模块1.22
        if _existing_m1_library_is_valid(
            row_1, COL_IMAGING, COL_M122_INPUT, COL_M122_OUTPUT,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.22"),
            audit_cols=(COL_M221_INPUT, COL_M221_OUTPUT),
        ):
            print(f"  ⏭️  模块1.22 已有数据，跳过")
        else:
            p122, r122 = _retry_module_call(
                module_1_22_generate_imaging,
                args=(seed_text,), kwargs={'staging_system': staging_system, **_ev_kwargs},
                module_name="模块1.22"
            )
            row_1[COL_M122_INPUT] = p122 or ''
            row_1[COL_M122_OUTPUT] = r122 or ''
            row_1[COL_IMAGING] = r122 or ''
            _save()
            print(f"  📝 模块1.22 已保存到CSV")

        # 模块1.221 查核（影像检查金标准概率）
        if row_1.get(COL_M221_OUTPUT, '').strip():
            print(f"  ⏭️  模块1.221 已有数据，跳过")
        else:
            a221_prompt, a221_resp, corrected_img, names_img = module_1_221_audit_imaging(
                row_1[COL_IMAGING], seed_text, staging_system
            )
            row_1[COL_M221_INPUT]  = a221_prompt or ''
            row_1[COL_M221_OUTPUT] = a221_resp or ''
            if names_img:
                row_1[COL_IMAGING] = corrected_img
                print(f"  ✅ 模块1.221 修正了 {len(names_img)} 个影像检查金标准概率: {names_img}")
            else:
                print(f"  ✅ 模块1.221 无需修正（已全部满足或无金标准条目）")
            _save()

        # 模块1.23
        if _existing_m1_library_is_valid(
            row_1, COL_FUNCTIONAL_TESTS, COL_M123_INPUT, COL_M123_OUTPUT,
            lambda text: validate_m1_flat_probability_schema(text, staging_system, module_name="模块1.23"),
            audit_cols=(COL_M231_INPUT, COL_M231_OUTPUT),
        ):
            print(f"  ⏭️  模块1.23 已有数据，跳过")
        else:
            p123, r123 = _retry_module_call(
                module_1_23_generate_functional_tests,
                args=(seed_text,), kwargs={'staging_system': staging_system, **_ev_kwargs},
                module_name="模块1.23"
            )
            row_1[COL_M123_INPUT] = p123 or ''
            row_1[COL_M123_OUTPUT] = r123 or ''
            row_1[COL_FUNCTIONAL_TESTS] = r123 or ''
            _save()
            print(f"  📝 模块1.23 已保存到CSV")

        # 模块1.231 查核（功能检查金标准概率）
        if row_1.get(COL_M231_OUTPUT, '').strip():
            print(f"  ⏭️  模块1.231 已有数据，跳过")
        else:
            a231_prompt, a231_resp, corrected_func, names_func = module_1_231_audit_functional(
                row_1[COL_FUNCTIONAL_TESTS], seed_text, staging_system
            )
            row_1[COL_M231_INPUT]  = a231_prompt or ''
            row_1[COL_M231_OUTPUT] = a231_resp or ''
            if names_func:
                row_1[COL_FUNCTIONAL_TESTS] = corrected_func
                print(f"  ✅ 模块1.231 修正了 {len(names_func)} 个功能检查金标准概率: {names_func}")
            else:
                print(f"  ✅ 模块1.231 无需修正（已全部满足或无金标准条目）")
            _save()

        # 模块1.5
        if row_1.get(COL_COMORBIDITIES, '').strip():
            print(f"  ⏭️  模块1.5 已有数据，跳过")
        else:
            p15, r15 = _retry_module_call(
                module_1_5_generate_comorbidities, args=(seed_text,),
                module_name="模块1.5"
            )
            row_1[COL_M15_INPUT] = p15 or ''
            row_1[COL_M15_OUTPUT] = r15 or ''
            row_1[COL_COMORBIDITIES] = r15 if r15 else '[]'
            _save()
            print(f"  📝 模块1.5 已保存到CSV")

        # 模块1.51
        if row_1.get(COL_COMPLICATION_PHENOTYPES, '').strip():
            print(f"  ⏭️  模块1.51 已有数据，跳过")
        else:
            comorbidities_raw = row_1.get(COL_COMORBIDITIES, '[]') or '[]'
            if comorbidities_raw.strip() and comorbidities_raw.strip() != '[]':
                p151, r151 = _retry_module_call(
                    module_1_51_generate_complication_phenotypes,
                    args=(seed_text, comorbidities_raw),
                    module_name="模块1.51"
                )
                row_1[COL_M151_INPUT] = p151 or ''
                row_1[COL_M151_OUTPUT] = r151 or ''
                row_1[COL_COMPLICATION_PHENOTYPES] = r151 if r151 else '[]'
                _save()
                print(f"  📝 模块1.51 已保存到CSV")
            else:
                print(f"  ⏭️  模块1.51 跳过（无伴随疾病）")
                row_1[COL_COMPLICATION_PHENOTYPES] = '[]'

        m1_elapsed = round(time.time() - t_module1, 1)
        print(f"\n  ✅ 模块1完成，概率库已保存到row_1，用时 {m1_elapsed}s")
        print(f"   保存至: {csv_path}")
        print(f"{'='*60}")

        return {'status': 'success', 'seed': seed_text, 'file': csv_path}

    except Exception as e:
        print(f"  ❌ 处理种子 [{seed_text}] 时出错: {e}")
        traceback.print_exc()
        return {'status': 'error', 'seed': seed_text, 'error': str(e)}


def _process_seed_wrapper_m1(args_tuple):
    """多进程包装函数（模块1）"""
    idx, seed_text, output_dir, enable_evidence, evidence_env, evidence_workers = args_tuple
    print(f"\n{'#'*60}")
    print(f"# [进程] 处理第 {idx+1} 个种子: {seed_text}")
    print(f"{'#'*60}")
    return process_seed_module1(
        seed_text, output_dir,
        enable_evidence=enable_evidence,
        evidence_env=evidence_env,
        evidence_workers=evidence_workers,
    )


def run_module1(seed_file, output_dir,
                enable_evidence=False, evidence_env=DEFAULT_EVIDENCE_ENV,
                evidence_workers=DEFAULT_EVIDENCE_WORKERS,
                num_seed_workers=DEFAULT_NUM_SEED_WORKERS,
                seed_index=None):
    """
    主入口：读取种子文件，并行处理所有种子的模块1。
    """
    if not os.path.exists(seed_file):
        print(f"错误：种子文件不存在: {seed_file}")
        return

    # 读取种子，同时跟踪每个种子所属科室（来自 "# 科室名（...）" 注释行）
    seeds = []          # 疾病名列表
    seed_depts = []     # 与 seeds 一一对应的科室名
    current_dept = '其他'
    with open(seed_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                # 注释行：提取科室名（取第一个 # 后、去掉 ICD 括号前的部分）
                comment = stripped.lstrip('#').strip()
                dept_match = re.match(r'^([^（(【\s][^（(【]*?)(?:[（(【]|$)', comment)
                if dept_match:
                    current_dept = dept_match.group(1).strip() or current_dept
                continue
            name = stripped.split('#')[0].strip()
            if name:
                # 保留 `#` 后的年龄范围、性别比例、急慢性和 ICD。
                # 文件名仍由 process_seed_module1 仅取 `#` 前诊断生成。
                seeds.append(stripped)
                seed_depts.append(current_dept)

    if not seeds:
        print("错误：种子文件为空")
        return

    print(f"读取到 {len(seeds)} 个患者种子")
    for i, seed in enumerate(seeds):
        print(f"  [{i}] [{seed_depts[i]}] {seed}")

    import time as _t
    timestamp = _t.strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(output_dir, f'sim_patient_V4_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🆕 新建运行：创建时间戳目录 {output_dir}")

    print(f"\n输出目录: {output_dir}")

    # 构建 (全局idx, 疾病名, 该疾病对应的科室子目录) 列表
    def _dept_dir(dept):
        safe = re.sub(r'[^\w一-鿿]', '_', dept)
        d = os.path.join(output_dir, safe)
        os.makedirs(d, exist_ok=True)
        return d

    if seed_index is not None:
        if seed_index < 0 or seed_index >= len(seeds):
            print(f"错误：种子索引 {seed_index} 超出范围 [0, {len(seeds)-1}]")
            return
        seeds_to_process = [(seed_index, seeds[seed_index], _dept_dir(seed_depts[seed_index]))]
        use_parallel = False
    else:
        seeds_to_process = [(i, seeds[i], _dept_dir(seed_depts[i])) for i in range(len(seeds))]
        use_parallel = len(seeds_to_process) > 1

    start_time = time.time()
    results = []

    if use_parallel:
        print(f"\n使用 {num_seed_workers} 个进程并行处理 {len(seeds_to_process)} 个种子")
        tasks = [
            (idx, seed_text, dept_out_dir, enable_evidence, evidence_env, evidence_workers)
            for idx, seed_text, dept_out_dir in seeds_to_process
        ]
        with mp.Pool(processes=num_seed_workers) as pool:
            results = pool.map(_process_seed_wrapper_m1, tasks)
    else:
        for idx, seed_text, dept_out_dir in seeds_to_process:
            result = process_seed_module1(
                seed_text, dept_out_dir,
                enable_evidence=enable_evidence,
                evidence_env=evidence_env,
                evidence_workers=evidence_workers,
            )
            results.append(result)

    total_time = round(time.time() - start_time, 2)
    print(f"\n{'='*60}")
    print(f"模块1全部处理完成")
    print(f"处理种子数: {len(results)}")
    print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"失败: {sum(1 for r in results if r['status'] != 'success')}")
    print(f"总耗时: {total_time} 秒")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    print(f"\n下一步：运行 atlas_based_patient_generation.py --csv_dir {output_dir}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='phenotypic_atlas.py：模块1 - 表型概率库生成'
    )
    parser.add_argument('--seed_file', type=str, default=DEFAULT_SEED_FILE)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument('--num_seed_workers', type=int, default=DEFAULT_NUM_SEED_WORKERS)
    parser.add_argument('--seed_index', type=int, default=None,
                        help='只处理指定索引的种子（从0开始）')
    parser.add_argument('--enable_evidence', action='store_true', default=False,
                        help='启用循证检索增强（默认关闭，需显式开启）')
    parser.add_argument('--evidence_env', type=str, default=DEFAULT_EVIDENCE_ENV,
                        choices=['yufa', 'online'])
    parser.add_argument('--evidence_workers', type=int, default=DEFAULT_EVIDENCE_WORKERS)
    args = parser.parse_args()

    run_module1(
        seed_file=args.seed_file,
        output_dir=args.output_dir,
        enable_evidence=args.enable_evidence,
        evidence_env=args.evidence_env,
        evidence_workers=args.evidence_workers,
        num_seed_workers=args.num_seed_workers,
        seed_index=args.seed_index,
    )


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    mp.set_start_method('spawn', force=True)
    main()
