# -*- coding: utf-8 -*-
"""
disease_demographics.py
疾病 → 成人住院人群的 (年龄区间, 男性比例) 推断 + JSON 缓存。

V5 种子只含疾病名，缺少 V4 里手工写死的"年龄/性别"字段。本模块补齐这一维度，
供 utils._parse_seed 在读取 V5 种子时按疾病查询 demographics，然后上层随机抽样。

缓存文件：patient_seed/disease_demographics.json
结构：{"<疾病名>": {"age_min": 30, "age_max": 75, "male_ratio": 0.60}}
"""

import os
import sys
import json
import time
import threading

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import call_gpt5


CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'patient_seed', 'disease_demographics.json'
)
CACHE_PATH = os.path.normpath(CACHE_PATH)

_AGE_MIN_FLOOR = 18
_AGE_MAX_CEIL = 95

_cache_lock = threading.RLock()
_cache = None

# 按疾病名粒度的锁，防止 50 路并发对同一疾病同时发起 GPT 调用（TOCTOU 竞态）
_per_disease_locks: dict = {}
_per_disease_locks_meta = threading.Lock()


def _load_cache():
    """惰性加载 JSON 缓存到内存。缺失或损坏时返回空字典。"""
    global _cache
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                    _cache = json.load(f)
                if not isinstance(_cache, dict):
                    _cache = {}
            except Exception:
                _cache = {}
        else:
            _cache = {}
    return _cache


def _atomic_write_cache(data: dict):
    """带文件锁的原子写。文件锁不可用（如 NFS 上 Errno 37）时回退 threading.Lock。"""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp_path = CACHE_PATH + '.tmp'

    def _do_write():
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CACHE_PATH)

    if _HAS_FCNTL:
        lock_path = CACHE_PATH + '.lock'
        try:
            with open(lock_path, 'w') as lf:
                # 用 LOCK_NB 避免 NFS 上永久阻塞；拿不到锁就退回 threading 锁
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    _do_write()
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            return
        except OSError:
            # NFS 不支持 flock 或锁已被占用：退回 threading 锁
            pass
    with _cache_lock:
        _do_write()


_VALID_ACUITY = {'急性', '慢性', '慢性急性加重'}


def _validate(d: dict):
    """对 LLM 返回做类型和范围校验，返回规范化后的 dict，非法时抛 ValueError。"""
    if not isinstance(d, dict):
        raise ValueError("non-dict")
    age_min = int(d.get('age_min'))
    age_max = int(d.get('age_max'))
    male_ratio = float(d.get('male_ratio'))
    acuity = str(d.get('acuity', '') or '').strip()
    if age_min < _AGE_MIN_FLOOR:
        age_min = _AGE_MIN_FLOOR
    if age_max > _AGE_MAX_CEIL:
        age_max = _AGE_MAX_CEIL
    if age_min >= age_max:
        raise ValueError(f"invalid age range: {age_min}-{age_max}")
    if not (0.0 <= male_ratio <= 1.0):
        raise ValueError(f"invalid male_ratio: {male_ratio}")
    if acuity and acuity not in _VALID_ACUITY:
        # acuity 字段不合法时，宽容地置为空字符串，由调用方决定如何兜底
        acuity = ''
    return {
        'age_min': age_min,
        'age_max': age_max,
        'male_ratio': male_ratio,
        'acuity': acuity,
    }


_PROMPT_TEMPLATE = """你是一位临床流行病学专家。请根据流行病学数据，为以下成人住院患者疾病给出典型人群画像：

疾病名：{disease}

要求：
1. 只考虑成人住院患者（年龄范围限定在 18–95 岁之间）
2. 给出该疾病住院患者的典型年龄区间 age_min / age_max
   - 区间要覆盖该病的主流发病年龄段（约 60%-80% 住院病例）
   - 不是极端最小值/最大值，而是有代表性的区间
3. 给出男性比例 male_ratio（0.0-1.0 之间的小数，0.5 表示男女各半）
4. 给出本次住院该疾病最典型的临床形式 acuity，**三选一**：
   - "急性"：本次发病为急性起病（如急性心肌梗死、急性阑尾炎、社区获得性肺炎首次发作）
   - "慢性"：本次为慢性疾病的稳定期管理或随访（如稳定期 COPD、慢性肾病随访）
   - "慢性急性加重"：在慢性病基础上本次为急性加重（如 AECOPD、慢性心衰急性失代偿、糖尿病酮症酸中毒）
   选择最常见的住院场景；只能选一个。

输出格式：只输出一个 Python 字典，不要任何解释。例如：
{{"age_min": 55, "age_max": 85, "male_ratio": 0.65, "acuity": "急性"}}
"""


def _query_llm(disease: str, tag: str = '') -> dict:
    """调用 GPT 推断 demographics，解析 + 校验。失败抛异常。"""
    import ast as _ast
    prompt = _PROMPT_TEMPLATE.format(disease=disease)
    response = call_gpt5(prompt, tag=tag or f"demographics-{disease}")
    if not response:
        raise RuntimeError("call_gpt5 returned empty")
    text = response.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"no dict braces in LLM response: {text[:200]}")
    parsed = _ast.literal_eval(text[start:end + 1])
    return _validate(parsed)


def get_demographics(disease_name: str, tag: str = '') -> dict:
    """
    返回 {'age_min', 'age_max', 'male_ratio'}。

    先查内存缓存 → JSON 文件 → LLM 推断。推断结果校验通过后写回 JSON。
    查询失败时返回保守默认值（18-80 岁，男性 0.5），保证流水线不中断。

    并发安全：对同一疾病名使用独立锁，保证同时只有 1 个线程调用 GPT，
    其余线程等锁释放后直接命中缓存，避免 50 路并发惊群打垮 API 网关。
    """
    disease_name = str(disease_name).strip()

    # 快路径：无锁先查内存缓存
    cache = _load_cache()
    if disease_name in cache:
        try:
            return _validate(cache[disease_name])
        except Exception:
            pass  # 缓存条目损坏，继续往下走

    # 拿到该疾病专属锁（不同疾病之间互不阻塞）
    with _per_disease_locks_meta:
        if disease_name not in _per_disease_locks:
            _per_disease_locks[disease_name] = threading.Lock()
    disease_lock = _per_disease_locks[disease_name]

    with disease_lock:
        # 二次检查：可能在等锁期间已被另一线程写入
        cache = _load_cache()
        if disease_name in cache:
            try:
                return _validate(cache[disease_name])
            except Exception:
                pass

        try:
            result = _query_llm(disease_name, tag=tag)
        except Exception as e:
            print(f"  [demographics] ⚠️ 推断失败 \"{disease_name}\": {e}，使用默认值")
            return {'age_min': 18, 'age_max': 80, 'male_ratio': 0.5, 'acuity': ''}

        with _cache_lock:
            cache[disease_name] = result
            _atomic_write_cache(cache)
        print(f"  [demographics] 缓存新增 \"{disease_name}\" → {result}")
        return result


def sample_age_gender(age_min: int, age_max: int, male_ratio: float,
                      rng=None, **_ignored) -> tuple:
    """按区间均匀抽年龄、按 Bernoulli(male_ratio) 抽性别。返回 (age:int, gender:str)。
    额外的 kwargs（如 acuity）会被忽略，方便用 `**get_demographics(...)` 传参。"""
    import random as _random
    r = rng if rng is not None else _random
    age = r.randint(int(age_min), int(age_max))
    gender = '男' if r.random() < float(male_ratio) else '女'
    return age, gender


if __name__ == '__main__':
    # 简单 smoke test
    import argparse
    parser = argparse.ArgumentParser(description='查询单个疾病的 demographics')
    parser.add_argument('disease', nargs='?', default='社区获得性肺炎')
    args = parser.parse_args()
    result = get_demographics(args.disease)
    print(f"{args.disease}: {result}")
    a, g = sample_age_gender(**result)
    print(f"  抽样示例: {a}岁 {g}")
