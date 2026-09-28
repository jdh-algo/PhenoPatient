# -*- coding: utf-8 -*-
"""
utils.py
公共基础设施：常量、API封装、GPT调用、CSV列定义、I/O辅助函数

其他模块均从此文件导入所需工具。
"""

import os
import ast
import re
import time
import random
import traceback
import csv as _csv_module
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import math
import requests
import pandas as pd

# ============================================================
# 默认参数
# ============================================================
DEFAULT_MODEL_NAME = os.environ.get("PHENOPATIENT_MODEL_NAME", "GPT-5.5")
DEFAULT_LLM_URL = os.environ.get("PHENOPATIENT_LLM_URL", "")
DEFAULT_API_KEY = os.environ.get("PHENOPATIENT_API_KEY", "")
DEFAULT_MAX_TOKENS = 8192
DEFAULT_API_TIMEOUT_SECONDS = 600
DEFAULT_MAX_RETRIES = 100          # call_gpt5 内重试次数（"失败一直试到成功"）
DEFAULT_RETRY_SLEEP_SECONDS = 5
DEFAULT_REQUEST_DELAY = 0.5

# LLM 调用节流变量（import threading 供节流锁使用）
import os as _os_for_env
import threading as _threading

# GPT-5.5 节流：保证连续两次 POST 间隔不小于此值（秒），可由环境变量覆盖
_LLM_MIN_INTERVAL = float(_os_for_env.environ.get('LLM_MIN_INTERVAL', '0.55'))
_LLM_PACE_LOCK = _threading.Lock()
_LLM_LAST_CALL_TS = [0.0]

# 表型图谱默认也使用 GPT-5.5；保留 call_gpt54 函数名兼容旧模块 import。
_GPT54_MODEL_NAME = os.environ.get("PHENOPATIENT_ATLAS_MODEL_NAME", DEFAULT_MODEL_NAME)
_GPT54_MIN_INTERVAL = float(_os_for_env.environ.get('GPT54_MIN_INTERVAL', '0.55'))
_GPT54_PACE_LOCK = _threading.Lock()
_GPT54_LAST_CALL_TS = [0.0]

DEFAULT_MODULE_MAX_RETRIES = 100   # 模块级重试次数（"失败一直试到成功"）
DEFAULT_MODULE_RETRY_SLEEP = 15

DEFAULT_SEED_FILE = os.environ.get("PHENOPATIENT_SEED_FILE", "")
DEFAULT_OUTPUT_BASE = os.environ.get("PHENOPATIENT_OUTPUT_ROOT", "")
DEFAULT_NUM_PATIENTS_PER_SEED = 5
DEFAULT_NUM_SEED_WORKERS = 5
DEFAULT_CORRELATION_CO_OCCUR_PROB = 0.85
DEFAULT_MAX_INTERACTION_TURNS = 15
DEFAULT_MAX_FOLLOWUP_TURNS = 0

DEFAULT_EVIDENCE_ENV = 'yufa'
DEFAULT_EVIDENCE_WORKERS = 5
DEFAULT_EVIDENCE_TIMEOUT = 300
DEFAULT_EVIDENCE_MAX_RETRIES = 5
DEFAULT_EVIDENCE_CALL_MAX_RETRIES = 5
DEFAULT_EVIDENCE_CALL_RETRY_SLEEP = 3
DEFAULT_EVIDENCE_MIN_RESP_LEN = 500


# ============================================================
# 循证检索接口封装（来自 ZhiyiXunzhengApiLib）
# ============================================================

class ZhiyiXunzhengApiLib:
    """知医循证接口封装"""

    def __init__(self, api_config: dict):
        self.api_config = api_config
        self.env = api_config.get("env", "yufa")
        self.time_out = api_config.get("time_out", 300)
        self.max_retry_num = api_config.get("max_retry_num", 3)
        self.retry_sleep_time = api_config.get("retry_sleep_time", 3)
        self.stream = api_config.get("stream", False)
        self.fallback_stream_if_empty = api_config.get("fallback_stream_if_empty", True)

        self.chat_create_url = os.environ.get(
            "PHENOPATIENT_EVIDENCE_CHAT_CREATE_URL", ""
        )
        self.router_rest_url = os.environ.get(
            "PHENOPATIENT_EVIDENCE_ROUTER_URL", ""
        )
        if not self.chat_create_url or not self.router_rest_url:
            raise RuntimeError(
                "缺少循证接口地址：请设置 "
                "PHENOPATIENT_EVIDENCE_CHAT_CREATE_URL 和 "
                "PHENOPATIENT_EVIDENCE_ROUTER_URL"
            )

    def _evidence_app_key(self) -> str:
        app_key = os.environ.get("PHENOPATIENT_EVIDENCE_APP_KEY", "")
        if not app_key:
            raise RuntimeError("缺少循证接口 app_key：请设置 PHENOPATIENT_EVIDENCE_APP_KEY 环境变量")
        return app_key

    def _get_chat_code(self, query: str) -> str:
        params = {
            "app_key": self._evidence_app_key(),
            "pin": os.environ.get("PHENOPATIENT_EVIDENCE_PIN", ""),
            "scene_code": os.environ.get("PHENOPATIENT_EVIDENCE_CHAT_SCENE", ""),
            "session_id": str(uuid.uuid4()),
            "chat_ttl": 3600,
            "message_history": [{"role": "user", "content": query}],
        }
        resp = requests.post(self.chat_create_url, json=params, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        chat_code = result.get("data", {}).get("chat_code")
        if not chat_code:
            raise RuntimeError(f"获取 chat_code 失败: {result}")
        return chat_code

    def _call_router_rest(self, chat_code: str, query: str, stream: bool) -> str:
        req_params = {
            "model": os.environ.get("PHENOPATIENT_EVIDENCE_MODEL", ""),
            "app_key": self._evidence_app_key(),
            "method": "jdh.chatbot.conversational",
            "scene_code": os.environ.get("PHENOPATIENT_EVIDENCE_ROUTER_SCENE", ""),
            "chat_code": chat_code,
            "messages": [{"role": "user", "content": query}],
            "stream": stream,
        }
        accept = "text/event-stream" if stream else "application/json"
        headers = {"Content-Type": "application/json", "Accept": accept}
        resp = requests.post(
            self.router_rest_url,
            json=req_params,
            headers=headers,
            timeout=self.time_out,
        )
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _parse_sse(raw_text: str):
        events = []
        for part in raw_text.split("data:"):
            part = part.strip()
            if not part or part == "[DONE]":
                continue
            try:
                events.append(json.loads(part))
            except json.JSONDecodeError:
                pass
        full_text, attachments = [], []
        for ev in events:
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                c = delta.get("content")
                if c:
                    full_text.append(c)
                att = delta.get("attachment")
                if att:
                    attachments.extend(att)
        return "".join(full_text), attachments, len(events)

    @staticmethod
    def _parse_json(raw_text: str):
        data = json.loads(raw_text)
        payload = data
        for key in ("data", "result"):
            inner = data.get(key)
            if isinstance(inner, dict) and "choices" in inner:
                payload = inner
                break
        full_text, attachments = [], []
        for ch in payload.get("choices", []):
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or ch.get("delta") or {}
            c = msg.get("content")
            if isinstance(c, str) and c:
                full_text.append(c)
            for att_key in ("attachment", "attachments"):
                att = msg.get(att_key)
                if isinstance(att, list):
                    attachments.extend(att)
                elif att:
                    attachments.append(att)
        return "".join(full_text), attachments, len(payload.get("choices", []))

    def _parse_response(self, raw_body: str, stream: bool):
        if stream:
            return self._parse_sse(raw_body)
        try:
            return self._parse_json(raw_body)
        except json.JSONDecodeError:
            return self._parse_sse(raw_body)

    @staticmethod
    def _filter_attachments(attachments: list) -> list:
        keep_keys = ["author", "chunk_content", "doc_name", "doc_type",
                      "doc_url", "doi", "pub_date", "pub_org"]
        selected = []
        for item in attachments:
            for item_content in (item.get("content") or []):
                if item_content is None:
                    continue
                for piece in item_content:
                    if "chunk_content" in piece:
                        selected.append(item_content)
        filtered = [{k: v for k, v in a.items() if k in keep_keys} for a in selected]
        seen, unique = set(), []
        for a in filtered:
            c = a.get("chunk_content")
            if c is not None and c not in seen:
                unique.append(a)
                seen.add(c)
        return unique

    MIN_RESP_TEXT_LEN = 200

    def _is_response_abnormal(self, text: str, filtered: list, raw_body: str):
        if not (text or "").strip() and not filtered:
            return "正文和文献均为空"
        if "ReadTimeout" in raw_body:
            if len((text or "").strip()) < self.MIN_RESP_TEXT_LEN:
                return f"Model_client ReadTimeout 且正文过短({len((text or '').strip())}字)"
        if len((text or "").strip()) < self.MIN_RESP_TEXT_LEN:
            return f"正文过短({len((text or '').strip())}字 < {self.MIN_RESP_TEXT_LEN})"
        return None

    def req_xunzheng(self, query: str):
        err_msgs = []
        retry_num = 0
        best_resp_obj = None
        best_text_len = -1
        start_time = time.time()

        for attempt in range(1, self.max_retry_num + 1):
            try:
                chat_code = self._get_chat_code(query)
                router_calls = []
                raw_body = self._call_router_rest(chat_code, query, self.stream)
                router_calls.append({"stream": self.stream, "raw_body": raw_body})
                text, attachments, event_count = self._parse_response(raw_body, self.stream)

                if (self.fallback_stream_if_empty
                        and not self.stream
                        and not (text or "").strip()
                        and not attachments):
                    raw_body = self._call_router_rest(chat_code, query, True)
                    router_calls.append({"stream": True, "raw_body": raw_body})
                    text, attachments, event_count = self._parse_response(raw_body, True)

                filtered = self._filter_attachments(attachments)
                last_raw_body = router_calls[-1]["raw_body"]
                resp_obj = {
                    "chat_code": chat_code,
                    "resp_text": text,
                    "ref_num": len(filtered),
                    "citation_info": filtered,
                    "raw_attachment_count": len(attachments),
                    "event_count": event_count,
                    "router_calls": router_calls,
                    "retry_attempts": attempt,
                }
                cur_text_len = len((text or "").strip())
                if cur_text_len > best_text_len:
                    best_text_len = cur_text_len
                    best_resp_obj = resp_obj

                abnormal_reason = self._is_response_abnormal(text, filtered, last_raw_body)
                if abnormal_reason is None:
                    used_time = time.time() - start_time
                    return 0, resp_obj, retry_num, used_time

                err_msgs.append(f"attempt {attempt}: {abnormal_reason}")
                if attempt < self.max_retry_num:
                    retry_num += 1
                    time.sleep(self.retry_sleep_time)

            except Exception as e:
                err_msgs.append(f"attempt {attempt}: {type(e).__name__}: {e}")
                if attempt < self.max_retry_num:
                    retry_num += 1
                    time.sleep(self.retry_sleep_time)

        used_time = time.time() - start_time
        if best_resp_obj is not None:
            best_resp_obj["retry_errors"] = err_msgs
            return 0, best_resp_obj, retry_num, used_time
        return -1, {"errors": err_msgs}, retry_num, used_time


# ============================================================
# 循证日志记录器
# ============================================================

class EvidenceLogger:
    """记录所有循证接口调用的输入与输出到CSV"""

    CSV_HEADER = ['时间戳', '模块名', '查询内容', '返回文本长度', '返回文本', '原始返回内容']

    def __init__(self, output_dir, seed_name):
        log_dir = os.path.join(output_dir, '循证调用记录')
        os.makedirs(log_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', seed_name)
        self.csv_path = os.path.join(log_dir, f'{safe_name}_循证调用记录.csv')
        self._lock = __import__('threading').Lock()
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = _csv_module.writer(f)
                writer.writerow(self.CSV_HEADER)

    def log(self, query, resp_text, raw_json_str, module_name=''):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        resp_len = len((resp_text or '').strip())
        row = [ts, module_name, query, resp_len, resp_text or '', raw_json_str or '']
        with self._lock:
            with open(self.csv_path, 'a', encoding='utf-8-sig', newline='') as f:
                writer = _csv_module.writer(f)
                writer.writerow(row)


# ============================================================
# 循证接口调用封装
# ============================================================

_evidence_api_instance = None
_evidence_api_lock = __import__('threading').Lock()


def _get_evidence_api(env='yufa', timeout=300, max_retries=5):
    global _evidence_api_instance
    with _evidence_api_lock:
        if _evidence_api_instance is None:
            _evidence_api_instance = ZhiyiXunzhengApiLib({
                "env": env,
                "time_out": timeout,
                "max_retry_num": max_retries,
                "retry_sleep_time": 3,
                "stream": False,
                "fallback_stream_if_empty": True,
            })
    return _evidence_api_instance


def call_evidence_api(query, evidence_logger=None, tag='', env='yufa',
                      max_retries=DEFAULT_EVIDENCE_CALL_MAX_RETRIES,
                      min_resp_len=DEFAULT_EVIDENCE_MIN_RESP_LEN):
    """调用循证接口，返回 resp_text。

    若返回内容为空或不足 min_resp_len 字符，自动重试最多 max_retries 次。
    全部重试仍不足则返回空字符串，由上层退化为纯GPT生成。
    """
    api = _get_evidence_api(env=env)
    best_text = ''
    best_len = 0
    total_t0 = time.time()

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            err_code, resp_obj, retry_num, used_time = api.req_xunzheng(query)
            resp_text = resp_obj.get('resp_text', '') if isinstance(resp_obj, dict) else ''
            raw_json_str = json.dumps(resp_obj, ensure_ascii=False) if isinstance(resp_obj, dict) else str(resp_obj)
            cur_len = len((resp_text or '').strip())

            if evidence_logger:
                attempt_tag = f"{tag}(attempt{attempt}/{max_retries})" if attempt > 1 else tag
                evidence_logger.log(query, resp_text, raw_json_str, module_name=attempt_tag)

            if cur_len > best_len:
                best_text = resp_text
                best_len = cur_len

            if err_code == 0 and cur_len >= min_resp_len:
                print(f"    [{tag}] 循证检索成功: '{query[:40]}...' → {cur_len}字 ({round(used_time,1)}s)")
                return resp_text

            reason = f"err={err_code}" if err_code != 0 else f"内容过短({cur_len}<{min_resp_len}字)"
            if attempt < max_retries:
                print(f"    [{tag}] 循证检索第{attempt}次不足({reason})，{DEFAULT_EVIDENCE_CALL_RETRY_SLEEP}s后重试...")
                time.sleep(DEFAULT_EVIDENCE_CALL_RETRY_SLEEP)
            else:
                print(f"    [{tag}] 循证检索第{attempt}次仍不足({reason})，已达最大重试次数")

        except Exception as e:
            if evidence_logger:
                attempt_tag = f"{tag}(attempt{attempt}/{max_retries})" if attempt > 1 else tag
                evidence_logger.log(query, '', f'ERROR: {type(e).__name__}: {e}', module_name=attempt_tag)
            if attempt < max_retries:
                print(f"    [{tag}] 循证检索第{attempt}次异常({type(e).__name__})，{DEFAULT_EVIDENCE_CALL_RETRY_SLEEP}s后重试...")
                time.sleep(DEFAULT_EVIDENCE_CALL_RETRY_SLEEP)
            else:
                print(f"    [{tag}] 循证检索第{attempt}次异常({type(e).__name__})，已达最大重试次数")

    total_elapsed = round(time.time() - total_t0, 1)
    print(f"    [{tag}] 循证检索{max_retries}次均未获得有效结果(最佳{best_len}字)，"
          f"将跳过循证直接GPT生成 ({total_elapsed}s)")
    return ''


def batch_evidence_queries(queries, evidence_logger=None, tag='', env='yufa', max_workers=5):
    """并发调用循证接口，返回 {query: resp_text} 字典。"""
    results = {}
    if not queries:
        return results

    def _single(q):
        return q, call_evidence_api(q, evidence_logger=evidence_logger, tag=tag, env=env)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_single, q): q for q in queries}
        for f in as_completed(futures):
            try:
                q, resp = f.result()
                results[q] = resp
            except Exception as e:
                q = futures[f]
                results[q] = ''
                print(f"    [{tag}] 并发循证异常: '{q[:30]}...' → {e}")
    return results


# ============================================================
# CSV 列名定义
# ============================================================
COL_CASE_ID = 'case_id'
COL_SEED = 'seed'
COL_STAGING_SYSTEM = '分级系统'
COL_DIFFERENTIAL_DIAGNOSIS = '鉴别诊断'
COL_AGE = '年龄'
COL_GENDER = '性别'
COL_DIAGNOSIS = '诊断'
COL_STAGE = '时期'
COL_SYMPTOMS = '症状'
COL_SIGNS = '体征'
COL_LAB_TESTS = '实验室检查'
COL_IMAGING = '影像检查'
COL_FUNCTIONAL_TESTS = '功能检查'
COL_KEY_LAB_TESTS = '该诊断重要化验检查'
COL_DIFF_KEY_LAB_TESTS = '鉴别诊断重要化验检查'
COL_TIME_ORDER = '时间顺序'
COL_COMORBIDITIES = '伴随疾病'
COL_TREATMENT_FACTORS = '治疗影响因素'
COL_QUANTIFIED = '范围数值化'
COL_SPECIFIC = '具体数值化'
COL_INTERACTION = 'v4模拟患者_交互历史'
COL_TREATMENT = 'v4模拟患者_治疗方案'
COL_PATIENT_ADR = '不良反应'
COL_TREATMENT_OUTCOME = '治疗效果'

COL_STANDARD_TREATMENT = '标准治疗'
COL_ADR_LIBRARY = '不良反应库'
COL_M10_INPUT = '模块1.0_输入'
COL_M10_OUTPUT = '模块1.0_输出'
COL_M101_INPUT = '模块1.01_输入'
COL_M101_OUTPUT = '模块1.01_输出'
COL_M111_INPUT = '模块1.11_输入'
COL_M111_OUTPUT = '模块1.11_输出'
COL_M112_INPUT = '模块1.12_输入'
COL_M112_OUTPUT = '模块1.12_输出'
COL_M121_INPUT = '模块1.21_输入'
COL_M121_OUTPUT = '模块1.21_输出'
COL_M122_INPUT = '模块1.22_输入'
COL_M122_OUTPUT = '模块1.22_输出'
COL_M123_INPUT = '模块1.23_输入'
COL_M123_OUTPUT = '模块1.23_输出'
COL_M211_INPUT  = '模块1.211_输入'
COL_M211_OUTPUT = '模块1.211_输出'
COL_M221_INPUT  = '模块1.221_输入'
COL_M221_OUTPUT = '模块1.221_输出'
COL_M231_INPUT  = '模块1.231_输入'
COL_M231_OUTPUT = '模块1.231_输出'
COL_M124_INPUT = '模块1.24_输入'
COL_M124_OUTPUT = '模块1.24_输出'
COL_M125_INPUT = '模块1.25_输入'
COL_M125_OUTPUT = '模块1.25_输出'
COL_M131_INPUT = '模块1.31_输入'
COL_M131_OUTPUT = '模块1.31_输出'
COL_M132_INPUT = '模块1.32_输入'
COL_M132_OUTPUT = '模块1.32_输出'
COL_M15_INPUT = '模块1.5_输入'
COL_M15_OUTPUT = '模块1.5_输出'
COL_M151_INPUT = '模块1.51_输入'
COL_M151_OUTPUT = '模块1.51_输出'
COL_M16_INPUT = '模块1.6_输入'
COL_M16_OUTPUT = '模块1.6_输出'
COL_M23_INPUT = '模块2.3_输入'
COL_M23_OUTPUT = '模块2.3_输出'
COL_M24_INPUT = '模块2.4_输入'
COL_M24_OUTPUT = '模块2.4_输出'
COL_M25_INPUT = '模块2.5_输入'
COL_M25_OUTPUT = '模块2.5_输出'
COL_M26_INPUT = '模块2.6_输入'
COL_M26_OUTPUT = '模块2.6_输出'
COL_M27_INPUT = '模块2.7_输入'
COL_M27_OUTPUT = '模块2.7_输出'
COL_CHIEF_COMPLAINT = '主诉'
COL_M41_INPUT = '模块4.1_输入'
COL_M41_OUTPUT = '模块4.1_输出'
COL_M42_INPUT = '模块4.2_输入'
COL_M42_OUTPUT = '模块4.2_输出'

# M5 基线对比模块（2026-05-15 新增）：GPT 直接扮演患者，与同款医生交互
COL_M5_INTERACTION = 'GPT直演患者_交互历史'
COL_M5_TREATMENT = 'GPT直演患者_治疗方案'
COL_M5_EHR = 'GPT直演患者_病案'
COL_M5_INPUT = '模块5_输入'
COL_M5_OUTPUT = '模块5_输出'

COL_M102_INPUT = '模块1.02_输入'
COL_M102_OUTPUT = '模块1.02_输出'
COL_LATERALITY_TYPE = '侧别类型'
COL_PATIENT_LATERALITY = '患者侧别'
COL_COMPLICATION_PHENOTYPES = '并发症表型库'

# ============================================================
# v4 新增列
# ============================================================
# row_1（概率库行）
COL_DIAGNOSIS_COMPONENTS = '诊断子疾病'           # 多疾病诊断的拆分结果，如 ["糖尿病","糖尿病肾病"]
COL_KEY_SYMPTOMS = '该诊断重要症状'                # 1.25 新语义：诊断关键症状
COL_DIFF_KEY_SYMPTOMS = '鉴别诊断重要症状'         # 1.25 新语义：鉴别诊断关键症状
COL_KEY_SIGNS = '该诊断重要体征'                   # 1.26
COL_DIFF_KEY_SIGNS = '鉴别诊断重要体征'             # 1.26
COL_M126_INPUT = '模块1.26_输入'
COL_M126_OUTPUT = '模块1.26_输出'

# row_2+（患者行）
COL_ACUITY = '急慢性'                              # 急性 / 慢性 / 慢性急性加重
COL_PRIOR_VISITED = '此前就诊状态'                 # 未就诊 / 就诊过
COL_PRIOR_VISIT_COUNT = '此前就诊次数'             # 0~3
COL_DURATION_TOTAL = '患病总时长'                  # 如 "3年" / "5天" / "6小时"
COL_PRIOR_VISIT_HISTORY = '既往就诊经历'           # 2.6 输出文本

COL_ABSENT_SYMPTOMS = '该患者无_症状'
COL_ABSENT_SIGNS = '该患者无_体征'
COL_ABSENT_LAB_TESTS = '该患者无_实验室检查'
COL_ABSENT_IMAGING = '该患者无_影像检查'
COL_ABSENT_FUNCTIONAL = '该患者无_功能检查'


_MIRROR_ANATOMY_TERMS = (
    '内脏反位', '镜像', '镜面', '反位',
    '左位阑尾', '左侧阑尾', '左位胆囊', '左侧胆囊',
)


def fixed_anatomic_laterality(context):
    text = str(context or '').strip()
    if not text or any(term in text for term in _MIRROR_ANATOMY_TERMS):
        return None
    if '阑尾炎' in text or '胆囊炎' in text:
        return '右侧'
    return None

CSV_COLUMNS = [
    COL_CASE_ID,
    COL_SEED, COL_STAGING_SYSTEM,
    COL_LATERALITY_TYPE,
    COL_AGE, COL_GENDER, COL_DIAGNOSIS, COL_STAGE, COL_PATIENT_LATERALITY,
    COL_SYMPTOMS, COL_SIGNS,
    COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS,
    COL_DIAGNOSIS_COMPONENTS,
    COL_TIME_ORDER, COL_COMORBIDITIES, COL_COMPLICATION_PHENOTYPES,
    COL_QUANTIFIED, COL_SPECIFIC,
    COL_CHIEF_COMPLAINT,
    COL_ACUITY, COL_PRIOR_VISITED, COL_PRIOR_VISIT_COUNT,
    COL_DURATION_TOTAL, COL_PRIOR_VISIT_HISTORY,
    COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS,
    COL_ABSENT_LAB_TESTS, COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL,
    COL_INTERACTION,
    COL_M10_INPUT, COL_M10_OUTPUT,
    COL_M102_INPUT, COL_M102_OUTPUT,
    COL_M111_INPUT, COL_M111_OUTPUT,
    COL_M112_INPUT, COL_M112_OUTPUT,
    COL_M121_INPUT, COL_M121_OUTPUT,
    COL_M122_INPUT, COL_M122_OUTPUT,
    COL_M123_INPUT, COL_M123_OUTPUT,
    COL_M211_INPUT, COL_M211_OUTPUT,
    COL_M221_INPUT, COL_M221_OUTPUT,
    COL_M231_INPUT, COL_M231_OUTPUT,
    COL_M15_INPUT, COL_M15_OUTPUT,
    COL_M151_INPUT, COL_M151_OUTPUT,
    COL_M23_INPUT, COL_M23_OUTPUT,
    COL_M24_INPUT, COL_M24_OUTPUT,
    COL_M25_INPUT, COL_M25_OUTPUT,
    COL_M26_INPUT, COL_M26_OUTPUT,
    COL_M27_INPUT, COL_M27_OUTPUT,
    COL_M5_INTERACTION, COL_M5_TREATMENT, COL_M5_EHR,
    COL_M5_INPUT, COL_M5_OUTPUT,
]

_JSON_DICT_COLUMNS = {
    COL_STAGING_SYSTEM,
}

_MODULE_IO_COLUMNS = {
    COL_M10_INPUT, COL_M10_OUTPUT,
    COL_M102_INPUT, COL_M102_OUTPUT,
    COL_M111_INPUT, COL_M111_OUTPUT,
    COL_M112_INPUT, COL_M112_OUTPUT,
    COL_M121_INPUT, COL_M121_OUTPUT,
    COL_M122_INPUT, COL_M122_OUTPUT,
    COL_M123_INPUT, COL_M123_OUTPUT,
    COL_M211_INPUT, COL_M211_OUTPUT,
    COL_M221_INPUT, COL_M221_OUTPUT,
    COL_M231_INPUT, COL_M231_OUTPUT,
    COL_M15_INPUT, COL_M15_OUTPUT,
    COL_M151_INPUT, COL_M151_OUTPUT,
    COL_M23_INPUT, COL_M23_OUTPUT,
    COL_M24_INPUT, COL_M24_OUTPUT,
    COL_M25_INPUT, COL_M25_OUTPUT,
    COL_M26_INPUT, COL_M26_OUTPUT,
    COL_M27_INPUT, COL_M27_OUTPUT,
    COL_M5_INPUT, COL_M5_OUTPUT,
}

_MODULE_IO_POINTER_PREFIX = '@module_io/'

_LIST_COLUMNS = {
    COL_SYMPTOMS, COL_SIGNS,
    COL_LAB_TESTS, COL_IMAGING, COL_FUNCTIONAL_TESTS,
    COL_DIAGNOSIS_COMPONENTS,
    COL_ABSENT_SYMPTOMS, COL_ABSENT_SIGNS,
    COL_ABSENT_LAB_TESTS, COL_ABSENT_IMAGING, COL_ABSENT_FUNCTIONAL,
    COL_TIME_ORDER,
    COL_COMORBIDITIES, COL_COMPLICATION_PHENOTYPES,
    COL_QUANTIFIED, COL_SPECIFIC,
    COL_M111_OUTPUT, COL_M112_OUTPUT,
    COL_M121_OUTPUT, COL_M122_OUTPUT, COL_M123_OUTPUT,
    COL_M15_OUTPUT,
    COL_M24_OUTPUT, COL_M25_OUTPUT, COL_M27_OUTPUT,
}


def _empty_row():
    """返回一个所有列值为空字符串的字典"""
    return {col: '' for col in CSV_COLUMNS}


# ============================================================
# 模块2.8 辅助函数：seed解析 & 有界正态采样
# ============================================================

_NUMERIC_TOKEN_PATTERN = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_PARAM_RE = re.compile(
    rf'\{{({_NUMERIC_TOKEN_PATTERN}),({_NUMERIC_TOKEN_PATTERN}),(nan|{_NUMERIC_TOKEN_PATTERN}),(nan|{_NUMERIC_TOKEN_PATTERN})\}}',
    re.IGNORECASE,
)


def _fmt_num(v: float) -> str:
    """将浮点数格式化为最多2位小数、无多余尾零的字符串。"""
    rounded = round(v, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f'{rounded:.2f}'.rstrip('0')


def _fmt_num_ctx(v: float, decimals: int) -> str:
    """按指定精度格式化浮点数，除体温外去除无意义的尾零。"""
    if decimals == 0:
        return str(int(round(v, 0)))
    if decimals == 1:
        return f'{round(v, 1):.1f}'
    return f'{round(v, decimals):.{decimals}f}'.rstrip('0').rstrip('.')


_INTEGER_VITALS_KW = ('心率', '脉率', '心跳', '呼吸频率', '呼吸次数', 'SpO2', '血氧饱和度',
                      '收缩压', '舒张压', '血压', 'SBP', 'DBP')
_ONE_DECIMAL_VITALS_KW = ('体温',)


def _sample_bounded_normal(mean: float, std: float,
                            val_min, val_max,
                            max_iter: int = 10000) -> float:
    """
    从有界正态分布 N(mean, std²) 中采样，重试直到落在 [val_min, val_max] 区间内。
    val_min / val_max 为 None 时表示无对应边界。
    """
    if std <= 0:
        return mean
    for _ in range(max_iter):
        v = random.gauss(mean, std)
        if val_min is not None and v < val_min:
            continue
        if val_max is not None and v > val_max:
            continue
        return v
    clipped = mean
    if val_min is not None:
        clipped = max(clipped, val_min)
    if val_max is not None:
        clipped = min(clipped, val_max)
    return clipped


def _quantize_values(text: str) -> str:
    """将文本中所有形如 {mean,sd,min,max} 的参数块替换为有界正态采样的具体数值。"""
    if not text or not text.strip():
        return text

    def _replace(m: re.Match) -> str:
        mean_s, std_s, min_s, max_s = m.group(1), m.group(2), m.group(3), m.group(4)
        mean = float(mean_s)
        std  = float(std_s)
        val_min = None if min_s.lower() == 'nan' else float(min_s)
        val_max = None if max_s.lower() == 'nan' else float(max_s)
        sampled = _sample_bounded_normal(mean, std, val_min, val_max)
        label = text[max(0, m.start() - 20):m.start()]
        if any(kw in label for kw in _INTEGER_VITALS_KW):
            precision = 0
        elif any(kw in label for kw in _ONE_DECIMAL_VITALS_KW):
            precision = 1
        else:
            precision = 2

        scale_candidates = [
            abs(value) for value in (mean, std, val_min, val_max)
            if value is not None and value != 0
        ]
        if val_min is not None and val_max is not None and val_max > val_min:
            scale_candidates.append(val_max - val_min)
        smallest_scale = min(scale_candidates, default=1.0)
        if smallest_scale < 1:
            precision = max(
                precision,
                min(14, int(math.ceil(-math.log10(smallest_scale))) + 2),
            )

        # 格式化不得把已在边界内的样本四舍五入到边界外。
        while precision < 15:
            rounded = round(sampled, precision)
            if (val_min is None or rounded >= val_min) and \
                    (val_max is None or rounded <= val_max):
                break
            precision += 1
        return f'；实际值={_fmt_num_ctx(sampled, precision)}'

    return _PARAM_RE.sub(_replace, str(text))


_DEMO_EXIST_RE = re.compile(
    r'^(.+?)\s*#\s*(\d+)-(\d+)\s+(\d+\.\d+)(?:\s+(急性|慢性|慢性急性加重))?\b'
)

# acuity 字段的合法值
ACUITY_ACUTE = '急性'
ACUITY_CHRONIC = '慢性'
ACUITY_CHRONIC_EXACERBATION = '慢性急性加重'
_ACUITY_VALID = {ACUITY_ACUTE, ACUITY_CHRONIC, ACUITY_CHRONIC_EXACERBATION}


def _parse_seed(seed_str: str):
    """
    解析 seed 行。支持两种格式：

    1. _demo_exist 格式（demographics 已内嵌）：
         疾病名  # age_min-age_max male_ratio [acuity]  [ICD_code]
         例如: 普通感冒  # 30-70 0.50 急性  J00
       直接解析，不调用 GPT。

    2. 普通 V5 格式：
         疾病名  # ICD_code
       调用 disease_demographics.get_demographics 查询（先查 JSON 缓存，缓存
       未命中才调用 GPT），同时返回 acuity 字段。

    Returns:
        (age_min: int, age_max: int, gender: str, diagnosis: str, acuity: str)
        acuity ∈ {急性, 慢性, 慢性急性加重}；无法确定时回退为 '急性'。
    """
    raw = str(seed_str).strip()

    # 空行/纯注释
    if not raw or raw.startswith('#'):
        return 18, 80, random.choice(['男', '女']), '', ACUITY_ACUTE

    # 尝试匹配 _demo_exist 格式
    m = _DEMO_EXIST_RE.match(raw)
    if m:
        diagnosis  = m.group(1).strip()
        age_min    = int(m.group(2))
        age_max    = int(m.group(3))
        male_ratio = float(m.group(4))
        acuity     = (m.group(5) or '').strip()
        if acuity not in _ACUITY_VALID:
            # demo_exist 未带 acuity 字段时尝试从缓存补
            try:
                from disease_demographics import get_demographics
                demo = get_demographics(diagnosis, tag=f"seed-{diagnosis}")
                acuity = str(demo.get('acuity', '') or '').strip()
            except Exception:
                acuity = ''
            if acuity not in _ACUITY_VALID:
                acuity = ACUITY_ACUTE
        gender = '男' if random.random() < male_ratio else '女'
        return age_min, age_max, gender, diagnosis, acuity

    # 普通格式：疾病名 # ICD_code
    from disease_demographics import get_demographics
    diagnosis = raw.split('#', 1)[0].strip()
    if not diagnosis:
        return 18, 80, random.choice(['男', '女']), '', ACUITY_ACUTE

    demo = get_demographics(diagnosis, tag=f"seed-{diagnosis}")
    age_min    = int(demo['age_min'])
    age_max    = int(demo['age_max'])
    male_ratio = float(demo['male_ratio'])
    acuity     = str(demo.get('acuity', '') or '').strip()
    if acuity not in _ACUITY_VALID:
        acuity = ACUITY_ACUTE
    gender = '男' if random.random() < male_ratio else '女'
    return age_min, age_max, gender, diagnosis, acuity


# ============================================================
# GPT-5 调用封装
# ============================================================

def _llm_pace(pace_lock, last_ts, min_interval):
    """全局发起节流：保证连续两次 POST 间隔不小于 min_interval 秒。"""
    with pace_lock:
        now = time.time()
        wait = last_ts[0] + min_interval - now
        if wait > 0:
            time.sleep(wait)
        last_ts[0] = time.time()


def _call_llm_stream_once(prompt, model, url, api_key, max_tokens, timeout):
    """单次 HTTP 流式调用，返回清洗后的文本，失败抛异常。"""
    if not api_key:
        raise RuntimeError('缺少模型 API key：请设置 PHENOPATIENT_API_KEY 环境变量')
    if not url:
        raise RuntimeError('缺少模型 API 地址：请设置 PHENOPATIENT_LLM_URL 环境变量')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'stream': True,
    }
    resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:300]}')
    result_text = ''
    for line in resp.iter_lines():
        line = line.decode('utf-8', errors='ignore')
        if 'data: [DONE]' in line:
            break
        if not line.startswith('data: '):
            continue
        try:
            data = json.loads(line[6:])
            if data.get('choices'):
                delta = data['choices'][0].get('delta', {}).get('content', '')
                if delta:
                    result_text += delta
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    cleaned = result_text.strip()
    cleaned = re.sub(r'^```[a-z]*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```$', '', cleaned)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    return cleaned


def _is_content_filter_error(message):
    text = str(message).casefold()
    return "response was filtered" in text or "content management policy" in text


def _rephrase_content_filter_prompt(prompt):
    safe_prompt = str(prompt)
    replacements = (
        ("吐出来的东西里带血", "呕吐物颜色异常"),
        ("带血的呕吐物", "深色呕吐物"),
        ("吐出来的血", "呕吐物中的异常内容"),
        ("吐血", "呕吐物异常"),
        ("一点血丝", "少量异常颜色"),
        ("鲜红一大口", "大量红色内容物"),
        ("肛诊黑便", "排便检查见黑便"),
        ("直肠指检鲜血", "排便检查见红色便"),
        ("鲜红血", "红色便"),
        ("直肠出血", "下消化道出血"),
        ("裂片状出血", "甲下线状改变"),
        ("直肠阴道瘘", "肠道瘘管"),
        ("咖啡渣样呕吐", "深色呕吐物"),
        ("呕血", "呕吐物含血性内容"),
        ("伤口止血变慢", "凝血较慢"),
        ("突然吐出来一大口血，量挺吓人的", "突然出现较多异常呕吐物"),
        ("咯血", "呼吸道血性分泌物"),
        ("明显失血", "明显血容量减少"),
        ("血尿", "尿液见红细胞"),
        ("肛门直肠", "下消化道"),
        ("直肠", "下消化道"),
        ("肛周", "肠道周围"),
        ("肛门", "排便部位"),
        ("生殖器", "会阴部"),
    )
    for source, replacement in replacements:
        safe_prompt = safe_prompt.replace(source, replacement)
    safe_prompt = re.sub(
        r"===\s*该患者无的体征（应按正常输出，不得报告异常）\s*==="
        r".*?(?=\n伴随疾病：)",
        "=== 未记录为阳性的体征 ===\n未列出的申请项目按正常结果输出。",
        safe_prompt,
        flags=re.DOTALL,
    )
    safe_prompt = re.sub(
        r"===\s*该患者无的（应正常）检查项目\s*==="
        r".*?(?=\n===\s*患者本人|\n医生申请的检查：)",
        "=== 未记录为阳性的检查项目 ===\n未列出的申请项目按正常结果输出。\n",
        safe_prompt,
        flags=re.DOTALL,
    )
    safe_prompt = re.sub(
        r"\n===\s*该患者有的临床表现.*?(?=\n既往疾病史：)",
        "\n",
        safe_prompt,
        flags=re.DOTALL,
    )
    safe_prompt = re.sub(
        r"\n【起病期（发病至就诊前）】.*?(?=\n- 实验室检查：)",
        "",
        safe_prompt,
        flags=re.DOTALL,
    )
    safe_prompt = re.sub(r"已切除\s*(约\s*)?", r"手术范围\1", safe_prompt)
    safe_prompt = safe_prompt.replace("切除术", "手术治疗")
    safe_prompt = safe_prompt.replace("切除", "手术处理")
    return safe_prompt


def call_gpt5(prompt, tag="", max_retries=DEFAULT_MAX_RETRIES, max_tokens=None):
    """
    调用 GPT-5.5（直接 HTTP 流式），自动过滤 <think> 推理标签。
    节流间隔受 _LLM_MIN_INTERVAL 控制，失败时反复重试直至成功或达 max_retries。
    """
    token_budget = DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    if token_budget < 1:
        raise ValueError('max_tokens 必须 >= 1')
    gpt_start = time.time()
    current_prompt = prompt
    for attempt in range(1, max_retries + 1):
        attempt_start = time.time()
        backoff_429 = False
        try:
            _llm_pace(_LLM_PACE_LOCK, _LLM_LAST_CALL_TS, _LLM_MIN_INTERVAL)
            cleaned = _call_llm_stream_once(
                current_prompt, DEFAULT_MODEL_NAME, DEFAULT_LLM_URL,
                DEFAULT_API_KEY, token_budget, DEFAULT_API_TIMEOUT_SECONDS,
            )
            if cleaned:
                elapsed = round(time.time() - gpt_start, 1)
                print(f"  [{tag}] ⏱ {DEFAULT_MODEL_NAME} 成功，本次 {round(time.time()-attempt_start,1)}s，"
                      f"总计 {elapsed}s，{len(cleaned)} 字符")
                time.sleep(DEFAULT_REQUEST_DELAY)
                return cleaned
            print(f"  [{tag}] {DEFAULT_MODEL_NAME} 空响应（第{attempt}次, "
                  f"{round(time.time()-attempt_start,1)}s）")
        except Exception as e:
            msg = str(e)
            if _is_content_filter_error(msg):
                current_prompt = _rephrase_content_filter_prompt(current_prompt)
            backoff_429 = ('HTTP 429' in msg or 'RATE_LIMIT' in msg
                           or 'too_many_requests' in msg or 'Too Many Requests' in msg)
            short = msg if len(msg) < 200 else msg[:200] + '...'
            print(f"  [{tag}] {DEFAULT_MODEL_NAME} 异常（第{attempt}次, "
                  f"{round(time.time()-attempt_start,1)}s）: {short}")
        if attempt < max_retries:
            sleep_sec = (random.uniform(2.0, 8.0) + min(attempt, 5)
                         if backoff_429
                         else min(DEFAULT_RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)), 60))
            print(f"  [{tag}] {round(sleep_sec,1)}s 后重试...")
            time.sleep(sleep_sec)

    elapsed = round(time.time() - gpt_start, 1)
    print(f"  [{tag}] 所有重试均失败，总计耗时 {elapsed}s")
    return None


def call_gpt54(prompt, tag="", max_retries=DEFAULT_MAX_RETRIES, max_tokens=None):
    """
    调用表型图谱模型（默认 GPT-5.5，直接 HTTP 流式）。
    与 call_gpt5 逻辑相同，保留函数名以兼容旧代码。
    """
    token_budget = DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    if token_budget < 1:
        raise ValueError('max_tokens 必须 >= 1')
    gpt_start = time.time()
    for attempt in range(1, max_retries + 1):
        attempt_start = time.time()
        backoff_429 = False
        try:
            _llm_pace(_GPT54_PACE_LOCK, _GPT54_LAST_CALL_TS, _GPT54_MIN_INTERVAL)
            cleaned = _call_llm_stream_once(
                prompt, _GPT54_MODEL_NAME, DEFAULT_LLM_URL,
                DEFAULT_API_KEY, token_budget, DEFAULT_API_TIMEOUT_SECONDS,
            )
            if cleaned:
                elapsed = round(time.time() - gpt_start, 1)
                print(f"  [{tag}] ⏱ {_GPT54_MODEL_NAME} 成功，本次 {round(time.time()-attempt_start,1)}s，"
                      f"总计 {elapsed}s，{len(cleaned)} 字符")
                time.sleep(DEFAULT_REQUEST_DELAY)
                return cleaned
            print(f"  [{tag}] {_GPT54_MODEL_NAME} 空响应（第{attempt}次, "
                  f"{round(time.time()-attempt_start,1)}s）")
        except Exception as e:
            msg = str(e)
            backoff_429 = ('HTTP 429' in msg or 'RATE_LIMIT' in msg
                           or 'too_many_requests' in msg or 'Too Many Requests' in msg)
            short = msg if len(msg) < 200 else msg[:200] + '...'
            print(f"  [{tag}] {_GPT54_MODEL_NAME} 异常（第{attempt}次, "
                  f"{round(time.time()-attempt_start,1)}s）: {short}")
        if attempt < max_retries:
            sleep_sec = (random.uniform(2.0, 8.0) + min(attempt, 5)
                         if backoff_429
                         else min(DEFAULT_RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)), 60))
            print(f"  [{tag}] {round(sleep_sec,1)}s 后重试...")
            time.sleep(sleep_sec)

    elapsed = round(time.time() - gpt_start, 1)
    print(f"  [{tag}] 所有重试均失败，总计耗时 {elapsed}s")
    return None


def _retry_module_call(module_func, args=(), kwargs=None,
                       max_retries=DEFAULT_MODULE_MAX_RETRIES,
                       retry_sleep=DEFAULT_MODULE_RETRY_SLEEP,
                       module_name="模块"):
    """
    模块级重试包装器：反复调用 module_func 直到返回有效结果，或达到最大重试次数。
    """
    if kwargs is None:
        kwargs = {}
    for attempt in range(1, max_retries + 1):
        result = module_func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            prompt, response = result
            if response is not None:
                return result
        elif result is not None:
            return result

        if attempt < max_retries:
            print(f"  [{module_name}] ⚠️ 第 {attempt} 次模块调用失败，"
                  f"{retry_sleep}s 后进行第 {attempt+1} 次重试...")
            time.sleep(retry_sleep)
        else:
            print(f"  [{module_name}] ❌ 经过 {max_retries} 次重试仍失败，终止后续流程")
            raise RuntimeError(f"{module_name} 经过 {max_retries} 次重试仍失败")


def parse_list_from_response(response_text):
    """
    从 GPT 响应中解析列表。兼容 Python 元组列表 (a,b) 与 JSON 数组 [a,b]。
    """
    if not response_text:
        return []

    text = response_text.strip()

    code_block_match = re.search(r'```(?:python|json)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket_match:
        text = bracket_match.group(0)

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [tuple(x) if isinstance(x, list) else x for x in parsed]
    except (ValueError, TypeError):
        pass

    tuples = re.findall(r'\(([^)]+)\)', text)
    result = []
    for t in tuples:
        try:
            parsed_tuple = ast.literal_eval(f'({t})')
            result.append(parsed_tuple)
        except (ValueError, SyntaxError):
            continue

    return result


def parse_staging_system(response_text):
    """Parse module 1.0 staging system output (Python dict format)"""
    if not response_text:
        return None
    text = response_text.strip()
    code_block_match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    dict_match = re.search(r'\{.*\}', text, re.DOTALL)
    if dict_match:
        text = dict_match.group(0)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict) and 'levels' in parsed:
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def _build_staging_levels_text(staging_system):
    """Build human-readable staging levels text for injection into prompts"""
    if not staging_system or 'levels' not in staging_system:
        return '- 级别1 (早期): 疾病早期阶段\n- 级别2 (进展期): 疾病进展后阶段'
    lines = []
    for lv in staging_system['levels']:
        lines.append(f"- 级别{lv['level']} ({lv['name']}): {lv['description']}")
    return '\n'.join(lines)


def _get_num_levels(staging_system):
    """Get number of levels from staging system"""
    if not staging_system or 'levels' not in staging_system:
        return 2
    return len(staging_system['levels'])


def _get_level_names(staging_system):
    """Get ordered list of level names from staging system"""
    if not staging_system or 'levels' not in staging_system:
        return ['早期', '进展期']
    return [lv['name'] for lv in staging_system['levels']]


# ============================================================
# 主诉提取（GPT调用）
# ============================================================

def _derive_chief_complaint(corrected_text: str, seed_text: str = '', tag: str = '主诉',
                            symptoms_list=None, diagnosis: str = '',
                            patient_stage: str = '',
                            suggested: str = '',
                            time_ordered_items=None) -> str:
    """
    选取 1 个最适合就诊的主诉。

    数据源优先级：
      1) 若显式传入 symptoms_list（模块2.4收敛后的 selected_symptoms，元组 item[0] 为症状名），优先用它；
      2) 否则尝试从 corrected_text 中解析 item[1]=='症状' 的条目（新 4 元组格式，类型在索引 1）；
      3) 若仍为空但存在客观表型/诊断，则返回“体检发现...相关异常”作为无症状就诊原因。

    **严格只选"症状"**，不得从体征/化验/影像/功能检查里兜底。
    例外：症状列表真实为空时，可以生成“体检发现/检查发现异常”类非症状就诊原因。
    suggested: 模块2.6提供的主诉提示，若非空则优先采用并校验是否在 symptom_names 中。
    time_ordered_items: 模块2.5输出的时间排序四元组 (name, category, phase, rel_time)，
        若提供则把"出现时间相对靠近就诊"作为额外的选择权重。
    """
    symptom_names = []

    if symptoms_list:
        for it in symptoms_list:
            if isinstance(it, (list, tuple)) and it:
                name = str(it[0]).strip()
                if name and name not in symptom_names:
                    symptom_names.append(name)

    if not symptom_names and corrected_text and corrected_text.strip() not in ('', '[]'):
        items = parse_list_from_response(corrected_text)
        for it in items:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            cat = None
            if len(it) >= 2 and str(it[1]) in ('症状', '体征', '实验室检查', '化验检查', '影像检查', '功能检查'):
                cat = str(it[1])
            if cat == '症状':
                name = str(it[0]).strip()
                if name and name not in symptom_names:
                    symptom_names.append(name)

    # 短路 1：调用方明确给出主诉提示（来自模块2.6）
    if suggested and str(suggested).strip() not in {'无', '无症状', '无明显症状'}:
        s = str(suggested).strip().strip('。，、.， "\'')
        if s:
            # 若提示能在 selected 症状中找到对应项，按对应症状返回
            for name in symptom_names:
                if name in s or s in name:
                    return name
            # 否则直接采用提示文本
            return s

    if not symptom_names:
        diagnosis_text = str(diagnosis or '').strip().strip('。，、.， "\'')
        if diagnosis_text:
            return f'体检发现{diagnosis_text}相关异常'
        objective_names = []
        source_items = time_ordered_items
        if source_items is None and corrected_text and corrected_text.strip() not in ('', '[]'):
            source_items = parse_list_from_response(corrected_text)
        for it in source_items or []:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            category = str(it[1]).strip()
            if category in ('体征', '实验室检查', '化验检查', '影像检查', '功能检查'):
                name = str(it[0]).strip()
                if name:
                    objective_names.append(name)
        if objective_names:
            return f'体检发现{objective_names[0]}'
        return '体检发现异常'

    symptom_list_str = '、'.join(symptom_names)
    seed_info = f'患者信息：{seed_text}\n' if seed_text else ''
    stage_info = f'疾病分期：{patient_stage}\n' if patient_stage else ''

    # 提取症状的时间标签，作为"出现时间近"的权重提示
    time_hint = ''
    if time_ordered_items:
        pairs = []
        for it in time_ordered_items:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            name = str(it[0]).strip()
            if name not in symptom_names:
                continue
            tlabel = str(it[-1]).strip() if len(it) >= 3 else ''
            if tlabel:
                pairs.append(f'{name}({tlabel})')
        if pairs:
            time_hint = f'症状出现时间标签（H/D/M/Y 数字越小越接近就诊当前）：{"、".join(pairs)}\n'

    prompt = f"""你是一位资深临床医生。{seed_info}{stage_info}{time_hint}该患者目前存在以下主观症状：{symptom_list_str}

请从中选出最适合作为该患者此次就诊主诉的 1 个症状。
选择原则（综合权衡两个维度，二者都越突出越优先选）：
1. **对患者日常生活的影响**：影响越大越优先（如气短、严重疼痛、危及生命表现）。
2. **出现时间距离就诊越近**：在时间标签中越接近 H0/D0 越优先（H-3 比 D-7 更接近，D-2 比 M-3 更接近，M-1 比 Y-2 更接近）。
其他硬性约束：
- 只能选"症状"列表中的条目，**严禁选体征、化验或影像指标**。
- 只能选患者能主观感受的症状，不要选体重下降、水肿等非典型首发表现，除非它确实是该患者最突出的不适。
- 只输出症状名称本身，不要加任何解释或标点符号。

示例：呼吸困难"""

    response = call_gpt5(prompt, tag=tag)
    if response:
        chief = response.strip().splitlines()[0].strip('。，、.， "\'')
        if chief and any(s in chief or chief in s for s in symptom_names):
            return chief
        for s in symptom_names:
            if s in response:
                return s
    return symptom_names[0]


# ============================================================
# CSV 格式化辅助函数 & I/O
# ============================================================

def _normalize_to_json(text):
    """
    将 Python dict/list 的 repr 字符串或 JSON 文本统一规整为
    JSON (ensure_ascii=False) 字符串。无法解析时原样返回。
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not s:
        return text

    m = re.search(r'```(?:python|json)?\s*\n?(.*?)```', s, re.DOTALL)
    if m:
        s = m.group(1).strip()

    payload = None
    try:
        payload = json.loads(s)
    except (ValueError, TypeError):
        try:
            payload = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            m2 = re.search(r'[\[\{].*[\]\}]', s, re.DOTALL)
            if m2:
                chunk = m2.group(0)
                try:
                    payload = json.loads(chunk)
                except (ValueError, TypeError):
                    try:
                        payload = ast.literal_eval(chunk)
                    except (ValueError, SyntaxError):
                        payload = None

    if payload is None:
        return text
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return text


def _format_tuples_for_csv(text):
    """
    将Python/JSON列表字符串格式化为每个元素单独一行，方便人工阅读CSV。

    支持：
      1. 元组列表：[(a,b),(c,d)]  →  [\n(a,b),\n(c,d)\n]
      2. JSON 对象数组：[{...},{...}]  →  [\n{...},\n{...}\n]
      3. JSON 嵌套数组：[[...],[...]]  →  [\n[...],\n[...]\n]
      4. 带 <就诊x>...</就诊x> 标签的多列表
    """
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return text

    def _split_top_level(list_body: str) -> str:
        """在最外层的 ),( / },{ / ],[ 处换行；保留内部嵌套不拆。"""
        depth_round = depth_square = depth_curly = 0
        in_str = False
        str_ch = ''
        escape = False
        out = []
        i = 0
        n = len(list_body)
        while i < n:
            ch = list_body[i]
            out.append(ch)
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == str_ch:
                    in_str = False
            else:
                if ch == '"' or ch == "'":
                    in_str = True
                    str_ch = ch
                elif ch == '(':
                    depth_round += 1
                elif ch == ')':
                    depth_round -= 1
                elif ch == '[':
                    depth_square += 1
                elif ch == ']':
                    depth_square -= 1
                elif ch == '{':
                    depth_curly += 1
                elif ch == '}':
                    depth_curly -= 1
                elif ch == ',':
                    if (depth_round == 0 and depth_square == 0
                            and depth_curly == 0):
                        j = i + 1
                        while j < n and list_body[j] == ' ':
                            j += 1
                        out.append('\n')
                        i = j - 1
            i += 1
        return ''.join(out)

    def _format_one_list(list_part: str) -> str:
        """接收形如 [...] 的片段，将最外层元素拆行。"""
        if not (list_part.startswith('[') and list_part.endswith(']')):
            return list_part
        inner = list_part[1:-1]
        if not inner.strip():
            return list_part
        formatted_inner = _split_top_level(inner)
        if '\n' not in formatted_inner:
            return list_part
        return '[\n' + formatted_inner + '\n]'

    visit_pattern = re.compile(r'(<就诊\d+>)(\[.*?\])(</就诊\d+>)', re.DOTALL)
    if visit_pattern.search(stripped):
        def _format_visit_block(m):
            return m.group(1) + _format_one_list(m.group(2)) + m.group(3)
        result = visit_pattern.sub(_format_visit_block, stripped)
        result = re.sub(r'(</就诊\d+>)\s*(<就诊\d+>)', r'\1\n\2', result)
        return result

    match = re.search(r'\[.*\]', stripped, re.DOTALL)
    if not match:
        return text

    list_part = match.group(0)
    formatted = _format_one_list(list_part)
    if formatted == list_part:
        return text

    return stripped[:match.start()] + formatted + stripped[match.end():]


def _module_io_dir(csv_path):
    """返回某个 csv 的 module_io 侧车目录，不存在则创建。"""
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_name = os.path.basename(csv_path)
    dir_path = os.path.join(base_dir, f'{csv_name}.module_io')
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def _module_io_rel(csv_path, row_idx):
    """返回形如 'xxx.csv.module_io/row_0.json' 的相对路径（CSV 同目录相对）。"""
    csv_name = os.path.basename(csv_path)
    return f'{csv_name}.module_io/row_{row_idx}.json'


def _module_io_abs(csv_path, row_idx):
    return os.path.join(_module_io_dir(csv_path), f'row_{row_idx}.json')



def _read_json_file(path, default=None):
    """Read a UTF-8 JSON sidecar file; return default when it is absent."""
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _atomic_write_json_file(path, payload):
    """Atomically write a UTF-8 JSON sidecar file with os.replace."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f'{path}.tmp.{os.getpid()}.{threading.get_ident()}'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _make_io_pointer(csv_path, row_idx, col):
    """cell 内的指针文本：@module_io/xxx.csv.module_io/row_{idx}.json#col"""
    return f'{_MODULE_IO_POINTER_PREFIX}{_module_io_rel(csv_path, row_idx)}#{col}'


def _resolve_module_io_pointer(val, csv_path):
    """若 val 是 @module_io/... 指针，解析并返回原始内容；否则原样返回。"""
    if not isinstance(val, str) or not val.startswith(_MODULE_IO_POINTER_PREFIX):
        return val
    try:
        rest = val[len(_MODULE_IO_POINTER_PREFIX):]
        if '#' not in rest:
            return val
        rel_path, col_key = rest.split('#', 1)
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        abs_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(abs_path):
            return ''
        with open(abs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(col_key, '') or ''
    except Exception:
        return val


def _load_existing_csv(csv_path):
    """
    加载已有的CSV文件，返回第1行（概率库）和已有的患者行数据。
    模块_输入/输出 列若是 @module_io 指针，会自动展开为原始内容，
    便于续跑时保留上下文。

    Returns:
        tuple: (row_1, existing_patients)
            - row_1: dict, 第1行数据；CSV不存在或为空时返回None
            - existing_patients: dict, {patient_idx(int): row_dict}
    """
    if not os.path.exists(csv_path):
        return None, {}
    try:
        df = pd.read_csv(
            csv_path,
            encoding='utf-8-sig',
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
        if len(df) == 0:
            return None, {}

        def _series_to_dict(series):
            d = {}
            for col in CSV_COLUMNS:
                raw = series.get(col, '')
                if pd.isna(raw):
                    d[col] = ''
                    continue
                val = str(raw)
                if col in _MODULE_IO_COLUMNS and val.startswith(_MODULE_IO_POINTER_PREFIX):
                    val = _resolve_module_io_pointer(val, csv_path) or val
                d[col] = val
            return d

        row_1 = _series_to_dict(df.iloc[0])
        existing_patients = {}
        for i in range(1, len(df)):
            existing_patients[i] = _series_to_dict(df.iloc[i])

        return row_1, existing_patients
    except Exception as e:
        print(f"  ⚠️ 加载CSV失败: {e}，将重新开始")
        return None, {}


def _save_csv(csv_path, all_rows):
    """
    安全保存CSV：
      * 使用固定列顺序；
      * 模块_输入/输出 列落盘前拆到 `<csv>.module_io/row_{idx}.json`，CSV 仅保留 `@module_io/...` 指针；
      * 使用"临时文件 + os.replace"方式原子写入，避免异常终止导致部分写入（POSIX 下 os.replace 原子）。
    """
    save_data = []
    for idx, r in enumerate(all_rows):
        save_row = {}
        io_payload = {}
        for col in CSV_COLUMNS:
            val = r.get(col, '')
            if col in _MODULE_IO_COLUMNS:
                if isinstance(val, str) and val.startswith(_MODULE_IO_POINTER_PREFIX):
                    save_row[col] = val
                    continue
                if isinstance(val, str) and val.strip():
                    io_payload[col] = val
                    save_row[col] = _make_io_pointer(csv_path, idx, col)
                else:
                    save_row[col] = ''
                continue
            if col in _JSON_DICT_COLUMNS:
                val = _normalize_to_json(val)
            if col in _LIST_COLUMNS:
                val = _normalize_to_json(val)
                val = _format_tuples_for_csv(val)
            save_row[col] = val
        save_data.append(save_row)

        if io_payload:
            try:
                abs_path = _module_io_abs(csv_path, idx)
                existing = {}
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            existing = json.load(f) or {}
                    except Exception:
                        existing = {}
                existing.update(io_payload)
                tmp_io = f'{abs_path}.tmp.{os.getpid()}.{threading.get_ident()}'
                with open(tmp_io, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                os.replace(tmp_io, abs_path)
            except Exception as e:
                print(f"  ⚠️ 写入 module_io 侧车文件失败: {e}（将回退到 CSV 单元格内存）")
                for c, v in io_payload.items():
                    save_data[-1][c] = v
    df = pd.DataFrame(save_data, columns=CSV_COLUMNS)
    tmp_path = f"{csv_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        df.to_csv(tmp_path, index=False, encoding='utf-8-sig')
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
