#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_and_reset_seed.py
扫描单个 seed CSV，检测每位患者各模块完成状态，清空未完成的 M3-4 对话后写回文件。

用法：
    python check_and_reset_seed.py <csv_path>

退出码：
    0 — 正常（含"已全部完成"）
    1 — 文件不存在或读取错误
"""

import sys
import os
import re
import pandas as pd

# ─── M3 完成判断（与 virtual_clinical_interaction.py 保持一致）─────────────
_M3_COMPLETION_MARKERS = (
    '[鉴别诊断]',
    '[医生诊断]',
    '[诊断]',
    '鉴别诊断：',
    '诊断：',
    '（诊断生成失败）',
)

def _is_interaction_complete(text: str) -> bool:
    if not text or not text.strip():
        return False
    tail = text[-2000:]
    return any(marker in tail for marker in _M3_COMPLETION_MARKERS)

# ─── 需要清空的 M3-4 列（M3未完成时）──────────────────────────────────────
_M3_RESET_COLS = [
    'v4模拟患者_交互历史',
    'v4模拟患者_治疗方案',
    '不良反应',
    '治疗效果',
    '模块4.1_输入',
    '模块4.1_输出',
    '模块4.2_输入',
    '模块4.2_输出',
    'v4模拟患者_病案',  # EHR 依赖 M3，一并清空
]

# ─── 主逻辑 ────────────────────────────────────────────────────────────────

def check_and_reset(csv_path: str) -> dict:
    """
    返回 dict:
      total_patients  — 患者总数（不含 row_1）
      m2_done         — M2 已完成数
      m3_done         — M3-4 已完成数
      m5_done         — M5 已完成数
      reset_count     — 本次清空的患者数
    """
    if not os.path.exists(csv_path):
        print(f"[error] 文件不存在: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str).fillna('')
    except Exception as e:
        print(f"[error] 读取失败: {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if len(df) < 2:
        print(f"[info] {csv_path}: 无患者行（仅有 row_1 或空文件），跳过")
        return {'total_patients': 0, 'm2_done': 0, 'm3_done': 0, 'm5_done': 0, 'reset_count': 0}

    # row_1 是表型图谱行（索引0），患者行从索引1开始
    patient_rows = df.index[1:]
    total = len(patient_rows)
    m2_done = m3_done = m5_done = reset_count = 0
    modified = False

    for idx in patient_rows:
        row = df.loc[idx]

        # M2 完成判断
        m2_ok = bool(str(row.get('具体数值化', '')).strip())
        if m2_ok:
            m2_done += 1

        # M3 完成判断
        interaction = str(row.get('v4模拟患者_交互历史', '') or '')
        m3_ok = _is_interaction_complete(interaction)
        if m3_ok:
            m3_done += 1
        elif interaction.strip():
            # 有半截对话但未完成 → 清空
            for col in _M3_RESET_COLS:
                if col in df.columns:
                    df.at[idx, col] = ''
            reset_count += 1
            modified = True

        # M5 完成判断
        m5_val = str(row.get('GPT直演患者_病案', '') or '').strip()
        m5_ok = bool(m5_val) and m5_val != 'nan'
        if m5_ok:
            m5_done += 1

    if modified:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[reset] {os.path.basename(csv_path)}: 清空 {reset_count} 位患者的半截 M3-4 对话")

    return {
        'total_patients': total,
        'm2_done': m2_done,
        'm3_done': m3_done,
        'm5_done': m5_done,
        'reset_count': reset_count,
    }


def main():
    if len(sys.argv) < 2:
        print(f"用法: {sys.argv[0]} <csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    result = check_and_reset(csv_path)
    t = result['total_patients']
    print(
        f"[check] {os.path.basename(csv_path)}: "
        f"患者{t}人  M2完成{result['m2_done']}  M3完成{result['m3_done']}  "
        f"M5完成{result['m5_done']}  本次清空{result['reset_count']}"
    )


if __name__ == '__main__':
    main()
