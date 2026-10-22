#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题四全年逐段结果写入器 + 独立审计。

从 ``results/q4_segments.csv``（另一路并行任务的产物，本脚本只读）读取 4-2 与
4-3 两个分支的全年逐段结果，写出：

  * ``result4-2.xlsx`` —— 计划购电量 / 充放电量 / 紧急购电量
  * ``result4-3.xlsx`` —— 计划购电量 / 调整购电量 / 充放电量 / 紧急购电量

并把独立复算的审计写入 ``results/q4_write_audit.json``。

布局完全沿用 ``scripts/solve_q2.py::write_template`` 与
``scripts/solve_q3.py::write_result3``（copy 模板 → 只写数据区，表头与
144 个时段标签原样保留）；两个 4-2/4-3 模板取自 ``附件/附件5/``，仓库根
的 ``result2.xlsx`` / ``result3.xlsx`` 仅作布局参照，绝不覆盖 ``附件/``。

结算口径（两支**独立实现**，不复用同一个费用函数，也不 import q4_models）：
  * 4-2：账单 = Σ_t p·plan_t + 5·Σ_t p·emergency_t   （计划冻结，无差额项）
  * 4-3：账单 = Σ_t p·min(x0_t, a_t)
                + 0.5·Σ_t p·(x0_t − a_t)⁺
                + 1.5·Σ_t p·(a_t − x0_t)⁺
                + 5·Σ_t p·emergency_t
    其中 x0=plan_kwh（0:00 计划量），a=adjusted_kwh（执行前最后确认量），
    p=price（实现价）。

输入契约（列严格）：
  date,slot,variant,price,load_kwh,pv_kwh,plan_kwh,adjusted_kwh,
  charge_kwh,discharge_kwh,emergency_kwh,soc_end_kwh
  variant ∈ {"4-2","4-3"}；slot ∈ 1..144；date 覆盖 2025-02-01..2025-12-31。

用法：
  python scripts/write_result4.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

try:                                     # Windows GBK 控制台兼容
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RES = ROOT / "results"
SEGMENTS_CSV = RES / "q4_segments.csv"
SUMMARY_JSON = RES / "harness_q4_summary.json"
AUDIT_JSON = RES / "q4_write_audit.json"


def _relative_to_root(path: Path) -> str:
    """审计记录中的路径统一写成相对工程根，避免把本机绝对路径写进结果文件。"""
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name

TEMPLATE_42 = ROOT / "附件" / "附件5" / "result4-2.xlsx"
TEMPLATE_43 = ROOT / "附件" / "附件5" / "result4-3.xlsx"
REF_42 = ROOT / "附件" / "附件5" / "result2.xlsx"      # 布局参照
REF_43 = ROOT / "附件" / "附件5" / "result3.xlsx"      # 布局参照
OUT_42 = ROOT / "result4-2.xlsx"
OUT_43 = ROOT / "result4-3.xlsx"

# ---- 与 scripts/q4_models.py 一致的储能/时间参数（仅用于结构推算）----
ETA, EMIN, EMAX = 0.9, 1200.0, 10800.0
PMAX = 5000.0
BMAX = PMAX / 6.0
E0 = 6000.0
T = 144

COLS = ["date", "slot", "variant", "price", "load_kwh", "pv_kwh", "plan_kwh",
        "adjusted_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
        "unused_kwh", "soc_end_kwh"]
VARIANTS = ("4-2", "4-3")

# 模板列序：slot2..144 → 列 2..144，slot1 → 列 145；列146=全天量，列147=费用
SOURCE_SLOTS = list(range(2, 145)) + [1]
BLOCKS = [(b * 24 + 1, (b + 1) * 24) for b in range(6)]   # 六个 4h 区间

EXPECTED_DAYS = 334
EXPECTED_ROWS = EXPECTED_DAYS * T * len(VARIANTS)

# 对账：harness_q4_summary.json 中与 4-2 / 4-3 主分支对应的键
REF_KEY_42 = ("total_yuan_42", "Q42_nb")   # 4-2 主路线 = 报童计划 + 5×紧急
REF_KEY_43 = ("total_yuan", "Q0_fixed")    # 4-3 主路线 = 样本外最优稳健价格信号
FEE_TOL = 1.0                              # 对账容差（元）


# ============================================================ 标签 / 区间工具
def hm(minute_index: int) -> str:
    """0-based 段号 → 该段**起点**时刻标签（与 solve_q2/q3 完全一致）。"""
    m = minute_index * 10
    return f"{m // 60}:{m % 60:02d}" if m < 1440 else f"0:{m % 60:02d}+1"


def emergency_ranges(em: np.ndarray) -> list[tuple[str, float]]:
    """合并相邻紧急购电时段为一个事件：返回 [(区间标签, 电量), ...]。"""
    em = np.asarray(em, dtype=float)
    active = em > 1e-6
    ans: list[tuple[str, float]] = []
    i = 0
    n = len(em)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1]:
            j += 1
        ans.append((f"{hm(i)}-{hm(j + 1)}", float(em[i:j + 1].sum())))
        i = j + 1
    return ans


def slot_from_label(label) -> int:
    """由模板时段标签（形如 '0:10-0:20' / '0:00-0:10+1' / '7:0-7:10'）反推 slot。"""
    left = str(label).split("-")[0]
    h, m = left.split(":")
    return (int(h) * 60 + int(m)) // 10 + 1


# ============================================================ 费用函数（独立）
def fee_q42(price, plan, emergency) -> float:
    """4-2 全年费用：Σ p·plan + 5·Σ p·emergency（无少购/超购项）。"""
    p = np.asarray(price, dtype=float)
    z = np.asarray(plan, dtype=float)
    g = np.asarray(emergency, dtype=float)
    plan_cost = float(np.sum(p * z))
    em_cost = float(np.sum(5.0 * p * g))
    return plan_cost + em_cost


def fee_q43(price, plan, adjusted, emergency) -> float:
    """4-3 全年费用：口径 C（计划内 + 少购 + 超购 + 5×紧急）。"""
    p = np.asarray(price, dtype=float)
    x0 = np.asarray(plan, dtype=float)
    a = np.asarray(adjusted, dtype=float)
    g = np.asarray(emergency, dtype=float)
    under = np.maximum(x0 - a, 0.0)
    over = np.maximum(a - x0, 0.0)
    within_cost = float(np.sum(p * np.minimum(x0, a)))
    under_cost = float(np.sum(0.5 * p * under))
    over_cost = float(np.sum(1.5 * p * over))
    em_cost = float(np.sum(5.0 * p * g))
    return within_cost + under_cost + over_cost + em_cost


# ============================================================ 输入
def _fmt_err(path: Path, n: int) -> str:
    return (f"[错误] {path} 行数不足：{n} < {EXPECTED_ROWS}"
            f"（= {EXPECTED_DAYS} 天 × 144 段 × {len(VARIANTS)} 分支）。")


def load_segments(path: Path = SEGMENTS_CSV) -> pd.DataFrame:
    """读取并校验逐段 CSV；文件缺失或行数不足则报清晰错误并退出。"""
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"[错误] 找不到输入文件：{path}\n"
            f"        该文件应由问题四逐段求解流程产出（列严格为：{','.join(COLS)}）。\n"
            f"        预期覆盖 {EXPECTED_DAYS} 天 × 144 段 × 2 分支 = {EXPECTED_ROWS} 行。\n"
            f"        （本脚本不会伪造 / 硬编码任何结果数据。）")
    df = pd.read_csv(path)
    missing = [c for c in COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLS]
    if missing or extra:
        sys.exit(f"[错误] {path} 列不符合契约：缺少 {missing}；多出 {extra}\n"
                 f"        期望列严格为 {COLS}")
    bad = sorted(set(df["variant"].astype(str).unique()) - set(VARIANTS))
    if bad:
        sys.exit(f"[错误] variant 含未预期取值 {bad}；期望 {list(VARIANTS)}")
    if len(df) < EXPECTED_ROWS:
        sys.exit(fmt_err(path, len(df)))
    df = df[COLS].copy()
    df["date"] = df["date"].astype(str)
    df["slot"] = df["slot"].astype(int)
    return df


def validate(df: pd.DataFrame) -> None:
    """硬约束校验；违反则抛 ValueError（含清晰诊断）。"""
    counts = df.groupby(["date", "variant"])["slot"].count()
    bad = counts[counts != T]
    if len(bad):
        raise ValueError(f"存在非 144 段的日/分支：{bad.head(5).to_dict()}")
    for col in ("plan_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh"):
        v = df[col].to_numpy(float)
        if np.nanmin(v) < -1e-6:
            raise ValueError(f"{col} 出现负值：min={float(np.nanmin(v))}")
    adj = df["adjusted_kwh"].to_numpy(float)
    if np.nanmin(adj) < -1e-6:
        raise ValueError(f"adjusted_kwh 出现负值（售电）：min={float(np.nanmin(adj))}")
    soc = df["soc_end_kwh"].to_numpy(float)
    smin, smax = float(np.nanmin(soc)), float(np.nanmax(soc))
    if smin < EMIN - 1e-6 or smax > EMAX + 1e-6:
        raise ValueError(
            f"SOC 越界：soc_end ∈ [{smin:.3f}, {smax:.3f}] 超出 [{EMIN}, {EMAX}] kWh")


def build_index(df: pd.DataFrame) -> dict:
    idx = {}
    for (day, variant), g in df.groupby(["date", "variant"]):
        idx[(day, variant)] = g.sort_values("slot").reset_index(drop=True)
    return idx


# ============================================================ xlsx 写入
def _date_fmt(ws) -> str:
    try:
        f = ws.cell(2, 1).number_format
    except Exception:  # noqa: BLE001
        f = None
    return f if (f and f != "General") else "yyyy-mm-dd"


def _clear_data(ws) -> None:
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)


def _storage_labels(ws) -> list[str]:
    labs = [ws.cell(r, 2).value for r in range(2, 8)]
    if any(v is None for v in labs):
        labs = [f"{4 * b}:00-{4 * (b + 1)}:00" for b in range(6)]
    return [str(v) for v in labs]


def _write_plan_sheet(ws, days, idx, variant, fmt, fee_col):
    """写「计划购电量」/「调整购电量」：slot 逐段量 + 全天量 + 费用列。

    fee_col: "plan"  → Σ p·量
             "q43"   → 口径 C 全天总费用（4-3 调整购电量末列）
    """
    _clear_data(ws)
    val_col = "plan_kwh" if fee_col != "q43" else "adjusted_kwh"
    for i, day in enumerate(days, start=2):
        g = idx[(day, variant)]
        gs = g.set_index("slot")
        ws.cell(i, 1, pd.Timestamp(day).to_pydatetime()).number_format = fmt
        for j, slot in enumerate(SOURCE_SLOTS):
            ws.cell(i, 2 + j, round(float(gs.at[slot, val_col]), 6))
        ws.cell(i, 146, round(float(g[val_col].sum()), 6))
        if fee_col == "q43":
            fee = fee_q43(g["price"], g["plan_kwh"], g["adjusted_kwh"],
                          g["emergency_kwh"])
        else:
            fee = float((g[val_col] * g["price"]).sum())
        ws.cell(i, 147, round(fee, 6))


def _write_storage_sheet(ws, days, idx, variant, fmt):
    labels = _storage_labels(ws)
    _clear_data(ws)
    row = 2
    for day in days:
        g = idx[(day, variant)]
        c = g["charge_kwh"].to_numpy(float)
        d = g["discharge_kwh"].to_numpy(float)
        soc_end = g["soc_end_kwh"].to_numpy(float)
        slot = g["slot"].to_numpy(int)
        soc0 = float(soc_end[0] - ETA * c[0] + d[0] / ETA)   # 反推 0:00
        soc24 = float(soc_end[-1])                            # 24:00
        for b in range(6):
            lo, hi = BLOCKS[b]
            m = (slot >= lo) & (slot <= hi)
            cell = ws.cell(row, 1, pd.Timestamp(day).to_pydatetime() if b == 0 else None)
            if b == 0:
                cell.number_format = fmt
            ws.cell(row, 2, labels[b])
            ws.cell(row, 3, round(float(c[m].sum()), 6))
            ws.cell(row, 4, round(float(d[m].sum()), 6))
            if b == 0:
                ws.cell(row, 5, "0:00")
                ws.cell(row, 6, round(soc0, 6))
            if b == 1:
                ws.cell(row, 5, "24:00")
                ws.cell(row, 6, round(soc24, 6))
            row += 1


def _write_emergency_sheet(ws, days, idx, variant, fmt):
    _clear_data(ws)
    row = 2
    for day in days:
        g = idx[(day, variant)].sort_values("slot")
        em = g["emergency_kwh"].to_numpy(float)
        for k, (interval, amount) in enumerate(emergency_ranges(em)):
            cell = ws.cell(row, 1, pd.Timestamp(day).to_pydatetime() if k == 0 else None)
            if k == 0:
                cell.number_format = fmt
            ws.cell(row, 2, interval)
            ws.cell(row, 3, round(amount, 6))
            row += 1


def write_result42(df, template=TEMPLATE_42, target=OUT_42) -> None:
    idx = build_index(df)
    days = sorted(df["date"].unique())
    shutil.copy2(template, target)
    wb = load_workbook(target)
    ws = wb["计划购电量"]
    fmt = _date_fmt(ws)
    _write_plan_sheet(ws, days, idx, "4-2", fmt, "plan")
    _write_storage_sheet(wb["充放电量"], days, idx, "4-2", fmt)
    _write_emergency_sheet(wb["紧急购电量"], days, idx, "4-2", fmt)
    wb.save(target)


def write_result43(df, template=TEMPLATE_43, target=OUT_43) -> None:
    idx = build_index(df)
    days = sorted(df["date"].unique())
    shutil.copy2(template, target)
    wb = load_workbook(target)
    ws = wb["计划购电量"]
    fmt = _date_fmt(ws)
    _write_plan_sheet(ws, days, idx, "4-3", fmt, "plan")   # 0:00 计划 x0 + 计划统计费
    _write_plan_sheet(wb["调整购电量"], days, idx, "4-3", fmt, "q43")  # a + 口径C总费
    _write_storage_sheet(wb["充放电量"], days, idx, "4-3", fmt)
    _write_emergency_sheet(wb["紧急购电量"], days, idx, "4-3", fmt)
    wb.save(target)


# ============================================================ 读回
def _cell_date(v):
    if v is None:
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    return str(v)


def read_plan(path: Path, sheet: str = "计划购电量") -> dict:
    wb = load_workbook(path, data_only=True)
    ws = wb[sheet]
    out = {}
    for r in range(2, ws.max_row + 1):
        d = _cell_date(ws.cell(r, 1).value)
        if d is None:
            continue
        out[d] = {
            "slots": {SOURCE_SLOTS[j]: ws.cell(r, 2 + j).value for j in range(T)},
            "total": ws.cell(r, 146).value,
            "fee": ws.cell(r, 147).value,
        }
    wb.close()
    return out


def read_storage(path: Path) -> dict:
    wb = load_workbook(path, data_only=True)
    ws = wb["充放电量"]
    out = {}
    for r in range(2, ws.max_row + 1):
        d = _cell_date(ws.cell(r, 1).value)
        if d is None:
            continue
        blocks = [(ws.cell(r + b, 3).value, ws.cell(r + b, 4).value)
                  for b in range(6)]
        out[d] = {"blocks": blocks, "soc0": ws.cell(r, 6).value,
                  "soc24": ws.cell(r + 1, 6).value}
    wb.close()
    return out


def read_emergency(path: Path, sheet: str = "紧急购电量") -> dict:
    wb = load_workbook(path, data_only=True)
    ws = wb[sheet]
    out: dict = {}
    cur = None
    for r in range(2, ws.max_row + 1):
        d = _cell_date(ws.cell(r, 1).value)
        if d is not None:
            cur = d
        if cur is None:
            continue
        out.setdefault(cur, []).append((ws.cell(r, 2).value, ws.cell(r, 3).value))
    wb.close()
    return out


def sheet_dims(path: Path) -> dict:
    wb = load_workbook(path)
    dims = {name: (wb[name].max_row, wb[name].max_column) for name in wb.sheetnames}
    wb.close()
    return dims


# ============================================================ 审计
def run_audit(df: pd.DataFrame, out42: Path, out43: Path,
              summary: dict | None) -> list[dict]:
    checks: list[dict] = []
    days = sorted(df["date"].unique())
    df42 = df[df["variant"] == "4-2"].reset_index(drop=True)
    df43 = df[df["variant"] == "4-3"].reset_index(drop=True)
    plan42, plan43 = read_plan(out42), read_plan(out43)
    st42, st43 = read_storage(out42), read_storage(out43)
    em42, em43 = read_emergency(out42), read_emergency(out43)

    # ---- 1) 每天全天量 = 144 段之和（计划 / 充电 / 放电 / 紧急 各一）----
    d_plan = d_chg = d_dis = d_em = 0.0
    for variant, pland, std, emd in (("4-2", plan42, st42, em42),
                                     ("4-3", plan43, st43, em43)):
        sub = df[df["variant"] == variant]
        for day in days:
            g = sub[sub["date"] == day].sort_values("slot")
            d_plan = max(d_plan, abs(float(pland[day]["total"]) - float(g["plan_kwh"].sum())))
            blk_ch = sum(float(b[0]) for b in std[day]["blocks"])
            blk_dis = sum(float(b[1]) for b in std[day]["blocks"])
            d_chg = max(d_chg, abs(blk_ch - float(g["charge_kwh"].sum())))
            d_dis = max(d_dis, abs(blk_dis - float(g["discharge_kwh"].sum())))
            d_em = max(d_em, abs(sum(float(a) for _, a in emd.get(day, []))
                                 - float(g["emergency_kwh"].sum())))
    mx = max(d_plan, d_chg, d_dis, d_em)
    checks.append({
        "id": 1, "name": "daily_total_equals_segment_sum",
        "pass": bool(mx < 1e-4),   # 逐段 round(·,6) 在 144 段上的累积舍入
        "detail": (f"每天全天量 vs 144 段和 最大偏差：计划 {d_plan:.2e}、"
                   f"充电 {d_chg:.2e}、放电 {d_dis:.2e}、紧急 {d_em:.2e} kWh"),
        "values": {"plan": d_plan, "charge": d_chg, "discharge": d_dis,
                   "emergency": d_em}})

    # ---- 2) 用实现价独立复算，与 harness 对账（≤1 元）----
    fee42 = fee_q42(df42["price"], df42["plan_kwh"], df42["emergency_kwh"])
    fee43 = fee_q43(df43["price"], df43["plan_kwh"], df43["adjusted_kwh"],
                    df43["emergency_kwh"])
    ref42 = ref43 = None
    if summary is not None:
        ref42 = (summary.get(REF_KEY_42[0]) or {}).get(REF_KEY_42[1])
        ref43 = (summary.get(REF_KEY_43[0]) or {}).get(REF_KEY_43[1])
    ok42 = ref42 is not None and abs(fee42 - float(ref42)) <= FEE_TOL
    ok43 = ref43 is not None and abs(fee43 - float(ref43)) <= FEE_TOL
    gap42 = None if ref42 is None else fee42 - float(ref42)
    gap43 = None if ref43 is None else fee43 - float(ref43)
    checks.append({
        "id": 2, "name": "fee_recompute_vs_harness",
        "pass": bool(ok42 and ok43),
        "detail": (f"4-2 重算 {fee42:,.2f} 元 vs harness "
                   f"{REF_KEY_42[0]}.{REF_KEY_42[1]}={ref42 if ref42 is None else format(float(ref42), ',.2f')}"
                   f"（差 {gap42 if gap42 is None else format(gap42, '+,.4f')}）；"
                   f"4-3 重算 {fee43:,.2f} 元 vs harness "
                   f"{REF_KEY_43[0]}.{REF_KEY_43[1]}={ref43 if ref43 is None else format(float(ref43), ',.2f')}"
                   f"（差 {gap43 if gap43 is None else format(gap43, '+,.4f')}）"),
        "values": {"q42_recompute": fee42, "q42_ref": ref42, "q42_gap": gap42,
                   "q43_recompute": fee43, "q43_ref": ref43, "q43_gap": gap43,
                   "tol": FEE_TOL}})

    # ---- 3) SOC 全程在 [1200,10800] 且跨日连续 ----
    soc_min, soc_max, max_break = float("inf"), float("-inf"), 0.0
    first_starts = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        prev_end = None
        for day in days:
            g = sub[sub["date"] == day].sort_values("slot")
            soc_end = g["soc_end_kwh"].to_numpy(float)
            c = g["charge_kwh"].to_numpy(float)
            d = g["discharge_kwh"].to_numpy(float)
            soc_start = float(soc_end[0] - ETA * c[0] + d[0] / ETA)
            soc_min = min(soc_min, float(soc_end.min()), soc_start)
            soc_max = max(soc_max, float(soc_end.max()), soc_start)
            if prev_end is not None:
                max_break = max(max_break, abs(soc_start - prev_end))
            else:
                first_starts[variant] = soc_start
            prev_end = float(soc_end[-1])
    in_bounds = (soc_min >= EMIN - 1e-6) and (soc_max <= EMAX + 1e-6)
    continuous = max_break < 1e-3
    checks.append({
        "id": 3, "name": "soc_bounds_and_continuity",
        "pass": bool(in_bounds and continuous),
        "detail": (f"SOC ∈ [{soc_min:.3f}, {soc_max:.3f}] kWh（界 [{EMIN},{EMAX}]）；"
                   f"跨日最大断点 {max_break:.3e} kWh；"
                   f"首日 0:00 储电量 {first_starts}"),
        "values": {"soc_min": soc_min, "soc_max": soc_max,
                   "max_daily_break": max_break, "first_start": first_starts}})

    # ---- 4) 紧急购电与充电不在同一时段同时发生 ----
    both = df[(df["emergency_kwh"] > 1e-6) & (df["charge_kwh"] > 1e-6)]
    checks.append({
        "id": 4, "name": "emergency_charge_exclusive",
        "pass": bool(len(both) == 0),
        "detail": (f"同时段紧急购电且充电的段数 = {len(both)}"
                   + ("" if len(both) == 0 else f"（例：{both[['date','variant','slot']].head(3).to_dict('records')}）")),
        "values": {"conflict_slots": int(len(both))}})

    # ---- 5) 不售电：任何时段 adjusted 不为负 ----
    amin = float(df["adjusted_kwh"].min())
    checks.append({
        "id": 5, "name": "no_negative_adjusted",
        "pass": bool(amin >= -1e-6),
        "detail": f"adjusted_kwh 最小值 = {amin:.6f}（≥0 即不售电）",
        "values": {"adjusted_min": amin}})

    # ---- 6) round-trip：从 xlsx 读回重算，与被写数据一致 ----
    dev = 0.0
    for day in days:
        g = df42[df42["date"] == day].sort_values("slot")
        for slot, v in plan42[day]["slots"].items():
            exp = float(g[g["slot"] == slot]["plan_kwh"].iloc[0])
            dev = max(dev, abs(float(v) - exp))
        g = df43[df43["date"] == day].sort_values("slot")
        for slot, v in plan43[day]["slots"].items():
            exp = float(g[g["slot"] == slot]["plan_kwh"].iloc[0])
            dev = max(dev, abs(float(v) - exp))
        for std, var in ((st42, "4-2"), (st43, "4-3")):
            g = df[(df["variant"] == var) & (df["date"] == day)].sort_values("slot")
            slot_arr = g["slot"].to_numpy(int)
            c = g["charge_kwh"].to_numpy(float)
            for b in range(6):
                lo, hi = BLOCKS[b]
                m = (slot_arr >= lo) & (slot_arr <= hi)
                dev = max(dev, abs(float(std[day]["blocks"][b][0]) - float(c[m].sum())))
    checks.append({
        "id": 6, "name": "xlsx_roundtrip_matches",
        "pass": bool(dev < 1e-6),
        "detail": f"xlsx 读回逐格重算 vs 写入数据 最大偏差 = {dev:.3e}",
        "values": {"max_dev": dev}})

    return checks


# ============================================================ 主流程
def run_pipeline(df, template42=TEMPLATE_42, template43=TEMPLATE_43,
                 out42=OUT_42, out43=OUT_43, audit_path=AUDIT_JSON,
                 summary_path=SUMMARY_JSON) -> dict:
    validate(df)
    write_result42(df, template42, out42)
    write_result43(df, template43, out43)

    summary = None
    summary_path = Path(summary_path)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text("utf-8"))

    checks = run_audit(df, Path(out42), Path(out43), summary)

    df42 = df[df["variant"] == "4-2"]
    df43 = df[df["variant"] == "4-3"]
    fee42 = fee_q42(df42["price"], df42["plan_kwh"], df42["emergency_kwh"])
    fee43 = fee_q43(df43["price"], df43["plan_kwh"], df43["adjusted_kwh"],
                    df43["emergency_kwh"])
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "segments_csv": _relative_to_root(SEGMENTS_CSV),
        "days": int(df["date"].nunique()),
        "rows": int(len(df)),
        "fee_q42_yuan": fee42,
        "fee_q43_yuan": fee43,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
    }
    Path(audit_path).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="问题四全年逐段结果写入 + 独立审计")
    ap.add_argument("--csv", default=str(SEGMENTS_CSV))
    ap.add_argument("--summary", default=str(SUMMARY_JSON))
    ap.add_argument("--out42", default=str(OUT_42))
    ap.add_argument("--out43", default=str(OUT_43))
    ap.add_argument("--audit", default=str(AUDIT_JSON))
    args = ap.parse_args(argv)

    df = load_segments(Path(args.csv))
    audit = run_pipeline(df, out42=Path(args.out42), out43=Path(args.out43),
                         audit_path=Path(args.audit),
                         summary_path=Path(args.summary))

    d42 = sheet_dims(Path(args.out42))
    d43 = sheet_dims(Path(args.out43))
    print("=" * 72)
    print("问题四全年逐段结果写入 + 独立审计")
    print("=" * 72)
    print(f"输入：{args.csv}  （{audit['rows']} 行 = {audit['days']} 天 × 144 段 × 2 分支）")
    print(f"\nresult4-2.xlsx 各表 [行×列]：")
    for name, (r, c) in d42.items():
        print(f"    {name:<8} {r:>6} × {c}")
    print(f"result4-3.xlsx 各表 [行×列]：")
    for name, (r, c) in d43.items():
        print(f"    {name:<8} {r:>6} × {c}")
    print(f"\n4-2 全年费用 = {audit['fee_q42_yuan']:,.2f} 元")
    print(f"4-3 全年费用 = {audit['fee_q43_yuan']:,.2f} 元")
    print(f"\n审计（写入 {args.audit}）：")
    for chk in audit["checks"]:
        tag = "PASS" if chk["pass"] else "FAIL"
        print(f"  [{tag}] {chk['id']}. {chk['name']}: {chk['detail']}")
    print(f"\n结论：{'全部通过' if audit['all_pass'] else '存在未通过项'}")
    if not audit["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
