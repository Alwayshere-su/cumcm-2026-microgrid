#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题四"四个节气日"论文用报告生成器。

从 ``results/q4_segments.csv`` 抽取四个指定日期
（2025-03-20 春分 / 2025-06-21 夏至 / 2025-09-23 秋分 / 2025-12-21 冬至）的
逐段调度明细，按两套结算口径聚合，产出：

  * ``results/q4_dates_report.md``     —— 论文用报告（每个日期 4-3 与 4-2 两张表）；
  * ``results/q4_dates_segments.csv``  —— 四天 × 144 段 × 2 variant 的逐段明细。

结算口径（与 ``scripts/q4_models.py::settlement_q42 / settlement_q43`` 完全一致）：
  * 4-2：账单 = Σ_t p·计划量 + 5·Σ_t p·紧急购电量；计划量冻结、无差额项。
  * 4-3：口径 C，账单 = Σ_t p·min(x0,a)          （计划内）
                       + 0.5·Σ_t p·(x0−a)⁺       （少购 u）
                       + 1.5·Σ_t p·(a−x0)⁺       （超购 v）
                       + 5·Σ_t p·g               （紧急 g）
    其中 x0=计划量(plan)、a=调整量(adjusted)、p=实现电价、g=紧急购电量。

CSV 输入契约（列严格为）：
  date,slot,variant,price,load_kwh,pv_kwh,plan_kwh,adjusted_kwh,
  charge_kwh,discharge_kwh,emergency_kwh,soc_end_kwh
  variant ∈ {"4-2","4-3"}；slot 1..144（每段 10 分钟）。

用法：
  python scripts/report_q4_dates.py               # 读真实 results/q4_segments.csv
  python scripts/report_q4_dates.py --synthetic   # 用随机数据自测脚本逻辑（无需真实文件）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RES = ROOT / "results"

DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
VARIANTS = ["4-3", "4-2"]
COLS = ["date", "slot", "variant", "price", "load_kwh", "pv_kwh", "plan_kwh",
        "adjusted_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh",
        "soc_end_kwh"]

# 与 scripts/q4_models.py 一致的储能参数（仅用于由首段 soc_end 反推 0:00 储电量）
ETA, EMIN, EMAX = 0.9, 1200.0, 10800.0
BMAX = 5000.0 / 6.0
E0 = 6000.0
BLOCKS = [("00-04", 1, 24), ("04-08", 25, 48), ("08-12", 49, 72),
          ("12-16", 73, 96), ("16-20", 97, 120), ("20-24", 121, 144)]


# --------------------------------------------------------------------- 输入
def gen_synthetic(seed: int = 20250320) -> pd.DataFrame:
    """生成符合契约的随机数据，用于在真实 CSV 未就绪时自测脚本逻辑。"""
    rng = np.random.default_rng(seed)
    slot = np.arange(1, 145)
    hour = (slot - 1) / 6.0                      # 10min -> 小时
    rows: list[dict] = []
    for date in DATES:
        for variant in VARIANTS:
            base = (0.30 + 0.28 * np.sin((hour - 6) / 24 * 2 * np.pi)
                    + 0.10 * rng.standard_normal(144))
            price = np.clip(base, 0.05, 1.20)
            load = np.clip(900 + 320 * np.sin((hour - 8) / 24 * 2 * np.pi)
                           + 150 * rng.standard_normal(144), 50, None)
            pv = np.clip(1500 * np.sin(np.clip((hour - 6) / 12, 0, 1) * np.pi)
                         + 120 * rng.standard_normal(144), 0, None)
            net = load - pv
            plan = np.clip(net + 120 * rng.standard_normal(144), 0, None)
            if variant == "4-3":
                adjusted = np.clip(plan + 180 * rng.standard_normal(144),
                                   0, None)
                emergency = np.zeros(144)       # 4-3 常规购电无上限，紧急不触发
            else:
                adjusted = plan.copy()          # 4-2 计划量冻结，无差额项
                emergency = np.clip(net - plan, 0, None) * rng.uniform(0, .6, 144)
            charge = np.clip((price < np.quantile(price, 0.4))
                             * rng.uniform(0, 350, 144), 0, None)
            discharge = np.clip((price > np.quantile(price, 0.5))
                                * rng.uniform(0, 350, 144), 0, None)
            e = float(E0)
            for t in range(144):
                ch, dis = float(charge[t]), float(discharge[t])
                if e + ETA * ch > EMAX:                      # 充电触顶
                    ch = max((EMAX - e) / ETA, 0.0)
                e1 = e + ETA * ch
                if e1 - dis / ETA < EMIN:                    # 放电触底
                    dis = max((e1 - EMIN) * ETA, 0.0)
                e = e1 - dis / ETA
                charge[t], discharge[t] = ch, dis
                rows.append({
                    "date": date, "slot": int(slot[t]), "variant": variant,
                    "price": float(price[t]), "load_kwh": float(load[t]),
                    "pv_kwh": float(pv[t]), "plan_kwh": float(plan[t]),
                    "adjusted_kwh": float(adjusted[t]),
                    "charge_kwh": ch, "discharge_kwh": dis,
                    "emergency_kwh": float(emergency[t]), "soc_end_kwh": e,
                })
    return pd.DataFrame(rows)[COLS]


def load_real(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(
            f"[错误] 找不到输入文件：{csv_path}\n"
            f"        该文件应由问题四逐段求解流程产出（列：{','.join(COLS)}）。\n"
            f"        若只想自测本脚本逻辑，请改用："
            f"python scripts/report_q4_dates.py --synthetic\n"
            f"        （脚本不会伪造/硬编码任何费用数据。）")
    df = pd.read_csv(csv_path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"[错误] {csv_path} 缺少列 {missing}；期望列为 {COLS}")
    return df[COLS].copy()


# ----------------------------------------------------------------- 单日聚合
def _price_cols(df: pd.DataFrame):
    p = df["price"].to_numpy(float)
    plan = df["plan_kwh"].to_numpy(float)
    adjusted = df["adjusted_kwh"].to_numpy(float)
    g = df["emergency_kwh"].to_numpy(float)
    return p, plan, adjusted, g


def day_43(d: pd.DataFrame) -> dict:
    d = d.sort_values("slot")
    p, plan, adjusted, g = _price_cols(d)
    u = np.maximum(plan - adjusted, 0.0)         # 少购 (x0 - a)+
    v = np.maximum(adjusted - plan, 0.0)         # 超购 (a - x0)+
    within = np.minimum(plan, adjusted)          # 计划内
    within_cost = float((p * within).sum())
    under_cost = float((0.5 * p * u).sum())
    over_cost = float((1.5 * p * v).sum())
    em_cost = float((5.0 * p * g).sum())
    total = within_cost + under_cost + over_cost + em_cost

    charge = d["charge_kwh"].to_numpy(float)
    discharge = d["discharge_kwh"].to_numpy(float)
    slot = d["slot"].to_numpy(int)
    blocks = []
    for name, lo, hi in BLOCKS:
        m = (slot >= lo) & (slot <= hi)
        blocks.append((name, float(charge[m].sum()), float(discharge[m].sum())))
    soc_end = d["soc_end_kwh"].to_numpy(float)
    soc_0 = float(soc_end[0] - ETA * charge[0] + discharge[0] / ETA)  # 反推 0:00
    soc_24 = float(soc_end[-1])                                        # 24:00
    return {
        "plan_kwh": float(plan.sum()), "adjusted_kwh": float(adjusted.sum()),
        "u_kwh": float(u.sum()), "v_kwh": float(v.sum()),
        "em_kwh": float(g.sum()),
        "within_cost": within_cost, "under_cost": under_cost,
        "over_cost": over_cost, "em_cost": em_cost, "total": total,
        "blocks": blocks, "soc_0": soc_0, "soc_24": soc_24,
    }


def day_42(d: pd.DataFrame) -> dict:
    d = d.sort_values("slot")
    p, plan, _adjusted, g = _price_cols(d)
    plan_cost = float((p * plan).sum())
    em_cost = float((5.0 * p * g).sum())
    return {
        "plan_kwh": float(plan.sum()), "em_kwh": float(g.sum()),
        "plan_cost": plan_cost, "em_cost": em_cost,
        "total": plan_cost + em_cost,
    }


# --------------------------------------------------------------------- 报告
def build_report(df: pd.DataFrame, summary: dict | None, use_synth: bool,
                 missing_dates: list[str]) -> tuple[str, dict]:
    data = df.copy()
    data["_d"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    agg = {"4-3": {k: 0.0 for k in
                   ("plan_kwh", "adjusted_kwh", "u_kwh", "v_kwh", "em_kwh",
                    "within_cost", "under_cost", "over_cost", "em_cost",
                    "total")},
           "4-2": {k: 0.0 for k in
                   ("plan_kwh", "em_kwh", "plan_cost", "em_cost", "total")}}

    L: list[str] = []
    A = L.append
    A("# 问题四 四个节气日逐段费用报告\n")
    A(f"> 数据来源：`results/q4_segments.csv`"
      f"{'（**合成数据·仅用于自测**）' if use_synth else ''}；"
      f"日期：{'、'.join(DATES)}；每段 10 分钟、每天 144 段。\n")
    A("> **结算口径**（与 `scripts/q4_models.py` 一致）：")
    A("> - **4-3 = 口径 C**：账单 = Σ p·min(x0,a) + 0.5·Σ p·(x0−a)⁺ + "
      "1.5·Σ p·(a−x0)⁺ + 5·Σ p·g，其中 x0=计划量、a=调整量、p=实现电价、"
      "g=确认计划之外、为维持实时能量平衡而触发的紧急购电量。")
    A("> - **4-2 = 计划费 + 5×紧急费**：账单 = Σ p·x0 + 5·Σ p·g，"
      "计划量冻结、无差额项。\n")
    if missing_dates:
        A(f"> ⚠️ 以下日期在输入中缺失，已跳过（未伪造）：{missing_dates}\n")

    # ---- 逐日明细 ----
    for date in DATES:
        sub = data[data["_d"] == date]
        if sub.empty:
            continue
        A(f"## {date}\n")
        # 4-3
        d43 = sub[sub["variant"] == "4-3"]
        if not d43.empty:
            r = day_43(d43)
            for k, v in r.items():
                if k in agg["4-3"]:
                    agg["4-3"][k] += v
            A(f"### {date} · 4-3（口径 C）\n")
            A("| 指标 | 数值 |")
            A("| --- | ---: |")
            A(f"| 计划量 x0（kWh） | {r['plan_kwh']:,.1f} |")
            A(f"| 调整量 a（kWh） | {r['adjusted_kwh']:,.1f} |")
            A(f"| 少购量 u=(x0−a)⁺（kWh） | {r['u_kwh']:,.1f} |")
            A(f"| 超购量 v=(a−x0)⁺（kWh） | {r['v_kwh']:,.1f} |")
            A(f"| 紧急购电量 g（kWh） | {r['em_kwh']:,.1f} |")
            A(f"| 计划内费 Σp·min(x0,a)（元） | {r['within_cost']:,.2f} |")
            A(f"| 少购费 0.5·Σp·u（元） | {r['under_cost']:,.2f} |")
            A(f"| 超购费 1.5·Σp·v（元） | {r['over_cost']:,.2f} |")
            A(f"| 紧急费 5·Σp·g（元） | {r['em_cost']:,.2f} |")
            A(f"| **全天费用（元）** | **{r['total']:,.2f}** |")
            A(f"| 0:00 储电量（kWh） | {r['soc_0']:,.1f} |")
            A(f"| 24:00 储电量（kWh） | {r['soc_24']:,.1f} |")
            A("")
            A("| 4h 区间 | 充电量（kWh） | 放电量（kWh） |")
            A("| --- | ---: | ---: |")
            for name, ch, dis in r["blocks"]:
                A(f"| {name} | {ch:,.1f} | {dis:,.1f} |")
            A("")
        # 4-2
        d42 = sub[sub["variant"] == "4-2"]
        if not d42.empty:
            r = day_42(d42)
            for k, v in r.items():
                if k in agg["4-2"]:
                    agg["4-2"][k] += v
            A(f"### {date} · 4-2（计划费 + 5×紧急费）\n")
            A("| 指标 | 数值 |")
            A("| --- | ---: |")
            A(f"| 计划量（kWh） | {r['plan_kwh']:,.1f} |")
            A(f"| 紧急购电量（kWh） | {r['em_kwh']:,.1f} |")
            A(f"| 计划费 Σp·x0（元） | {r['plan_cost']:,.2f} |")
            A(f"| 紧急费 5·Σp·g（元） | {r['em_cost']:,.2f} |")
            A(f"| **全天费用（元）** | **{r['total']:,.2f}** |")
            A("")

    # ---- 四天合计与占比 ----
    tot43 = agg["4-3"]["total"]
    tot42 = agg["4-2"]["total"]
    A("## 四天合计\n")
    A(f"- **4-3 四天合计费用：{tot43:,.2f} 元**"
      f"（计划内 {agg['4-3']['within_cost']:,.2f} + "
      f"少购 {agg['4-3']['under_cost']:,.2f} + "
      f"超购 {agg['4-3']['over_cost']:,.2f} + "
      f"紧急 {agg['4-3']['em_cost']:,.2f}）。")
    A(f"- **4-2 四天合计费用：{tot42:,.2f} 元**"
      f"（计划费 {agg['4-2']['plan_cost']:,.2f} + "
      f"紧急费 {agg['4-2']['em_cost']:,.2f}）。\n")

    if summary is None:
        A("> （未找到 `results/harness_q4_summary.json`，无法计算全年占比。）\n")
    else:
        ty = summary.get("total_yuan", {}) or {}
        ty42 = summary.get("total_yuan_42", {}) or {}
        A("### 占全年汇总比例\n")
        A("参考 `results/harness_q4_summary.json`（全年 "
          f"{summary.get('days', '?')} 天结算汇总）。\n")
        if ty:
            A("**4-3 四天合计 vs 全年各口径**")
            A("")
            A("| 全年口径 total_yuan | 全年费用 / 元 | 四天 / 全年 |")
            A("| --- | ---: | ---: |")
            for k, v in ty.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    A(f"| {k} | （未提供） | — |")
                    continue
                A(f"| {k} | {v:,.2f} | {tot43 / v * 100:.4f}% |")
            A("")
        if ty42:
            A("**4-2 四天合计 vs 全年各口径**")
            A("")
            A("| 全年口径 total_yuan_42 | 全年费用 / 元 | 四天 / 全年 |")
            A("| --- | ---: | ---: |")
            for k, v in ty42.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    A(f"| {k} | （未提供） | — |")
                    continue
                A(f"| {k} | {v:,.2f} | {tot42 / v * 100:.4f}% |")
            A("")
    return "\n".join(L), agg


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="问题四四个节气日费用报告生成器")
    ap.add_argument("--csv", default=str(RES / "q4_segments.csv"),
                    help="输入逐段明细 CSV（默认 results/q4_segments.csv）")
    ap.add_argument("--summary", default=str(RES / "harness_q4_summary.json"),
                    help="全年汇总 JSON（用于计算占比）")
    ap.add_argument("--out-dir", default=str(RES), help="输出目录（默认 results/）")
    ap.add_argument("--synthetic", action="store_true",
                    help="用随机数据自测脚本逻辑（真实 CSV 未就绪时使用）")
    args = ap.parse_args()

    use_synth = args.synthetic
    if use_synth:
        df = gen_synthetic()
        print("[synthetic] 已生成合成数据（随机 seed=20250320），仅用于自测。")
    else:
        df = load_real(Path(args.csv))

    # 校验
    bad = sorted(set(df["variant"].unique()) - set(VARIANTS))
    if bad:
        sys.exit(f"[错误] variant 含未预期取值 {bad}；期望 {VARIANTS}")
    if not use_synth:
        seen = set(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))
        missing = [d for d in DATES if d not in seen]
    else:
        missing = []

    # 写出四天逐段明细
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = df[pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").isin(DATES)].copy()
    seg_path = out_dir / "q4_dates_segments.csv"
    sel[COLS].sort_values(["date", "variant", "slot"]).to_csv(
        seg_path, index=False, encoding="utf-8-sig")

    # 读全年汇总
    summary = None
    sp = Path(args.summary)
    if sp.exists():
        summary = json.loads(sp.read_text("utf-8"))
    else:
        print(f"[warn] 未找到 {sp}，报告将不含全年占比。")

    md, agg = build_report(df, summary, use_synth, missing)
    rep_path = out_dir / "q4_dates_report.md"
    rep_path.write_text(md, encoding="utf-8")

    # ---- 控制台摘要 ----
    print(f"已写出 {rep_path}")
    print(f"已写出 {seg_path}（{len(sel)} 行 = 4 天 × 144 段 × 2 variant）")
    if missing:
        print(f"[warn] 缺失日期：{missing}")
    print("\n四天费用摘要（元）：")
    print(f"  4-3 合计 = {agg['4-3']['total']:,.2f}"
          f"（计划内 {agg['4-3']['within_cost']:,.2f} + "
          f"少购 {agg['4-3']['under_cost']:,.2f} + "
          f"超购 {agg['4-3']['over_cost']:,.2f} + "
          f"紧急 {agg['4-3']['em_cost']:,.2f}）")
    print(f"  4-2 合计 = {agg['4-2']['total']:,.2f}"
          f"（计划费 {agg['4-2']['plan_cost']:,.2f} + "
          f"紧急费 {agg['4-2']['em_cost']:,.2f}）")
    if summary:
        ty = summary.get("total_yuan", {}) or {}
        ty42 = summary.get("total_yuan_42", {}) or {}
        if ty:
            k0 = next(iter(ty))
            v0 = ty[k0]
            if v0 and not (isinstance(v0, float) and np.isnan(v0)):
                print(f"  4-3 占全年({k0}) = {agg['4-3']['total'] / v0 * 100:.4f}%")
        if ty42:
            k1 = next(iter(ty42))
            v1 = ty42[k1]
            if v1 and not (isinstance(v1, float) and np.isnan(v1)):
                print(f"  4-2 占全年({k1}) = {agg['4-2']['total'] / v1 * 100:.4f}%")


if __name__ == "__main__":
    main()
