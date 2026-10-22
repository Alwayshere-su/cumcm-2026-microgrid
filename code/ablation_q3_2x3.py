# -*- coding: utf-8 -*-
"""问题三 2×3 费用消融：决策结构 × 预测器，选型最终判据 = 钱（口径C settlement_c）。

行（决策结构）：
  DET  确定性滚动 MPC(6/12/18) + 因果执行   —— solve_q3 现状结构
  SP   情景 SP + DP 价值执行器，四时刻(0/6/12/18) 滚动

列（预测器，均非 ML、因果）：
  C1  统计均值负荷 + 附件3原值光伏
  C2  岭回归负荷   + 附件3原值光伏
  C3  岭回归负荷   + 偏差校正附件3×岭回归PV 融合（含交叉项最小二乘滚动定权）

每格：全年费用 / 紧急购电量 / CVaR95 / 最坏日费用。
情景池 = 同日同发布时刻配对残差（负荷、光伏取自同一历史日）。

用法：
  python scripts/ablation_q3_2x3.py --days 6          # 冒烟
  python scripts/ablation_q3_2x3.py                   # 全年
  python scripts/ablation_q3_2x3.py --skip-det        # 只跑 SP 行
  python scripts/ablation_q3_2x3.py --skip-sp         # 只跑 DET 行
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()

import solve_q3 as q3  # noqa: E402
from solve_q2_twolayer import stat_forecast, M_SCEN  # noqa: E402
from probe_q3_rolling_sp import (two_stage_plan_q3,  # noqa: E402
                                 future_value_curves_h, retention_levels_h)

T = 144
UPDATES = (6, 12, 18)      # DET 行：0:00 初始计划 + 6/12/18 修订
ISSUES = (0, 6, 12, 18)    # SP 行：四时刻


def cell_metrics(out: pd.DataFrame, daily: pd.DataFrame) -> dict:
    dc = daily["total"].to_numpy(float)
    return {"total": float(dc.sum()),
            "emergency_cost": float(daily["emergency_cost"].sum()),
            "emergency_kwh": float(out["emergency_kwh"].sum()),
            "cvar95": float(dc[dc >= np.quantile(dc, .95)].mean()),
            "max_day": float(dc.max())}


# --------------------------------------------------------------------------- #
def build_panels(data, official, reg, dates):
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    stat_load = {}
    for day in dates:
        try:
            lh, _ = stat_forecast(by_day, day)
            stat_load[day] = lh
        except ValueError:
            continue
    reg_by_day = {d: x.sort_values("slot") for d, x in reg.groupby("date")}

    def make(load_dict, pv_col):
        rows = []
        for day in dates:
            if day not in load_dict:
                continue
            lh = load_dict[day]
            pv = reg_by_day[day][pv_col].to_numpy(float)
            for s in range(T):
                rows.append((day, s + 1, lh[s], pv[s]))
        return pd.DataFrame(rows, columns=["date", "slot", "lgb_load_kwh", "lgb_pv_kwh"])

    stat_panel = make(stat_load, "naive_pv_kwh")
    reg_panel = make({d: reg_by_day[d].reg_load_kwh.to_numpy(float) for d in dates},
                     "reg_pv_kwh")
    return by_day, stat_panel, reg_panel


def precompute_pv(data, official, reg_panel, dates, by_day):
    anchor, fused = {}, {}
    for day in sorted(by_day):
        for k in ISSUES:
            try:
                anchor[(day, k)] = q3._official_current_day(official, by_day[day], day, k)
            except ValueError:
                pass
    for day in dates:
        for k in ISSUES:
            if (day, k) not in anchor:
                continue
            try:
                fused[(day, k)], _ = q3.causal_fused_pv(data, reg_panel, official, day, k)
            except ValueError:
                pass
    return anchor, fused


# --------------------------------------------------------------------------- #
def build_sp_library(by_day, reg, anchor, fused, load_src, pv_src):
    load_hat, pv_hat, resid_load, resid_pv = {}, {}, {}, {}
    for day in sorted(by_day):
        try:
            if load_src == "stat":
                lh, _ = stat_forecast(by_day, day)
            else:
                r = reg[reg.date == day].sort_values("slot")
                if len(r) != T:
                    continue
                lh = r.reg_load_kwh.to_numpy(float)
        except (ValueError, KeyError):
            continue
        load_hat[day] = lh
        resid_load[day] = by_day[day].load_kwh.to_numpy(float) - lh
    for (day, k) in anchor:
        if day not in load_hat:
            continue
        ph = anchor[(day, k)] if pv_src == "anchor" else fused.get((day, k))
        if ph is None:
            continue
        s = k * 6
        pv_hat[(day, k)] = ph
        resid_pv[(day, k)] = by_day[day].pv_kwh.to_numpy(float)[s:] - ph
    return load_hat, pv_hat, resid_load, resid_pv


def run_sp(by_day, prm, dates, load_hat, pv_hat, resid_load, resid_pv, M: int = M_SCEN):
    eta_c = prm["charge_efficiency"]; eta_d = prm["discharge_efficiency"]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    emin = prm["operating_min_kwh"]; emax = prm["operating_max_kwh"]
    days_all = sorted(load_hat)
    energy = prm["initial_energy_kwh"]
    rows, daily_rows = [], []
    t0 = time.perf_counter()
    n_dates = len(dates)
    for i, day in enumerate(dates):
        if day not in load_hat:
            continue
        actual = by_day[day]
        price = actual.price_fixed.to_numpy(float)
        load_a = actual.load_kwh.to_numpy(float)
        pv_a = actual.pv_kwh.to_numpy(float)
        lib = [d for d in days_all if d < day][-M:]
        a_full, x0 = None, None
        eme_full = np.zeros(T)
        e = energy
        for k in ISSUES:
            s = k * 6
            if (day, k) not in pv_hat:
                continue
            lib_p = [d for d in lib if d in resid_load and (d, k) in resid_pv]
            if lib_p:
                L = np.array([np.maximum(load_hat[day][s:] + resid_load[d][s:], 0.0)
                              for d in lib_p])
                V = np.array([np.maximum(pv_hat[(day, k)] + resid_pv[(d, k)], 0.0)
                              for d in lib_p])
            else:
                L = np.array([np.maximum(load_hat[day][s:], 0.0)])
                V = np.array([np.maximum(pv_hat[(day, k)], 0.0)])
            ref = None if x0 is None else x0[s:]
            a_seg = two_stage_plan_q3(price[s:], L, V, e, ref, prm)
            if a_full is None:
                a_full = np.zeros(T)
            a_full[s:] = a_seg
            if x0 is None:
                x0 = a_full.copy()
            R = retention_levels_h(
                future_value_curves_h((L - V) - a_seg[None, :], price[s:], prm),
                price[s:], prm)
            nxt = [x for x in ISSUES if x > k]
            stop = min(nxt[0] * 6, T) if nxt else T
            for t in range(s, stop):
                r = (load_a[t] - pv_a[t]) - a_full[t]
                if r > 0:
                    dis = min(r, S, eta_d * max(e - R[t - s], 0.0))
                    chg = sur = 0.0
                    eme = r - dis
                else:
                    chg = min(-r, S, (emax - e) / eta_c)
                    dis = eme = 0.0
                    sur = -r - chg
                e_next = e + eta_c * chg - dis / eta_d
                rows.append((day, t + 1, a_full[t], chg, dis, sur, eme, e, e_next, price[t]))
                eme_full[t] = eme
                e = e_next
        energy = e
        sett = q3.settlement_c(x0, a_full, price, eme_full)
        daily_rows.append({"date": day, "total": sett["total"],
                           "emergency_cost": sett["emergency_cost"]})
        if (i + 1) % 50 == 0 or i == n_dates - 1:
            print(f"    SP {i + 1}/{n_dates} {day.date()} {time.perf_counter() - t0:.1f}s",
                  flush=True)
    out = pd.DataFrame(rows, columns=["date", "slot", "buy_kwh", "charge_kwh",
                                      "discharge_kwh", "surplus_kwh", "emergency_kwh",
                                      "energy_start_kwh", "energy_end_kwh", "price"])
    daily = pd.DataFrame(daily_rows)
    return out, daily


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--skip-det", action="store_true")
    ap.add_argument("--skip-sp", action="store_true")
    args = ap.parse_args()

    data = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                       parse_dates=["date", "timestamp_end"])
    reg = pd.read_csv(ROOT / "data_processed/q2_forecasts_reg.csv", parse_dates=["date"])
    official = pd.read_csv(ROOT / "data_processed/pv_forecast_hourly.csv",
                           parse_dates=["issue_at", "target_end"])
    prm = json.loads((ROOT / "data_processed/q1_storage_parameters.json")
                     .read_text(encoding="utf-8"))
    dates = pd.date_range("2025-02-01", "2025-12-31")
    if args.days:
        dates = dates[:args.days]

    by_day, stat_panel, reg_panel = build_panels(data, official, reg, dates)
    print("预计算光伏预报（锚点 + 校正融合）...", flush=True)
    anchor, fused = precompute_pv(data, official, reg_panel, dates, by_day)

    # 列定义：panel / pv_mode（DET 行），load_src / pv_src（SP 行）
    cols = [
        ("C1 统计+附件3原值", stat_panel, "anchor", "stat", "anchor"),
        ("C2 岭回归+附件3原值", reg_panel, "anchor", "reg", "anchor"),
        ("C3 岭回归+校正融合", reg_panel, "fused", "reg", "fused"),
    ]
    result = {}

    if not args.skip_det:
        for name, panel, pv_mode, _, _ in cols:
            t0 = time.perf_counter()
            out, daily = q3.run_q3_period(data, panel, official, dates,
                                          prm["initial_energy_kwh"], prm, UPDATES, pv_mode)
            m = cell_metrics(out, daily)
            result[("DET", name)] = m
            print(f"DET × {name}: total={m['total']:,.0f} 紧急电量={m['emergency_kwh']:,.0f} "
                  f"CVaR95={m['cvar95']:,.0f} 最坏日={m['max_day']:,.0f} "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)

    if not args.skip_sp:
        for name, _, _, load_src, pv_src in cols:
            t0 = time.perf_counter()
            lh, ph, rl, rp = build_sp_library(by_day, reg, anchor, fused, load_src, pv_src)
            out, daily = run_sp(by_day, prm, dates, lh, ph, rl, rp)
            m = cell_metrics(out, daily)
            result[("SP", name)] = m
            print(f"SP  × {name}: total={m['total']:,.0f} 紧急电量={m['emergency_kwh']:,.0f} "
                  f"CVaR95={m['cvar95']:,.0f} 最坏日={m['max_day']:,.0f} "
                  f"[{time.perf_counter() - t0:.1f}s]", flush=True)

    # 输出
    rows = ["DET", "SP"]
    colnames = [c[0] for c in cols]
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/q3_ablation_2x3.json", "w", encoding="utf-8") as f:
        json.dump({f"{r}|{c}": result[(r, c)] for r in rows for c in colnames},
                  f, ensure_ascii=False, indent=2)

    def tbl(metric, title, wan=False):
        L = [f"## {title}", "",
             "| 决策结构 | " + " | ".join(colnames) + " |",
             "| --- | " + " | ".join(["---:"] * len(colnames)) + " |"]
        for r in rows:
            cells = []
            for c in colnames:
                v = result[(r, c)][metric]
                cells.append(f"{v / 1e4:.2f} 万" if wan else f"{v:,.0f}")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
        return "\n".join(L) + "\n"

    md = ["# 问题三 2×3 费用消融（口径C）", "",
          f"评价期 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 天）；情景数 M={M_SCEN}；",
          "配对残差：同日同发布时刻抽取。", ""]
    md += [tbl("total", "全年总费用（万元）", wan=True),
           tbl("emergency_kwh", "紧急购电量（kWh）"),
           tbl("cvar95", "CVaR95 日均费用（元）"),
           tbl("max_day", "最坏日费用（元）")]
    (ROOT / "results/q3_ablation_2x3.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    recs = [{"row": r, "col": c, **result[(r, c)]} for r in rows for c in colnames]
    pd.DataFrame(recs).to_csv(ROOT / "results/q3_ablation_2x3.csv",
                              index=False, encoding="utf-8-sig")
    print("\n" + "\n".join(md))
    print("已写出 results/q3_ablation_2x3.{md,json,csv}")


if __name__ == "__main__":
    main()
