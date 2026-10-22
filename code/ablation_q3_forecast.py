# -*- coding: utf-8 -*-
"""问题三 2×3 消融：预测来源 × 决策结构（全程不使用机器学习）。

行（决策结构）：
  det = 确定性滚动 MPC(6/12/18) + 贪心自适应执行（"模型2"）
  sp  = 情景两阶段 SP(0/6/12/18) + DP 价值执行器（"模型1"）

列（预测来源）：
  stat_anchor  : 负荷 = 星期几×时段稳健均值；光伏 = 附件3 原始整点锚点插值
  ridge_anchor : 负荷 = 岭回归；           光伏 = 附件3 原始整点锚点插值
  ridge_fused  : 负荷 = 岭回归；           光伏 = 附件3 偏差校正 × 岭回归 动态融合

统一口径（所有格子完全一致）：
  回测窗 2025-02-01 ~ 12-31（跨日连续，初值 6000 kWh）
  情景数 M=30；情景日期池 = 三列共用的最近 30 个可用历史日（源荷同日配对）
  结算 = 口径C：p·min(x0,a) + 0.5p·(x0-a)+ + 1.5p·(a-x0)+ + 5p·g
  调整费的参照量恒为 0:00 原计划 x0（不随中间调整滚动，避免重复计费）
  非前视：每个决策时刻只用此前历史与当次发布的附件3预报，不读当天未来实测

用法：
  python scripts/ablation_q3_forecast.py --model sp --forecast ridge_fused --days 3 --scenarios 8
  python scripts/ablation_q3_forecast.py --model det --forecast ridge_fused --terminal soft
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
from probe_q3_rolling_sp import (future_value_curves_h,  # noqa: E402
                                 retention_levels_h, two_stage_plan_q3)
from solve_q2_twolayer import V_CONT  # noqa: E402

T = 144
ISSUES = (0, 6, 12, 18)
RESULTS = ROOT / "results"
STAT_PANEL = ROOT / "data_processed/q3_forecasts_noml.csv"
RIDGE_PANEL = ROOT / "data_processed/q2_forecasts_reg_ext.csv"
DISPATCH_COLS = ["date", "slot", "original_plan_kwh", "buy_kwh", "charge_kwh",
                 "discharge_kwh", "surplus_kwh", "emergency_kwh",
                 "energy_start_kwh", "energy_end_kwh", "load_kwh", "pv_kwh", "price"]
_ORIG_SOLVE_MPC = q3.solve_mpc


# --------------------------------------------------------------------------- #
# 模型2 的执行器：申报量固定，储能按真实净负荷自适应充放，缺口先放电池到物理下限
# --------------------------------------------------------------------------- #
def greedy_segment(plan, load, pv, price, energy0, prm):
    rows, energy = [], float(energy0)
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    ec, ed = prm["charge_efficiency"], prm["discharge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    for t in range(len(load)):
        buy = max(0.0, float(plan["buy"][t]))
        net = buy + float(pv[t]) - float(load[t])
        chg = dis = sur = eme = 0.0
        if net >= 0.0:
            chg = min(net, S, max(emax - energy, 0.0) / ec)
            sur = net - chg
        else:
            dis = min(-net, S, ed * max(energy - emin, 0.0))
            eme = -net - dis
        end = energy + ec * chg - dis / ed
        rows.append((t + 1, buy, chg, dis, sur, eme, energy, end,
                     float(load[t]), float(pv[t]), float(price[t])))
        energy = end
    return pd.DataFrame(rows, columns=["slot", "buy_kwh", "charge_kwh", "discharge_kwh",
                                       "surplus_kwh", "emergency_kwh", "energy_start_kwh",
                                       "energy_end_kwh", "load_kwh", "pv_kwh", "price"])


# --------------------------------------------------------------------------- #
# 三套预测的（非前视）预报库与配对残差库
# --------------------------------------------------------------------------- #
def _panel_frames(panel: pd.DataFrame, col: str) -> dict:
    out = {}
    for d, g in panel.groupby("date"):
        g = g.sort_values("slot")
        if len(g) == T:
            out[pd.Timestamp(d)] = g[col].to_numpy(float)
    return out


def build_library(load_src: str, pv_src: str, data: pd.DataFrame,
                  official: pd.DataFrame, stat: pd.DataFrame, ridge: pd.DataFrame,
                  verbose: bool = True, pv_weight: str = "ls"):
    """返回 dict：load_hat / pv_hat / resid_load / resid_pv / lib_pool。全部非前视。"""
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    stat_load = _panel_frames(stat, "lgb_load_kwh")
    ridge_load = _panel_frames(ridge, "lgb_load_kwh")
    ridge_pv = _panel_frames(ridge, "lgb_pv_kwh")
    load_hat = stat_load if load_src == "stat" else ridge_load

    anchor: dict[tuple, np.ndarray] = {}          # 附件3 原始整点锚点插值
    for d, frame in by_day.items():
        for k in ISSUES:
            try:
                anchor[(d, k)] = q3._official_current_day(official, frame, d, k)
            except ValueError:
                continue

    pv_hat = dict(anchor)
    n_fused = 0
    if pv_src == "fused":
        panel_days = sorted(ridge_load)
        for d in panel_days:
            for k in ISSUES:
                if (d, k) not in anchor or d not in ridge_pv:
                    continue
                s = 6 * k
                prior = [p for p in panel_days if p < d][-28:]
                prior = [p for p in prior if (p, k) in anchor and p in ridge_pv]
                if not prior:
                    pv_hat[(d, k)] = ridge_pv[d][s:]
                    continue
                off_err = np.vstack([anchor[(p, k)] - by_day[p].pv_kwh.to_numpy(float)[s:]
                                     for p in prior])
                bias = np.median(off_err, axis=0)
                reg_err = np.vstack([ridge_pv[p][s:] - by_day[p].pv_kwh.to_numpy(float)[s:]
                                     for p in prior])
                if pv_weight == "ls":
                    # 含交叉项最小二乘（与负荷融合同一公式；反方差/MAE倒数是其
                    # 误差互不相关时的特例）
                    eA, eR = off_err - bias, reg_err
                    den = float(np.sum((eA - eR) ** 2))
                    w = (0.5 if den <= 0
                         else float(np.clip(np.sum(eR * (eR - eA)) / den, 0.0, 1.0)))
                else:
                    off_mae = float(np.mean(np.abs(off_err - bias)))
                    reg_mae = float(np.mean(np.abs(reg_err)))
                    w = reg_mae / (off_mae + reg_mae + 1e-9)
                pv_hat[(d, k)] = np.maximum(
                    0.0, w * np.maximum(0.0, anchor[(d, k)] - bias)
                    + (1.0 - w) * ridge_pv[d][s:])
                n_fused += 1

    resid_load = {d: by_day[d].load_kwh.to_numpy(float) - load_hat[d]
                  for d in sorted(load_hat) if d in by_day}
    resid_pv = {(d, k): by_day[d].pv_kwh.to_numpy(float)[6 * k:] - arr
                for (d, k), arr in pv_hat.items()}

    lib_pool = sorted(d for d in ridge_load
                      if d in stat_load and all((d, k) in pv_hat for k in ISSUES))
    if verbose:
        print(f"  预报库：负荷={load_src} 光伏={pv_src}（定权={pv_weight}）；情景日期池 {len(lib_pool)} 天"
              f"（{lib_pool[0].date()}~{lib_pool[-1].date()}），融合对 {n_fused}", flush=True)
    return {"load_hat": load_hat, "pv_hat": pv_hat, "resid_load": resid_load,
            "resid_pv": resid_pv, "lib_pool": lib_pool, "by_day": by_day}


def forecast_accuracy(lib: dict, dates) -> dict:
    """被评价日的逐点精度（kWh/10min → ×6 折成 kW），光伏额外给白昼口径。"""
    la, pa, lh, ph = [], [], [], []
    for d in dates:
        if d not in lib["load_hat"] or (d, 0) not in lib["pv_hat"]:
            continue
        la.append(lib["by_day"][d].load_kwh.to_numpy(float))
        pa.append(lib["by_day"][d].pv_kwh.to_numpy(float))
        lh.append(lib["load_hat"][d])
        ph.append(lib["pv_hat"][(d, 0)])
    la, pa = np.concatenate(la), np.concatenate(pa)
    lh, ph = np.concatenate(lh), np.concatenate(ph)
    day = pa * 6.0 > 50.0
    return {"load_mae_kw": round(float(np.mean(np.abs(lh - la)) * 6), 2),
            "load_bias_kw": round(float(np.mean(lh - la) * 6), 2),
            "pv_mae_kw": round(float(np.mean(np.abs(ph - pa)) * 6), 2),
            "pv_bias_kw": round(float(np.mean(ph - pa) * 6), 2),
            "pv_day_mae_kw": round(float(np.mean(np.abs(ph[day] - pa[day])) * 6), 2),
            "pv_day_bias_kw": round(float(np.mean(ph[day] - pa[day]) * 6), 2),
            "net_mae_kw": round(float(np.mean(np.abs((lh - ph) - (la - pa))) * 6), 2),
            "net_bias_kw": round(float(np.mean((lh - ph) - (la - pa)) * 6), 2)}


# --------------------------------------------------------------------------- #
# 行 det：确定性滚动 MPC + 贪心执行（可选硬终端 6000 / 软终端估值 V_CONT）
# --------------------------------------------------------------------------- #
def run_det(cfg, data, official, stat, ridge, dates, prm, terminal, updates=(6, 12, 18)):
    panel = stat if cfg["load"] == "stat" else ridge
    pv_mode = "anchor" if cfg["pv"] == "anchor" else "fused"
    q3.execute_segment = greedy_segment
    if terminal == "soft":
        def soft_mpc(*args, **kwargs):
            if kwargs.get("terminal_target") is not None:
                kwargs["terminal_penalty"] = V_CONT
            # 同时保留 terminal_target 以触发软约束分支（solve_mpc 内两者并存即为分段线性罚）
            return _ORIG_SOLVE_MPC(*args, **kwargs)
        q3.solve_mpc = soft_mpc
    else:
        q3.solve_mpc = _ORIG_SOLVE_MPC
    out, met = q3.run_q3_period(data, panel, official, dates,
                                prm["initial_energy_kwh"], prm, tuple(updates), pv_mode)
    q3.solve_mpc = _ORIG_SOLVE_MPC
    met["max_balance_residual_kwh"] = met["max_balance_residual_kwh"].abs()
    return out, met


# --------------------------------------------------------------------------- #
# 行 sp：情景两阶段 SP + DP 价值执行器
# --------------------------------------------------------------------------- #
def run_sp(cfg, lib, dates, prm, m_scen, issues=ISSUES):
    by_day, load_hat = lib["by_day"], lib["load_hat"]
    pv_hat, resid_load = lib["pv_hat"], lib["resid_load"]
    resid_pv, lib_pool = lib["resid_pv"], lib["lib_pool"]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    ec, ed = prm["charge_efficiency"], prm["discharge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    energy, rows, daily, t0 = prm["initial_energy_kwh"], [], [], time.perf_counter()
    for i, day in enumerate(dates):
        if day not in load_hat or (day, 0) not in pv_hat:
            continue
        actual = by_day[day]
        price = actual.price_fixed.to_numpy(float)
        load_a = actual.load_kwh.to_numpy(float)
        pv_a = actual.pv_kwh.to_numpy(float)
        lib_days = [d for d in lib_pool if d < day][-m_scen:]

        a_full, x0, e, start_row = None, None, energy, len(rows)
        for k in issues:
            s, Tx = 6 * k, T - 6 * k
            if (day, k) not in pv_hat:
                continue
            L = np.array([np.maximum(load_hat[day][s:] + resid_load[d][s:], 0.0)
                          for d in lib_days])
            V = np.array([np.maximum(pv_hat[(day, k)] + resid_pv[(d, k)], 0.0)
                          for d in lib_days])
            # 调整费用的参照恒为 0:00 原计划 x0（口径C，每段只结算一次）
            ref = None if k == 0 else x0[s:]
            a_seg = two_stage_plan_q3(price[s:], L, V, e, ref, prm)
            if a_full is None:
                a_full = np.zeros(T)
            a_full[s:] = a_seg
            if k == 0:
                x0 = a_full.copy()
            Hbar = future_value_curves_h((L - V) - a_seg[None, :], price[s:], prm)
            R = retention_levels_h(Hbar, price[s:], prm)
            stop = T if k == issues[-1] else 6 * (k + 6)
            for t in range(s, stop):
                r = (load_a[t] - pv_a[t]) - a_full[t]
                if r > 0:
                    dis = min(r, S, ed * max(e - R[t - s], 0.0))
                    chg = sur = 0.0
                    eme = r - dis
                else:
                    chg = min(-r, S, (emax - e) / ec)
                    dis = eme = 0.0
                    sur = -r - chg
                e_next = e + ec * chg - dis / ed
                rows.append((day, t + 1, x0[t], a_full[t], chg, dis, sur, eme,
                             e, e_next, load_a[t], pv_a[t], price[t]))
                e = e_next
        energy = e
        day_rows = rows[start_row:]
        costs = q3.settlement_c(x0, a_full, price,
                                np.array([r[7] for r in day_rows]))
        daily.append({"date": day, **costs, "terminal_soc_kwh": float(energy),
                      "adjusted_kwh": float(a_full.sum()),
                      "original_kwh": float(x0.sum())})
        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            print(f"  sp/{cfg['name']} [{i+1}/{len(dates)}] {day.date()} "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)

    out = pd.DataFrame(rows, columns=DISPATCH_COLS)
    daily = pd.DataFrame(daily)
    bal = (out.buy_kwh + out.pv_kwh + out.discharge_kwh + out.emergency_kwh
           - out.load_kwh - out.charge_kwh - out.surplus_kwh)
    daily["max_balance_residual_kwh"] = float(np.abs(bal).max())
    return out, daily


# --------------------------------------------------------------------------- #
def summarize(tag, out, daily) -> dict:
    cost = daily["total"].to_numpy()
    q95 = float(np.quantile(cost, .95))
    parts = daily[["within_cost", "under_cost", "over_cost", "emergency_cost"]] \
        .to_numpy().sum()
    return {"tag": tag,
            "days": int(out.date.nunique()),
            "slots_per_day_min": int(out.groupby("date").size().min()),
            "total": float(cost.sum()),
            "within_cost": float(daily.within_cost.sum()),
            "under_cost": float(daily.under_cost.sum()),
            "over_cost": float(daily.over_cost.sum()),
            "emergency_cost": float(daily.emergency_cost.sum()),
            "emergency_kwh": float(out.emergency_kwh.sum()),
            "surplus_kwh": float(out.surplus_kwh.sum()),
            "charge_kwh": float(out.charge_kwh.sum()),
            "discharge_kwh": float(out.discharge_kwh.sum()),
            "under_kwh": float(np.maximum(
                out.original_plan_kwh - out.buy_kwh, 0).sum()),
            "over_kwh": float(np.maximum(
                out.buy_kwh - out.original_plan_kwh, 0).sum()),
            "cvar95": float(cost[cost >= q95].mean()),
            "max_day": float(cost.max()),
            "terminal_soc_kwh": float(out.energy_end_kwh.iloc[-1]),
            "max_balance_residual_kwh": float(daily.max_balance_residual_kwh.max()),
            "settlement_identity_err": float(abs(cost.sum() - parts))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["det", "sp"], required=True)
    ap.add_argument("--forecast", required=True,
                    choices=["stat_anchor", "ridge_anchor", "ridge_fused"])
    ap.add_argument("--terminal", choices=["hard", "soft"], default="hard")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--start", default="2025-02-01")
    ap.add_argument("--residual-window", type=int, default=30)
    ap.add_argument("--updates", default="6,12,18")
    ap.add_argument("--issues", default="0,6,12,18",
                    help="sp 行的预报发布时刻子集，用于信息价值(VOI)消融")
    ap.add_argument("--ridge-panel", default=str(RIDGE_PANEL),
                    help="ridge 负荷面板路径（默认旧 Ridge；VOI 同口径时可指向 "
                         "data_processed/q3_forecasts_fused.csv）")
    args = ap.parse_args()
    updates = tuple(int(x) for x in args.updates.split(",") if x != "")
    issues = tuple(int(x) for x in args.issues.split(",") if x != "")

    load_src, pv_src = args.forecast.split("_")
    cfg = {"name": args.forecast, "load": load_src, "pv": pv_src}
    data = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                       parse_dates=["date", "timestamp_end"])
    official = pd.read_csv(ROOT / "data_processed/pv_forecast_hourly.csv",
                           parse_dates=["issue_at", "target_end"])
    prm = json.loads((ROOT / "data_processed/q1_storage_parameters.json")
                     .read_text(encoding="utf-8"))
    stat = pd.read_csv(STAT_PANEL, parse_dates=["date"])
    ridge = pd.read_csv(args.ridge_panel, parse_dates=["date"])
    dates = pd.date_range(args.start, "2025-12-31")
    if args.days:
        dates = dates[:args.days]

    tag = f"{args.model}_{args.forecast}" + (
        f"_{args.terminal}" if args.model == "det" else "")
    if args.model == "det" and updates != (6, 12, 18):
        tag += "_u" + "-".join(str(u) for u in updates)
    if args.model == "sp" and issues != ISSUES:
        tag += "_i" + "-".join(str(u) for u in issues)
    print(f"=== {tag} | {len(dates)} 天 | M={args.scenarios} | "
          f"残差窗 {args.residual_window} ===", flush=True)
    t0 = time.perf_counter()
    lib = build_library(load_src, pv_src, data, official, stat, ridge)
    if args.model == "det":
        out, daily = run_det(cfg, data, official, stat, ridge, dates, prm,
                             args.terminal, updates)
    else:
        out, daily = run_sp(cfg, lib, dates, prm, args.scenarios, issues)
    dt = time.perf_counter() - t0

    res = summarize(tag, out, daily)
    res.update({"seconds": round(dt, 1), "window": f"{dates[0].date()}~{dates[-1].date()}",
                "scenarios": args.scenarios, "residual_window": args.residual_window,
                "forecast_accuracy": forecast_accuracy(lib, dates)})
    out.to_csv(RESULTS / f"abl_{tag}_dispatch.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(RESULTS / f"abl_{tag}_daily.csv", index=False, encoding="utf-8-sig")
    (RESULTS / f"abl_{tag}_summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{tag}: 用时 {dt:.0f}s")
    for k in ("days", "slots_per_day_min", "total", "emergency_cost", "emergency_kwh",
              "under_kwh", "over_kwh", "cvar95", "max_day", "terminal_soc_kwh",
              "max_balance_residual_kwh", "settlement_identity_err"):
        print(f"  {k:26s} {res[k]:,.4f}" if isinstance(res[k], float)
              else f"  {k:26s} {res[k]}")
    print(f"  预测精度 {json.dumps(res['forecast_accuracy'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
