"""问题三推荐结构（全程无机器学习）全年验证。

结构 = 问题二方法三的"情景随机规划 + DP 价值执行器"，外挂问题三的调整费用与多时刻更新：

  预测  负荷 = 同星期几 35 天同时段均值；光伏 = 附件3 官方预报整点锚点线性插值
  申报  0:00 两阶段 SP 定计划 x^0；6/12/18 用当次预报重解剩余时段 SP 得调整量 a_t
        目标 = p·a + 0.5·p·(少购+超购) + E[5p·紧急购电] − v·E_T，含 a+u−v=x^0 精确分解
  执行  每个发布时刻用情景残差反向递推 DP 未来费用曲线，得保留水位 R_t，
        真实序列因果执行：缺额先放电至 R_t，剩余才 5 倍紧急购电
  情景  此前 30 天"同发布时刻—同提前量"的整日联合残差重采样（负荷与光伏同日配对）

用法：
  python scripts/probe_q3_rolling_sp.py --days 6 --updates 0,6,12,18   # 冒烟
  python scripts/probe_q3_rolling_sp.py --updates 0,6,12,18            # 全年
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()

import solve_q3 as q3  # noqa: E402
from q1_two_layer import CPWL, infconv, restrict  # noqa: E402
from solve_q2_twolayer import (stat_forecast, build_h_tilde, add,  # noqa: E402
                               _scale, add_linear, argmin_max, M_SCEN, V_CONT)

T = 144


def future_value_curves_h(r_scen, price, prm, v=V_CONT):
    """时段数可变的 DP 反向递推（solve_q2_twolayer 版本硬编码 144 段）。"""
    M, Tx = r_scen.shape
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    acc = [None] * (Tx + 1)
    for m in range(M):
        H = [None] * (Tx + 1)
        H[Tx] = CPWL(emin, -v * emin, [emax - emin], [-v])
        for t in range(Tx - 1, -1, -1):
            g = infconv(build_h_tilde(5.0 * price[t], float(r_scen[m, t]), eta, S), H[t + 1])
            H[t] = restrict(g, emin, emax)
            if H[t] is None:
                raise RuntimeError(f"scenario {m} step {t} 定义域为空")
        for t in range(Tx + 1):
            acc[t] = H[t] if acc[t] is None else add(acc[t], H[t])
    return [_scale(f, 1.0 / M) for f in acc]


def retention_levels_h(Hbar, price, prm):
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    return np.array([argmin_max(add_linear(Hbar[t + 1], 5.0 * price[t] * eta), emin, emax)
                     for t in range(len(price))])


def two_stage_plan_q3(price, L, V, energy0, x0_ref, prm, v=V_CONT):
    """两阶段随机规划：调整量 a 全情景共享，含 0.5p·|a−x^0| 调整费。"""
    M, Tx = L.shape
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    use_ref = x0_ref is not None
    n_dev = 2 * Tx if use_ref else 0
    per = 5 * Tx + 1

    def base(m):
        return Tx + n_dev + m * per

    def ix(m, name, t):
        return base(m) + {"c": 0, "d": Tx, "u": 2 * Tx, "b": 3 * Tx, "E": 4 * Tx}[name] + t

    n = Tx + n_dev + M * per
    cost = np.zeros(n)
    cost[:Tx] = price
    if use_ref:
        cost[Tx:Tx + Tx] = 0.5 * price
        cost[Tx + Tx:Tx + 2 * Tx] = 0.5 * price
    for m in range(M):
        cost[base(m) + 3 * Tx: base(m) + 4 * Tx] = 5.0 * price / M
        cost[ix(m, "E", Tx)] = -v / M

    rows = 2 * Tx * M + (Tx if use_ref else 0)
    Aeq, beq = lil_matrix((rows, n)), np.zeros(rows)
    for m in range(M):
        for t in range(Tx):
            r = m * 2 * Tx + t
            Aeq[r, t] = 1.0
            Aeq[r, ix(m, "b", t)] = 1.0
            Aeq[r, ix(m, "d", t)] = 1.0
            Aeq[r, ix(m, "c", t)] = -1.0
            Aeq[r, ix(m, "u", t)] = -1.0
            beq[r] = L[m, t] - V[m, t]
            r2 = r + Tx
            Aeq[r2, ix(m, "E", t + 1)] = 1.0
            Aeq[r2, ix(m, "E", t)] = -1.0
            Aeq[r2, ix(m, "c", t)] = -eta
            Aeq[r2, ix(m, "d", t)] = 1.0 / eta
    if use_ref:
        off = 2 * Tx * M
        for t in range(Tx):
            Aeq[off + t, t] = 1.0
            Aeq[off + t, Tx + t] = 1.0
            Aeq[off + t, Tx + Tx + t] = -1.0
            beq[off + t] = x0_ref[t]

    lo, hi = np.zeros(n), np.full(n, np.inf)
    for m in range(M):
        hi[base(m): base(m) + 2 * Tx] = S
        lo[base(m) + 4 * Tx: base(m) + 4 * Tx + Tx + 1] = emin
        hi[base(m) + 4 * Tx: base(m) + 4 * Tx + Tx + 1] = emax
        lo[ix(m, "E", 0)] = hi[ix(m, "E", 0)] = energy0

    res = linprog(cost, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=list(zip(lo, hi)), method="highs")
    if not res.success:
        raise RuntimeError(f"Q3 two-stage LP failed: {res.message}")
    return res.x[:Tx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--updates", type=str, default="0,6,12,18")
    ap.add_argument("--scenarios", type=int, default=M_SCEN)
    args = ap.parse_args()
    issues = tuple(int(x) for x in args.updates.split(","))
    M = args.scenarios

    data = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                       parse_dates=["date", "timestamp_end"])
    official = pd.read_csv(ROOT / "data_processed/pv_forecast_hourly.csv",
                           parse_dates=["issue_at", "target_end"])
    prm = json.loads((ROOT / "data_processed/q1_storage_parameters.json")
                     .read_text(encoding="utf-8"))
    eta_c, eta_d = prm["charge_efficiency"], prm["discharge_efficiency"]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]

    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    print("预计算非前视预报与分时刻残差库 ...", flush=True)
    load_hat, resid_load, pv_hat, resid_pv = {}, {}, {}, {}
    for day in sorted(by_day):
        try:
            lh, _ = stat_forecast(by_day, day)
        except ValueError:
            continue
        la = by_day[day].load_kwh.to_numpy(float)
        pa = by_day[day].pv_kwh.to_numpy(float)
        load_hat[day] = lh
        resid_load[day] = la - lh
        for k in issues:
            try:
                phk = q3._official_current_day(official, by_day[day], day, k)
            except ValueError:
                continue
            pv_hat[(day, k)] = phk
            resid_pv[(day, k)] = pa[k * 6:] - phk
    days_all = sorted(load_hat)
    print(f"  可用日期 {len(days_all)} 天，时刻 {issues}", flush=True)

    dates = pd.date_range("2025-02-01", "2025-12-31")
    if args.days:
        dates = dates[:args.days]

    energy, rows, t0 = prm["initial_energy_kwh"], [], time.perf_counter()
    for i, day in enumerate(dates):
        if day not in load_hat:
            continue
        actual = by_day[day]
        price = actual.price_fixed.to_numpy(float)
        load_a = actual.load_kwh.to_numpy(float)
        pv_a = actual.pv_kwh.to_numpy(float)
        lib = [d for d in days_all if d < day][-M:]

        a_full = None
        e = energy
        for k in issues:
            s = k * 6
            Tx = T - s
            if (day, k) not in pv_hat:
                continue
            L = np.array([np.maximum(load_hat[day][s:] + resid_load[d][s:], 0.0) for d in lib])
            V = np.array([np.maximum(pv_hat[(day, k)] + resid_pv[(d, k)], 0.0) for d in lib])
            x0_ref = None if k == 0 else a_full[:T] if a_full is not None else None
            ref = None if x0_ref is None else x0_ref[s:]
            a_seg = two_stage_plan_q3(price[s:], L, V, e, ref, prm)
            if a_full is None:
                a_full = np.zeros(T)
            a_full[s:] = a_seg

            R = retention_levels_h(future_value_curves_h((L - V) - a_seg[None, :],
                                                         price[s:], prm), price[s:], prm)
            nxt = [x for x in issues if x > k]
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
                e = e_next
        energy = e
        if (i + 1) % 25 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] {day.date()} {time.perf_counter()-t0:.1f}s", flush=True)

    out = pd.DataFrame(rows, columns=["date", "slot", "buy_kwh", "charge_kwh", "discharge_kwh",
                                      "surplus_kwh", "emergency_kwh", "energy_start_kwh",
                                      "energy_end_kwh", "price"])
    dcost = (out.buy_kwh * out.price + 5.0 * out.emergency_kwh * out.price)
    day_cost = dcost.groupby(out.date).sum().to_numpy()
    print(f"\nQ3 滚动SP+DP (updates={issues}, M={M}): "
          f"total={dcost.sum():,.0f}  购电费={(out.buy_kwh*out.price).sum():,.0f}  "
          f"紧急费={(5.0*out.emergency_kwh*out.price).sum():,.0f}  "
          f"紧急电量={out.emergency_kwh.sum():,.0f}  "
          f"CVaR95={day_cost[day_cost >= np.quantile(day_cost, .95)].mean():,.0f}  "
          f"最坏日={day_cost.max():,.0f}  末态SOC={out.energy_end_kwh.iloc[-1]:,.1f}")
    out.to_csv(ROOT / f"results/q3_rollingsp_{'-'.join(map(str, issues))}_dispatch.csv",
               index=False, encoding="utf-8-sig")
    (ROOT / f"results/q3_rollingsp_{'-'.join(map(str, issues))}_summary.json").write_text(json.dumps(
        {"issues": issues, "scenarios": M, "days": int(out.date.nunique()),
         "total": float(dcost.sum()), "emergency_cost": float((5.0 * out.emergency_kwh * out.price).sum()),
         "emergency_kwh": float(out.emergency_kwh.sum()),
         "cvar95": float(day_cost[day_cost >= np.quantile(day_cost, .95)].mean()),
         "max_day": float(day_cost.max()), "terminal_soc": float(out.energy_end_kwh.iloc[-1])},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
