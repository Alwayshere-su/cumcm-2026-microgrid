"""问题二 方法2：参考稿三层结构（统计预测 + 历史情景随机规划 + DP 价值执行器）。

严格对齐参考稿 §4：
  ① 统计预测：负载取"此前 35 天内同星期日期的同时段均值"（同星期样本 <2 天回退到
     此前最多 7 天）；光伏取"此前 m_d=min(7,d-1) 天的同时段均值"。非前视。
  ② 历史情景：此前 30 天的样本外残差整日重采样（保留 144 段日内相关性与负载—光伏配对），
     M=30 等权情景。1/1 无预测不进残差库。
  ③ 主模型：两阶段随机规划（here-and-now 计划 g 全情景共享，wait-and-see 情景内
     C/D/E/b/U）。目标 = 计划费用 + 期望 5 倍紧急费用 − 日末库存价值 v·E_T，v=0.481548。
  ④ 执行器：DP 价值执行器。逐情景反向递推路径未来费用 H_{t,ω}(e)（终端 −v·e），
     等权平均得 H_t(e)，由动态保留水平 R_t 投影式放电。

物理与结算口径与 solve_q2 一致：144 段/天、2025-02-01 至 12-31、初始 SOC 6000 kWh、
跨日连续、紧急购电按交易时刻电价 5 倍结算。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_two_layer import CPWL, add, infconv, restrict  # noqa: E402
from solve_q2 import DATA, PARAMS  # noqa: E402

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
T = 144
V_CONT = 0.481548          # 日末库存续存价值（元/kWh），参考稿取值
M_SCEN = 30                # 情景数
RESID_WINDOW = 30          # 残差库窗口（天）


# --------------------------------------------------------------------------- #
# ① 统计预测（非前视）
# --------------------------------------------------------------------------- #
def stat_forecast(by_day: dict, day: pd.Timestamp, load_win: int = 35, pv_win: int = 7):
    hist = [d for d in sorted(by_day) if d < day]
    if not hist:
        raise ValueError(f"no history before {day.date()}")
    lo = day - pd.Timedelta(days=load_win)
    same_wd = [d for d in hist if d >= lo and d.weekday() == day.weekday()]
    if len(same_wd) < 2:
        same_wd = hist[-7:]
    load_hat = np.mean([by_day[d]["load_kwh"].to_numpy(float) for d in same_wd], axis=0)
    m = min(pv_win, len(hist))
    pv_hat = np.mean([by_day[d]["pv_kwh"].to_numpy(float) for d in hist[-m:]], axis=0)
    return load_hat, pv_hat


def build_forecast_and_residuals(data: pd.DataFrame):
    """为每个日期预计算非前视预测，并返回 (forecast, residual) 字典。"""
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    fcast, resid = {}, {}
    for day in sorted(by_day):
        try:
            lh, ph = stat_forecast(by_day, day)
        except ValueError:
            continue
        fcast[day] = (lh, ph)
        al = by_day[day]["load_kwh"].to_numpy(float)
        av = by_day[day]["pv_kwh"].to_numpy(float)
        resid[day] = (al - lh, av - ph)
    return fcast, resid


def residuals_from_forecasts(data: pd.DataFrame, forecasts: dict) -> tuple[dict, dict]:
    """给定 day->(load_hat, pv_hat)，返回非前视 (forecast, residual) 字典。"""
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    fcast, resid = {}, {}
    for day, (lh, ph) in forecasts.items():
        if day not in by_day:
            continue
        lh = np.asarray(lh, float)
        ph = np.asarray(ph, float)
        fcast[day] = (lh, ph)
        resid[day] = (by_day[day]["load_kwh"].to_numpy(float) - lh,
                      by_day[day]["pv_kwh"].to_numpy(float) - ph)
    return fcast, resid


# --------------------------------------------------------------------------- #
# ③ 两阶段随机规划（计划 g 全情景共享）
# --------------------------------------------------------------------------- #
def two_stage_plan(price: np.ndarray, L: np.ndarray, V: np.ndarray,
                   energy0: float, prm: dict, v: float = V_CONT) -> np.ndarray:
    M = L.shape[0]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]

    n_g = T
    per = 5 * T + 1                      # c, d, u, b, E(0..T)
    n = n_g + M * per

    def base(m):
        return n_g + m * per

    def ix(m, name, t):
        off = base(m) + {"c": 0, "d": T, "u": 2 * T, "b": 3 * T, "E": 4 * T}[name]
        return off + t

    c = np.zeros(n)
    c[:T] = price
    for m in range(M):
        c[base(m) + 3 * T: base(m) + 4 * T] = 5.0 * price / M     # 期望紧急费用
        c[ix(m, "E", T)] = -v / M                                  # 日末库存价值

    Aeq = lil_matrix((2 * T * M, n))
    beq = np.zeros(2 * T * M)
    for m in range(M):
        for t in range(T):
            r = m * 2 * T + t
            Aeq[r, t] = 1.0                                        # g_t 共享
            Aeq[r, ix(m, "b", t)] = 1.0
            Aeq[r, ix(m, "d", t)] = 1.0
            Aeq[r, ix(m, "c", t)] = -1.0
            Aeq[r, ix(m, "u", t)] = -1.0
            beq[r] = L[m, t] - V[m, t]
            r2 = m * 2 * T + T + t
            Aeq[r2, ix(m, "E", t + 1)] = 1.0
            Aeq[r2, ix(m, "E", t)] = -1.0
            Aeq[r2, ix(m, "c", t)] = -eta
            Aeq[r2, ix(m, "d", t)] = 1.0 / eta

    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    for m in range(M):
        hi[base(m) + 0: base(m) + 2 * T] = S
        lo[base(m) + 4 * T: base(m) + 4 * T + T + 1] = emin
        hi[base(m) + 4 * T: base(m) + 4 * T + T + 1] = emax
        lo[ix(m, "E", 0)] = hi[ix(m, "E", 0)] = energy0

    res = linprog(c, A_eq=Aeq.tocsr(), b_eq=beq, bounds=list(zip(lo, hi)), method="highs")
    if not res.success:
        raise RuntimeError(f"two-stage LP failed: {res.message}")
    return res.x[:T]


# --------------------------------------------------------------------------- #
# ④ DP 价值执行器
# --------------------------------------------------------------------------- #
def build_h_tilde(c5: float, r: float, eta: float, S: float) -> CPWL:
    """反射单时段费用 H_tilde(y)=5c*[r+psi(-y)]^+，限制在允许的 y 区间。

    r>0（缺额，只能放电）：y∈[0, min(S,r)/eta]
    r<=0（富余，只能充电）：y∈[-eta*min(S,-r), 0]，且该区间上费用恒为 0
    """
    if r > 1e-12:
        y_hi = min(S, r) / eta
        y0 = r / eta
        if y0 >= y_hi - 1e-12:
            return CPWL(0.0, c5 * r, [y_hi], [-c5 * eta])
        return CPWL(0.0, c5 * r, [y0, y_hi - y0], [-c5 * eta, 0.0])
    y_lo = -eta * min(S, -r)
    return CPWL(y_lo, 0.0, [-y_lo], [0.0])


def future_value_curves(r_scen: np.ndarray, price: np.ndarray, prm: dict,
                        v: float = V_CONT) -> list[CPWL]:
    """返回平均未来费用 Hbar[0..T]（Hbar[t] 为进入第 t+1 时段的未来费用）。"""
    M = r_scen.shape[0]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]

    acc: list[CPWL | None] = [None] * (T + 1)
    for m in range(M):
        H = [None] * (T + 1)
        H[T] = CPWL(emin, -v * emin, [emax - emin], [-v])
        for t in range(T - 1, -1, -1):
            h = build_h_tilde(5.0 * price[t], float(r_scen[m, t]), eta, S)
            g = infconv(h, H[t + 1])
            H[t] = restrict(g, emin, emax)
            if H[t] is None:
                raise RuntimeError(f"scenario {m} step {t} 定义域为空")
        for t in range(T + 1):
            acc[t] = H[t] if acc[t] is None else add(acc[t], H[t])
    return [_scale(f, 1.0 / M) for f in acc]


def _scale(f: CPWL, a: float) -> CPWL:
    return CPWL(f.lo, f.f_lo * a, f.widths.copy(), f.slopes * a)


def add_linear(f: CPWL, a: float) -> CPWL:
    return CPWL(f.lo, f.f_lo + a * f.lo, f.widths.copy(), f.slopes + a)


def argmin_max(f: CPWL, lo: float, hi: float, tol: float = 1e-12) -> float:
    """取最大最优解（存在平台时取平台右端）。"""
    g = restrict(f, lo, hi)
    if g is None:
        raise RuntimeError("argmin_empty")
    if g.n_segments == 0:
        return g.lo
    idx = np.nonzero(g.slopes >= -tol)[0]
    if len(idx) == 0:
        return float(g.hi)
    j = int(idx[0])
    if abs(g.slopes[j]) <= tol:
        k = j
        while k + 1 < g.n_segments and abs(g.slopes[k + 1]) <= tol:
            k += 1
        return float(g.breaks[k + 1])
    return float(g.breaks[j])


def retention_levels(Hbar: list[CPWL], price: np.ndarray, prm: dict) -> np.ndarray:
    eta = prm["charge_efficiency"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    R = np.zeros(T)
    for t in range(T):
        g = add_linear(Hbar[t + 1], 5.0 * price[t] * eta)
        R[t] = argmin_max(g, emin, emax)
    return R


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
def run_backtest(max_days: int | None = None, m_scen: int = M_SCEN,
                 fcast: dict | None = None, resid: dict | None = None,
                 model_name: str | None = None,
                 plan_uplift: float = 1.0) -> dict:
    data = pd.read_csv(DATA, parse_dates=["date"])
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    eta_c = prm["charge_efficiency"]
    eta_d = prm["discharge_efficiency"]
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    emin, emax = prm["operating_min_kwh"], prm["operating_max_kwh"]
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}

    if fcast is None or resid is None:
        print("预计算非前视统计预测与残差库 ...", flush=True)
        fcast, resid = build_forecast_and_residuals(data)
    resid_days = sorted(resid)

    dates = pd.date_range("2025-02-01", "2025-12-31", freq="D")
    if max_days:
        dates = dates[:max_days]

    energy = prm["initial_energy_kwh"]
    rows = []
    t_start = time.perf_counter()
    for i, day in enumerate(dates):
        actual = by_day[day]
        price = actual["price_fixed"].to_numpy(float)
        load_a = actual["load_kwh"].to_numpy(float)
        pv_a = actual["pv_kwh"].to_numpy(float)
        lh, ph = fcast[day]

        # 情景：此前 m_scen 天的残差整日重采样
        lib = [d for d in resid_days if d < day][-m_scen:]
        L = np.array([np.maximum(lh + resid[d][0], 0.0) for d in lib])
        V = np.array([np.maximum(ph + resid[d][1], 0.0) for d in lib])

        g = two_stage_plan(price, L, V, energy, prm)
        if plan_uplift != 1.0:            # 敏感性实验：计划量整体放大
            g = g * plan_uplift
        r_scen = (L - V) - g[None, :]
        Hbar = future_value_curves(r_scen, price, prm)
        R = retention_levels(Hbar, price, prm)

        # 因果执行（真实序列）
        e = energy
        for t in range(T):
            r = (load_a[t] - pv_a[t]) - g[t]
            if r > 0:
                dis = min(r, S, eta_d * max(e - R[t], 0.0))
                chg = 0.0
                emer = r - dis
                surp = 0.0
            else:
                chg = min(-r, S, (emax - e) / eta_c)
                dis = 0.0
                emer = 0.0
                surp = -r - chg
            e_next = e + eta_c * chg - dis / eta_d
            rows.append((day, t + 1, g[t], chg, dis, surp, emer, e, e_next,
                         price[t] * g[t], 5.0 * price[t] * emer, R[t]))
            e = e_next
        energy = e
        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            el = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(dates)}] {day.date()} 累计 {el:.1f}s", flush=True)

    out = pd.DataFrame(rows, columns=["date", "slot", "plan_buy_kwh", "charge_kwh",
                                      "discharge_kwh", "surplus_kwh", "emergency_kwh",
                                      "energy_start_kwh", "energy_end_kwh",
                                      "plan_cost_yuan", "emergency_cost_yuan", "retention_kwh"])
    bal = (out.plan_buy_kwh + out.discharge_kwh + out.emergency_kwh
           - out.charge_kwh - out.surplus_kwh)
    dkey = data["date"].astype(str) + "_" + data["slot"].astype(str)
    act = data.set_index(dkey)
    key = (out.date.astype(str) + "_" + out.slot.astype(str)).to_numpy()
    residual = bal.to_numpy() - (act.loc[key, "load_kwh"].to_numpy()
                                 - act.loc[key, "pv_kwh"].to_numpy())
    soc = (out.energy_end_kwh - out.energy_start_kwh
           - eta_c * out.charge_kwh + out.discharge_kwh / eta_d)

    summary = {
        "model": model_name or "方法2：统计预测 + 历史情景随机规划 + DP 价值执行器",
        "scenarios": int(m_scen),
        "days": len(dates),
        "total_plan_purchase_kwh": float(out.plan_buy_kwh.sum()),
        "plan_cost_yuan": float(out.plan_cost_yuan.sum()),
        "emergency_cost_yuan": float(out.emergency_cost_yuan.sum()),
        "total_cost_yuan": float(out.plan_cost_yuan.sum() + out.emergency_cost_yuan.sum()),
        "total_emergency_kwh": float(out.emergency_kwh.sum()),
        "emergency_days": int((out.groupby("date").emergency_kwh.sum() > 1e-6).sum()),
        "emergency_intervals": int((out.emergency_kwh > 1e-6).sum()),
        "min_soc_kwh": float(min(out.energy_start_kwh.min(), out.energy_end_kwh.min())),
        "max_soc_kwh": float(max(out.energy_start_kwh.max(), out.energy_end_kwh.max())),
        "initial_soc_kwh": float(out.energy_start_kwh.iloc[0]),
        "terminal_soc_kwh": float(out.energy_end_kwh.iloc[-1]),
        "max_balance_residual_kwh": float(np.abs(residual).max()),
        "max_soc_residual_kwh": float(np.abs(soc).max()),
        "max_simultaneous_charge_discharge_kwh": float(np.minimum(out.charge_kwh, out.discharge_kwh).max()),
    }
    return {"summary": summary, "dispatch": out}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="仅跑前 N 天（调试）")
    ap.add_argument("--scenarios", type=int, default=M_SCEN)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    res = run_backtest(args.days, args.scenarios)
    s = res["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.save:
        RESULTS.mkdir(exist_ok=True)
        res["dispatch"].to_csv(RESULTS / "q2_twolayer_dispatch.csv", index=False, encoding="utf-8-sig")
        (RESULTS / "q2_twolayer_summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写出 results/q2_twolayer_summary.json")


if __name__ == "__main__":
    main()
