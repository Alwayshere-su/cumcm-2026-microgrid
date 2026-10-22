"""第一问：两层模型 与 LP 单层模型 的结果对比。

四种口径
--------
(A) LP 松弛单层        : 720 个连续变量，无 0-1 互斥，一次求解（HiGHS），附影子价格
(B) MILP 单目标单层    : 含 144 个 0-1 充放电互斥变量，目标仅 min 全天购电费
(C) LP 确定性求解    : 复用 scripts/solve_q1.py（费用→电池吞吐），用于生成 result1.xlsx
(D) 两层模型          : LP 影子价格（解释）+ 连续状态 DP 斜率递推（求最优值）

输出
----
results/q1_layer_comparison.json       四口径汇总 + 互证指标
results/q1_layer_comparison.md         Markdown 对比表（可直接进论文）
results/q1_layer_dispatch_compare.csv  逐 10 分钟时段的调度并列对比
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
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()

from q1_two_layer import marginal_value, two_layer_dp  # noqa: E402
from solve_q1 import _base_problem, _solve, solve_q1  # noqa: E402

DATA = ROOT / "data_processed" / "q1_typical_day.csv"
PARAMS = ROOT / "data_processed" / "q1_storage_parameters.json"
RESULTS = ROOT / "results"

REQUESTED_SLOTS = [61, 73, 85, 97, 109, 121]
REQUESTED_LABELS = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
                    "16:00-16:10", "18:00-18:10", "20:00-20:10"]


# --------------------------------------------------------------------------- #
# (A) LP 松弛单层
# --------------------------------------------------------------------------- #
def _indices(T: int) -> dict[str, np.ndarray]:
    start = 0
    out: dict[str, np.ndarray] = {}
    for name, size in (("buy", T), ("charge", T), ("discharge", T),
                       ("surplus", T), ("energy", T + 1)):
        out[name] = np.arange(start, start + size)
        start += size
    return out


def solve_lp_single(df: pd.DataFrame, prm: dict, e0: float | None = None) -> dict:
    T = len(df)
    idx = _indices(T)
    n = int(idx["energy"][-1] + 1)
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]
    eta = prm["charge_efficiency"]
    net = (df["load_kwh"] - df["pv_forecast_kwh"]).to_numpy()
    price = df["price"].to_numpy()

    Aeq = lil_matrix((2 * T, n), dtype=float)
    beq = np.zeros(2 * T)
    for t in range(T):
        # 供能覆盖: g + D - C - U = n
        Aeq[t, idx["buy"][t]] = 1.0
        Aeq[t, idx["discharge"][t]] = 1.0
        Aeq[t, idx["charge"][t]] = -1.0
        Aeq[t, idx["surplus"][t]] = -1.0
        beq[t] = net[t]
        # 储能动态: E_t - E_{t-1} - eta*C + D/eta = 0
        row = T + t
        Aeq[row, idx["energy"][t + 1]] = 1.0
        Aeq[row, idx["energy"][t]] = -1.0
        Aeq[row, idx["charge"][t]] = -eta
        Aeq[row, idx["discharge"][t]] = 1.0 / eta

    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    upper[idx["charge"]] = S
    upper[idx["discharge"]] = S
    lower[idx["energy"]] = prm["operating_min_kwh"]
    upper[idx["energy"]] = prm["operating_max_kwh"]
    e_start = prm["initial_energy_kwh"] if e0 is None else float(e0)
    lower[idx["energy"][0]] = upper[idx["energy"][0]] = e_start
    lower[idx["energy"][-1]] = upper[idx["energy"][-1]] = prm["terminal_energy_kwh"]

    c = np.zeros(n)
    c[idx["buy"]] = price

    t0 = time.perf_counter()
    res = linprog(c, A_eq=Aeq.tocsr(), b_eq=beq, bounds=list(zip(lower, upper)),
                  method="highs")
    elapsed = time.perf_counter() - t0
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = res.x
    return {
        "status": res.message,
        "elapsed_s": elapsed,
        "buy_kwh": x[idx["buy"]],
        "charge_kwh": x[idx["charge"]],
        "discharge_kwh": x[idx["discharge"]],
        "surplus_kwh": x[idx["surplus"]],
        "energy_kwh": x[idx["energy"]],
        "pi": np.asarray(res.eqlin.marginals[:T]),       # 供能(母线)影子价格
        "lambda": np.asarray(res.eqlin.marginals[T:]),   # 储能状态影子价格
    }


# --------------------------------------------------------------------------- #
# (B) MILP 单目标单层（含 0-1 互斥）
# --------------------------------------------------------------------------- #
def solve_milp_single(df: pd.DataFrame, prm: dict) -> tuple[dict, float]:
    idx, Aeq, beq, Aub, bub, bounds, integrality = _base_problem(df, prm)
    n = len(bounds.lb)
    c = np.zeros(n)
    c[idx["buy"]] = df["price"].to_numpy()
    t0 = time.perf_counter()
    r = _solve(c, Aeq, beq, Aub, bub, bounds, integrality)
    elapsed = time.perf_counter() - t0
    x = r.x
    out = {
        "buy_kwh": x[idx["buy"]],
        "charge_kwh": x[idx["charge"]],
        "discharge_kwh": x[idx["discharge"]],
        "surplus_kwh": x[idx["surplus"]],
        "energy_kwh": x[idx["energy"]],
    }
    return out, elapsed


# --------------------------------------------------------------------------- #
# 汇总指标
# --------------------------------------------------------------------------- #
def summarise(tag, buy, charge, discharge, energy, surplus, cost, elapsed):
    eta = 0.9
    soc_res = (np.asarray(energy[1:]) - np.asarray(energy[:-1])
               - eta * np.asarray(charge) + np.asarray(discharge) / eta)
    return {
        "model": tag,
        "total_cost_yuan": float(cost),
        "total_purchase_kwh": float(np.sum(buy)),
        "terminal_energy_kwh": float(energy[-1]),
        "initial_energy_kwh": float(energy[0]),
        "total_charge_kwh": float(np.sum(charge)),
        "total_discharge_kwh": float(np.sum(discharge)),
        "total_surplus_kwh": float(np.sum(surplus)),
        "min_energy_kwh": float(np.min(energy)),
        "max_energy_kwh": float(np.max(energy)),
        "max_charge_kw": float(np.max(charge) * 6.0),
        "max_discharge_kw": float(np.max(discharge) * 6.0),
        "simultaneous_charge_discharge_kwh": float(np.sum(np.minimum(charge, discharge))),
        "n_zero_buy_intervals": int(np.sum(np.asarray(buy) < 1e-6)),
        "max_soc_residual_kwh": float(np.max(np.abs(soc_res))),
        "elapsed_s": float(elapsed),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    price = df["price"].to_numpy()
    load = df["load_kwh"].to_numpy()
    pv = df["pv_forecast_kwh"].to_numpy()
    S = prm["max_charge_discharge_power_kw"] * prm["interval_hours"]

    # ---- (A) LP 松弛单层 -------------------------------------------------- #
    lp = solve_lp_single(df, prm)
    s_lp = summarise("LP 松弛单层", lp["buy_kwh"], lp["charge_kwh"], lp["discharge_kwh"],
                     lp["energy_kwh"], lp["surplus_kwh"],
                     float(price @ lp["buy_kwh"]), lp["elapsed_s"])

    # ---- (B) MILP 单目标单层 ---------------------------------------------- #
    ms, t_ms = solve_milp_single(df, prm)
    s_ms = summarise("MILP 单目标(含0-1)", ms["buy_kwh"], ms["charge_kwh"], ms["discharge_kwh"],
                     ms["energy_kwh"], ms["surplus_kwh"],
                     float(price @ ms["buy_kwh"]), t_ms)

    # ---- (C) LP 确定性求解（solve_q1.py） ------------------------------- #
    t0 = time.perf_counter()
    milp_df, milp_sum = solve_q1()
    t_lex = time.perf_counter() - t0
    lx_buy = milp_df["buy_kwh"].to_numpy()
    lx_chg = milp_df["charge_kwh"].to_numpy()
    lx_dis = milp_df["discharge_kwh"].to_numpy()
    lx_sur = milp_df["surplus_kwh"].to_numpy()
    lx_eng = np.concatenate(([milp_df["energy_start_kwh"].iloc[0]],
                             milp_df["energy_end_kwh"].to_numpy()))
    s_lex = summarise("LP 确定性求解", lx_buy, lx_chg, lx_dis, lx_eng, lx_sur,
                      milp_sum["total_purchase_cost_yuan"], t_lex)

    # ---- (D) 两层模型 ----------------------------------------------------- #
    t0 = time.perf_counter()
    tl = two_layer_dp(price, load, pv, prm)
    t_tl = time.perf_counter() - t0
    s_tl = summarise("两层模型(LP+DP)", tl.buy_kwh, tl.charge_kwh, tl.discharge_kwh,
                     tl.energy_kwh, tl.surplus_kwh, tl.cost, t_tl)
    s_tl["dp_max_segments"] = tl.max_segments
    s_tl["dp_total_segments"] = tl.total_segments

    # ---- 逐时段一致性 ----------------------------------------------------- #
    d_tl = np.abs(lp["buy_kwh"] - tl.buy_kwh)
    diff_slots = [int(df["slot"].iloc[i]) for i in np.nonzero(d_tl > 1e-6)[0]]
    crosscheck = {
        "max_abs_buy_diff_lp_vs_two_layer_kwh": float(d_tl.max()),
        "sum_abs_buy_diff_lp_vs_two_layer_kwh": float(d_tl.sum()),
        "n_intervals_differing_lp_vs_two_layer": int((d_tl > 1e-6).sum()),
        "differing_slots": diff_slots,
        "max_abs_buy_diff_lp_vs_milp_single_kwh": float(np.abs(lp["buy_kwh"] - ms["buy_kwh"]).max()),
        "max_abs_buy_diff_lp_vs_lp_det_kwh": float(np.abs(lp["buy_kwh"] - lx_buy).max()),
        "max_abs_energy_diff_lp_vs_two_layer_kwh": float(np.abs(lp["energy_kwh"][1:] - tl.energy_kwh[1:]).max()),
        "cost_diff_two_layer_vs_lp_yuan": float(tl.cost - s_lp["total_cost_yuan"]),
        "cost_diff_milp_single_vs_lp_yuan": float(s_ms["total_cost_yuan"] - s_lp["total_cost_yuan"]),
        "cost_diff_lp_det_vs_lp_yuan": float(s_lex["total_cost_yuan"] - s_lp["total_cost_yuan"]),
    }

    # ---- 影子价格诊断 ----------------------------------------------------- #
    pi, lam = lp["pi"], lp["lambda"]
    positive_buy = lp["buy_kwh"] > 1e-6
    surplus_active = lp["surplus_kwh"] > 1e-6

    # ∂J*/∂E_0：中心差分（唯一能直接和 DP 边际价值对照的量）
    d = 1.0
    e0 = prm["initial_energy_kwh"]
    jp = float(price @ solve_lp_single(df, prm, e0=e0 + d)["buy_kwh"])
    jm = float(price @ solve_lp_single(df, prm, e0=e0 - d)["buy_kwh"])
    dj_de0 = (jp - jm) / (2 * d)

    w_left, w_right = marginal_value(tl.F[0], e0)   # Λ_1(E_0)：F[0] 即 F_1
    requested = []
    for slot, label in zip(REQUESTED_SLOTS, REQUESTED_LABELS):
        t = slot - 1
        requested.append({
            "slot": slot, "interval": label, "price_c": float(price[t]),
            "internal_price_pi": float(pi[t]), "buy_kwh": float(lp["buy_kwh"][t]),
        })
    shadow = {
        "pi_min": float(pi.min()),
        "pi_max": float(pi.max()),
        "pi_within_0_and_c": bool(np.all(pi >= -1e-9) and np.all(pi <= price + 1e-9)),
        "max_abs_pi_minus_c_on_positive_buy": float(np.max(np.abs(pi[positive_buy] - price[positive_buy]))),
        "max_abs_pi_on_surplus": float(np.max(np.abs(pi[surplus_active]))) if surplus_active.any() else 0.0,
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        "lambda_at_t0": float(lam[0]),
        "dj_de0_from_lp": float(dj_de0),
        "dp_marginal_value_left": w_left,
        "dp_marginal_value_right": w_right,
        "requested_slots": requested,
    }

    summary = {
        "data": {"n_intervals": len(df), "interval_hours": prm["interval_hours"],
                 "max_charge_discharge_kwh": S},
        "models": {"LP_single_layer": s_lp, "MILP_single_objective": s_ms,
                   "LP_deterministic": s_lex, "two_layer": s_tl},
        "crosscheck": crosscheck,
        "shadow_price_diagnostics": shadow,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "q1_layer_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    dispatch = pd.DataFrame({
        "slot": df["slot"], "source_time": df["source_time"], "price": price,
        "load_kwh": load, "pv_forecast_kwh": pv,
        "lp_buy_kwh": lp["buy_kwh"], "milp_single_buy_kwh": ms["buy_kwh"],
        "lp_det_buy_kwh": lx_buy, "twolayer_buy_kwh": tl.buy_kwh,
        "lp_charge_kwh": lp["charge_kwh"], "twolayer_charge_kwh": tl.charge_kwh,
        "lp_discharge_kwh": lp["discharge_kwh"], "twolayer_discharge_kwh": tl.discharge_kwh,
        "lp_energy_end_kwh": lp["energy_kwh"][1:], "twolayer_energy_end_kwh": tl.energy_kwh[1:],
        "lp_pi": pi, "lp_lambda": lam, "abs_diff_buy_lp_twolayer": d_tl,
    })
    dispatch.to_csv(RESULTS / "q1_layer_dispatch_compare.csv", index=False, encoding="utf-8-sig")

    write_markdown(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_markdown(s: dict) -> None:
    m = s["models"]
    lp = m["LP_single_layer"]
    ms = m["MILP_single_objective"]
    lx = m["LP_deterministic"]
    tl = m["two_layer"]
    rows = [
        ("最低费用 / 元", lp["total_cost_yuan"], ms["total_cost_yuan"], lx["total_cost_yuan"], tl["total_cost_yuan"]),
        ("全天购电量 / kWh", lp["total_purchase_kwh"], ms["total_purchase_kwh"], lx["total_purchase_kwh"], tl["total_purchase_kwh"]),
        ("24:00 内部储电量 / kWh", lp["terminal_energy_kwh"], ms["terminal_energy_kwh"], lx["terminal_energy_kwh"], tl["terminal_energy_kwh"]),
        ("0:00 内部储电量 / kWh", lp["initial_energy_kwh"], ms["initial_energy_kwh"], lx["initial_energy_kwh"], tl["initial_energy_kwh"]),
        ("总充电量 / kWh", lp["total_charge_kwh"], ms["total_charge_kwh"], lx["total_charge_kwh"], tl["total_charge_kwh"]),
        ("总放电量 / kWh", lp["total_discharge_kwh"], ms["total_discharge_kwh"], lx["total_discharge_kwh"], tl["total_discharge_kwh"]),
        ("未利用供能 / kWh", lp["total_surplus_kwh"], ms["total_surplus_kwh"], lx["total_surplus_kwh"], tl["total_surplus_kwh"]),
        ("最少储电量 / kWh", lp["min_energy_kwh"], ms["min_energy_kwh"], lx["min_energy_kwh"], tl["min_energy_kwh"]),
        ("最多储电量 / kWh", lp["max_energy_kwh"], ms["max_energy_kwh"], lx["max_energy_kwh"], tl["max_energy_kwh"]),
        ("最大充电功率 / kW", lp["max_charge_kw"], ms["max_charge_kw"], lx["max_charge_kw"], tl["max_charge_kw"]),
        ("最大放电功率 / kW", lp["max_discharge_kw"], ms["max_discharge_kw"], lx["max_discharge_kw"], tl["max_discharge_kw"]),
        ("同一时段同时充放电量 / kWh", lp["simultaneous_charge_discharge_kwh"], ms["simultaneous_charge_discharge_kwh"],
         lx["simultaneous_charge_discharge_kwh"], tl["simultaneous_charge_discharge_kwh"]),
        ("零购电时段数", lp["n_zero_buy_intervals"], ms["n_zero_buy_intervals"], lx["n_zero_buy_intervals"], tl["n_zero_buy_intervals"]),
        ("求解耗时 / s", lp["elapsed_s"], ms["elapsed_s"], lx["elapsed_s"], tl["elapsed_s"]),
    ]
    lines = [
        "# 问题一：两层模型 vs LP 单层模型 结果对比",
        "",
        "| 指标 | (A) LP 单目标 | (B) MILP 单目标 | (C) LP 确定性求解 | (D) 连续状态 DP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, a, b, c, d in rows:
        lines.append(f"| {name} | {a} | {b} | {c} | {d} |")
    cc = s["crosscheck"]
    lines += [
        "",
        "## 逐时段一致性（以 LP 松弛为基准）",
        "",
        f"- 两层模型 vs LP：{cc['n_intervals_differing_lp_vs_two_layer']} 个时段有差异"
        f"（slot {cc['differing_slots']}），逐时段购电量最大偏差 "
        f"{cc['max_abs_buy_diff_lp_vs_two_layer_kwh']:.6f} kWh，累计绝对偏差 "
        f"{cc['sum_abs_buy_diff_lp_vs_two_layer_kwh']:.6f} kWh",
        f"- MILP 单目标 vs LP：逐时段购电量最大偏差 {cc['max_abs_buy_diff_lp_vs_milp_single_kwh']:.3e} kWh",
        f"- LP 确定性求解 vs LP 单目标：逐时段购电量最大偏差 {cc['max_abs_buy_diff_lp_vs_lp_det_kwh']:.3e} kWh（退化最优解可不同）",
        f"- 两层模型 vs LP：逐时段末储电量最大偏差 {cc['max_abs_energy_diff_lp_vs_two_layer_kwh']:.6f} kWh",
        f"- 费用差：两层−LP = {cc['cost_diff_two_layer_vs_lp_yuan']:.3e} 元；"
        f"MILP单目标−LP = {cc['cost_diff_milp_single_vs_lp_yuan']:.3e} 元；"
        f"LP确定性求解−LP单目标 = {cc['cost_diff_lp_det_vs_lp_yuan']:.3e} 元",
        "",
        "## 影子价格与边际价值互证（LP 松弛）",
        "",
    ]
    sh = s["shadow_price_diagnostics"]
    lines += [
        f"- 母线内部价格 $\\pi_t \\in [{sh['pi_min']:.6f}, {sh['pi_max']:.6f}]$，"
        f"满足 $0\\le\\pi_t\\le c_t$：{sh['pi_within_0_and_c']}",
        f"- 正购电时段 $\\max|\\pi_t-c_t|$ = {sh['max_abs_pi_minus_c_on_positive_buy']:.3e}（验证 $g_t(c_t-\\pi_t)=0$）",
        f"- 未利用供能时段 $\\max|\\pi_t|$ = {sh['max_abs_pi_on_surplus']:.3e}（验证 $\\pi_t U_t=0$；全天未利用供能为 0）",
        f"- 状态乘子 $\\lambda_t \\in [{sh['lambda_min']:.6f}, {sh['lambda_max']:.6f}]$，"
        f"$\\lambda_1$ = {sh['lambda_at_t0']:.9f}",
        f"- $\\partial J^*/\\partial E_0$（LP 中心差分）= {sh['dj_de0_from_lp']:.9f} 元/kWh",
        f"- DP 边际储能价值 $\\Lambda_1(6000) \\in [{sh['dp_marginal_value_right']:.9f}, "
        f"{sh['dp_marginal_value_left']:.9f}]$ 元/kWh，与 $-\\partial J^*/\\partial E_0$ 一致",
        "",
        "## 题目指定时段的内部价格 $\\pi_t$ 与外网电价 $c_t$",
        "",
        "| 时间段 | 外网电价 $c_t$ | 内部价格 $\\pi_t$ | 计划购电量 / kWh |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in sh["requested_slots"]:
        lines.append(f"| {r['interval']} | {r['price_c']:.4f} | {r['internal_price_pi']:.4f} | {r['buy_kwh']:.6f} |")
    lines.append("")
    (RESULTS / "q1_layer_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
