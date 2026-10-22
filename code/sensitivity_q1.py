# -*- coding: utf-8 -*-
"""问题1 计划购电模型（LP）的灵敏性 / 鲁棒性分析。

对储能设备的三个关键参数——充放电效率 η、储能容量上限 Cmax、最大充放电功率
Pmax——分别在其基准值的 ±20%、±10% 范围内摄动，其余参数保持不变，重新求解
线性规划模型，记录全天购电费用，考察费用对各参数的敏感程度。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path, add_style_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()
add_style_path()

from solve_q1 import _lp_problem  # noqa: E402


def _solve_cost(df: pd.DataFrame, prm: dict) -> float:
    """以给定参数求解问题1 LP，返回全天购电费用（元）。"""
    idx, Aeq, beq, lower, upper = _lp_problem(df, prm)
    price = df["price"].to_numpy()
    c = np.zeros(len(lower))
    c[idx["buy"]] = price
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=list(zip(lower, upper)), method="highs")
    if not r.success:
        raise RuntimeError(f"LP failed: {r.message}")
    return float(c @ r.x)


# 各参数的基准值、扰动因子与描述（用于表格与图例）
DELTAS = [-0.20, -0.10, 0.0, 0.10, 0.20]

PARAMS = [
    {
        "key": "efficiency",
        "label": "充放电效率 $\\eta$",
        "legend": "充放电效率η",
        "base": 0.9,
        "unit": "",
        "apply": lambda prm, v: {
            **prm,
            "charge_efficiency": v,
            "discharge_efficiency": v,
        },
    },
    {
        "key": "capacity",
        "label": "储能容量上限 $C_{\\max}$",
        "legend": "储能容量上限C_max",
        "base": 10800.0,
        "unit": "kWh",
        "apply": lambda prm, v: {**prm, "operating_max_kwh": v},
    },
    {
        "key": "power",
        "label": "最大充放电功率 $P_{\\max}$",
        "legend": "最大充放电功率P_max",
        "base": 5000.0,
        "unit": "kW",
        "apply": lambda prm, v: {**prm, "max_charge_discharge_power_kw": v},
    },
]


def main() -> None:
    df = pd.read_csv(ROOT / "data_processed" / "q1_typical_day.csv")
    prm = json.loads((ROOT / "data_processed" / "q1_storage_parameters.json")
                     .read_text(encoding="utf-8"))
    baseline = _solve_cost(df, prm)

    rows: list[dict] = []
    for p in PARAMS:
        costs = []
        for delta in DELTAS:
            value = p["base"] * (1.0 + delta)
            cost = _solve_cost(df, p["apply"](prm, value))
            costs.append(cost)
            rows.append({
                "parameter": p["legend"],
                "key": p["key"],
                "delta_pct": round(delta * 100, 1),
                "value": round(value, 6),
                "cost_yuan": round(cost, 4),
                "cost_change_pct": round((cost - baseline) / baseline * 100, 4),
            })

    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / "q1_sensitivity.csv", index=False, encoding="utf-8-sig")

    # 宽表（参数 × 扰动档位）便于写入论文
    pivot = result.pivot(index="parameter", columns="delta_pct", values="cost_yuan")
    pivot = pivot[[d * 100 for d in DELTAS]]
    pivot.to_csv(ROOT / "results" / "q1_sensitivity_wide.csv", encoding="utf-8-sig")

    summary = {"baseline_cost_yuan": baseline, "table": rows}
    (ROOT / "results" / "q1_sensitivity.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # —— 绘制灵敏性曲线（单图三条曲线，共享相对变化横轴）——
    from plot_style import apply_publication_style, export_figure
    import matplotlib.pyplot as plt

    style = apply_publication_style(journal="nature", lang="zh")
    colors = style["colors"]
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    x = np.array([d * 100 for d in DELTAS])
    for p, color in zip(PARAMS, colors):
        sub = result[result["key"] == p["key"]]
        ax.plot(x, sub["cost_yuan"], marker="o", ms=4, lw=1.6, color=color, label=p["legend"])
    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("参数相对基准值的变化 / %")
    ax.set_ylabel("全天购电费用 / 元")
    ax.set_xticks(x)
    ax.legend(ncol=1, frameon=False, loc="best")
    export_figure(fig, ROOT / "figures" / "process_q1_sensitivity", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
