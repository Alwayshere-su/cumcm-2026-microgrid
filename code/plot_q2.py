"""Publication figures for the official Q2 method-3 evidence chain."""

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_style_path  # noqa: E402
ROOT = find_project_root(__file__)
FIG = ROOT / "figures"
add_style_path()
from plot_style import apply_publication_style, export_figure

REPS = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])


def main() -> None:
    FIG.mkdir(exist_ok=True)
    colors = apply_publication_style(journal="nature", lang="zh")["colors"]
    forecast = pd.read_csv(ROOT / "data_processed" / "q2_forecasts_reg.csv", parse_dates=["date"])
    dispatch = pd.read_csv(ROOT / "results" / "q2_method3_dispatch.csv", parse_dates=["date"])
    comparison = json.loads((ROOT / "results" / "q2_three_methods_comparison.json").read_text(encoding="utf-8"))

    # raw 1: four representative net-load forecasts.
    fig, axes = plt.subplots(4, 1, figsize=(7.1, 7.0), sharex=True)
    for ax, day in zip(axes, REPS):
        x = forecast[forecast.date == day].sort_values("slot")
        hour = (x.slot - 1) / 6
        ax.plot(hour, (x.actual_load_kwh - x.actual_pv_kwh) * 6, color=colors[0], lw=1, label="实际")
        ax.plot(hour, (x.reg_load_kwh - x.reg_pv_kwh) * 6, color=colors[1], ls="--", lw=1, label="Ridge预测")
        ax.text(.01, .82, day.strftime("%m-%d"), transform=ax.transAxes)
        ax.set_ylabel("净负荷/kW")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "raw_q2_representative_forecast", dpi=300)

    # raw 2: net-load residual autocorrelation, motivating whole-day blocks.
    residual = ((forecast.actual_load_kwh - forecast.actual_pv_kwh)
                - (forecast.reg_load_kwh - forecast.reg_pv_kwh)).to_numpy(float)
    ac = [pd.Series(residual).autocorr(lag=i) for i in range(1, 37)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(np.arange(1, 37) * 10, ac, color=colors[0])
    ax.axhline(0, color="black", lw=.6)
    ax.set(xlabel="滞后时间/min", ylabel="净负荷残差自相关系数")
    export_figure(fig, FIG / "raw_q2_residual_autocorrelation", dpi=300)

    # raw 3: monthly regression improvement over the previous-day baseline.
    z = forecast.copy()
    z["month"] = z.date.dt.month
    actual_net = z.actual_load_kwh - z.actual_pv_kwh
    z["naive"] = np.abs((z.naive_load_kwh - z.naive_pv_kwh) - actual_net) * 6
    z["ridge"] = np.abs((z.reg_load_kwh - z.reg_pv_kwh) - actual_net) * 6
    monthly = z.groupby("month")[["naive", "ridge"]].mean()
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(monthly.index, monthly.naive, marker="o", label="前一日同槽", color=colors[1])
    ax.plot(monthly.index, monthly.ridge, marker="s", label="Ridge回归", color=colors[0])
    ax.set(xlabel="月份", ylabel="净负荷MAE/kW", xticks=range(1, 13))
    ax.legend(frameon=False)
    export_figure(fig, FIG / "raw_q2_monthly_forecast_error", dpi=300)

    # process 1: 30 paired whole-day residual scenarios around one forecast center.
    day = pd.Timestamp("2025-09-23")
    center = forecast[forecast.date == day].sort_values("slot")
    history = forecast[forecast.date < day].copy()
    history["net_resid"] = ((history.actual_load_kwh - history.actual_pv_kwh)
                            - (history.reg_load_kwh - history.reg_pv_kwh))
    curves = history[history.date.isin(sorted(history.date.unique())[-30:])].pivot(
        index="date", columns="slot", values="net_resid")
    center_net = (center.reg_load_kwh - center.reg_pv_kwh).to_numpy(float) * 6
    hour = np.arange(144) / 6
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    for row in curves.to_numpy(float):
        ax.plot(hour, center_net + row * 6, color=colors[2], alpha=.18, lw=.55)
    ax.plot(hour, center_net, color=colors[0], lw=1.2, label="回归情景中心")
    ax.set(xlabel="时刻/h", ylabel="情景净负荷/kW")
    ax.legend(frameon=False)
    export_figure(fig, FIG / "process_q2_joint_residual_scenarios", dpi=300)

    # process 2: dynamic retention level and realized SOC on a representative day.
    x = dispatch[dispatch.date == day].sort_values("slot")
    hour = (x.slot - 1) / 6
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.fill_between(hour, 1200, x.retention_kwh, color=colors[2], alpha=.45, label="保留区")
    ax.plot(hour, x.retention_kwh, color=colors[1], ls="--", lw=1, label="动态保留水位")
    ax.plot(hour, x.energy_start_kwh, color=colors[0], lw=1.1, label="实际SOC")
    ax.set(xlabel="时刻/h", ylabel="储电量/kWh", ylim=(1000, 11000))
    ax.legend(frameon=False, ncol=3)
    export_figure(fig, FIG / "process_q2_retention_policy", dpi=300)

    # process 3: forecast hit rates, keeping point accuracy distinct from scenarios.
    accuracy = json.loads((ROOT / "results" / "q2_accuracy_comparison.json").read_text(encoding="utf-8"))
    ridge = accuracy["method1_regression"]
    labels = ["负荷±10%", "负荷±20%", "光伏±20%", "光伏±30%"]
    values = [ridge["load_重合率@10%"], ridge["load_重合率@20%"],
              ridge["pv_白昼重合率@20%"], ridge["pv_白昼重合率@30%"]]
    values = np.asarray(values, float)
    if values.max() <= 1.01:
        values *= 100
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(labels, values, color=[colors[0], colors[0], colors[1], colors[1]])
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set(ylabel="命中率/%", ylim=(0, 105))
    export_figure(fig, FIG / "process_q2_forecast_hit_rates", dpi=300)

    # result 1: three-method ablation with cost decomposition.
    keys = ["method1_reg_detLP", "method2_stat_SP_DP", "method3_reg_SP_DP"]
    labels = ["回归+确定性LP", "统计+SP-DP", "回归+SP-DP"]
    plan = np.array([comparison[k]["plan_cost_yuan"] for k in keys]) / 1e4
    emergency = np.array([comparison[k]["emergency_cost_yuan"] for k in keys]) / 1e4
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pos = np.arange(3)
    ax.bar(pos, plan, color=colors[0], label="计划购电费")
    ax.bar(pos, emergency, bottom=plan, color=colors[1], label="紧急购电费")
    ax.set(xticks=pos, xticklabels=labels, ylabel="全年费用/万元")
    ax.legend(frameon=False)
    export_figure(fig, FIG / "result_q2_cost_comparison", dpi=300)

    # result 2: annual emergency-purchase calendar for method 3.
    daily = dispatch.groupby("date").emergency_kwh.sum()
    matrix = np.full((12, 31), np.nan)
    for date, value in daily.items():
        matrix[pd.Timestamp(date).month - 1, pd.Timestamp(date).day - 1] = value
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    # 零紧急购电显示为纯白；正值从浅黄连续过渡到深红。
    base = plt.get_cmap("YlOrRd")
    emergency_cmap = LinearSegmentedColormap.from_list(
        "WhiteYlOrRd", [(0.0, "#ffffff"), (0.015, base(0.02)),
                         (0.35, base(0.35)), (0.7, base(0.7)),
                         (1.0, base(1.0))])
    emergency_cmap.set_bad("#ffffff")
    shown = np.ma.masked_where(np.isnan(matrix) | (matrix <= 1e-9), matrix)
    image = ax.imshow(shown, aspect="auto", cmap=emergency_cmap, vmin=0)
    ax.set(xlabel="日", ylabel="月份", xticks=[0, 4, 9, 14, 19, 24, 30],
           xticklabels=[1, 5, 10, 15, 20, 25, 31], yticks=range(12), yticklabels=range(1, 13))
    fig.colorbar(image, ax=ax, label="紧急购电量/kWh")
    export_figure(fig, FIG / "result_q2_emergency_calendar", dpi=300)

    # result 3: official dates; power and inventory are separated to avoid dual axes.
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.0), sharex=True)
    for row, day in enumerate(REPS):
        x = dispatch[dispatch.date == day].sort_values("slot")
        hour = (x.slot - 1) / 6
        axes[row, 0].plot(hour, x.plan_buy_kwh * 6, color=colors[0], label="计划购电")
        axes[row, 0].fill_between(hour, 0, x.emergency_kwh * 6, color=colors[1], alpha=.55, label="紧急购电")
        axes[row, 0].set_ylabel("功率/kW")
        axes[row, 0].text(.01, .82, day.strftime("%m-%d"), transform=axes[row, 0].transAxes)
        axes[row, 1].plot(hour, x.energy_start_kwh, color=colors[3], lw=1, label="SOC")
        axes[row, 1].plot(hour, x.retention_kwh, color=colors[1], ls="--", lw=.9, label="保留水位")
        axes[row, 1].set_ylabel("储电量/kWh")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("时刻/h")
    axes[-1, 1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "result_q2_representative_dispatch", dpi=300)
    plt.close("all")


if __name__ == "__main__":
    main()
