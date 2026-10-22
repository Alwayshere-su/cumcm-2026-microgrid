"""问题四最终模型候选图：raw/process/result 各三张。"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path, add_style_path  # noqa: E402
ROOT = find_project_root(__file__)
FIG, RES = ROOT / "figures", ROOT / "results"
add_style_path()
add_script_path()
from plot_style import apply_publication_style, export_figure
from forecast_q4_price import day_ahead_price, intraday_update

REPS = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])


def main() -> None:
    FIG.mkdir(exist_ok=True)
    colors = apply_publication_style(journal="nature", lang="zh")["colors"]
    data = pd.read_csv(ROOT / "data_processed" / "microgrid_10min.csv",
                       parse_dates=["date"])
    daily = pd.read_csv(RES / "harness_q4_daily.csv", parse_dates=["date"])
    seg = pd.read_csv(RES / "q4_segments.csv", parse_dates=["date"])
    summary = json.loads((RES / "harness_q4_summary.json").read_text("utf-8"))
    dates = pd.Index(sorted(data.date.unique()))
    price = data.pivot(index="date", columns="slot", values="price_dynamic").loc[dates].to_numpy()
    fixed = data.pivot(index="date", columns="slot", values="price_fixed").loc[dates].to_numpy()
    hour = np.arange(144) / 6

    # raw 1: four representative dynamic-price days.
    fig, axes = plt.subplots(4, 1, figsize=(7.1, 6.7), sharex=True)
    for ax, day in zip(axes, REPS):
        i = dates.get_loc(day)
        ax.step(hour, price[i], where="post", color=colors[0], lw=1.0,
                label="实时波动电价")
        ax.step(hour, fixed[i], where="post", color=colors[2], lw=.8, ls="--",
                label="附件1固定分时价")
        ax.text(.01, .78, day.strftime("%m-%d"), transform=ax.transAxes)
        ax.set_ylabel("元/kWh")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "raw_q4_representative_prices", dpi=300)

    # raw 2: annual dynamic-price heatmap.
    test_price = price[dates >= pd.Timestamp("2025-02-01")]
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    im = ax.imshow(test_price, aspect="auto", cmap="YlOrRd", extent=[0, 24, 12, 2])
    ax.set(xlabel="时刻/h", ylabel="月份", yticks=range(2, 13))
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label("电价/(元·kWh$^{-1}$)")
    export_figure(fig, FIG / "raw_q4_dynamic_price_heatmap", dpi=300)

    # raw 3: normalized shape distribution and fixed-price robust prior.
    norm = test_price / np.maximum(test_price.mean(axis=1, keepdims=True), 1e-8)
    q10, med, q90 = np.quantile(norm, [.1, .5, .9], axis=0)
    fshape = fixed[0] / fixed[0].mean()
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.fill_between(hour, q10, q90, color=colors[2], alpha=.28, label="动态价形状10%—90%分位带")
    ax.plot(hour, med, color=colors[0], lw=1.1, label="动态价形状中位数")
    ax.plot(hour, fshape, color=colors[1], lw=1, ls="--", label="固定分时价形状")
    ax.set(xlabel="时刻/h", ylabel="归一化价格形状")
    ax.legend(frameon=False, ncol=3)
    export_figure(fig, FIG / "raw_q4_price_shape_prior", dpi=300)

    # process 1: causal price predictions before and after intraday updates.
    day = pd.Timestamp("2025-09-23"); i = dates.get_loc(day)
    base = day_ahead_price(price, i)
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.step(hour, price[i], where="post", color="black", lw=1.0, label="实现电价")
    ax.plot(hour, base, color=colors[2], ls="--", lw=.9, label="0:00因果预测")
    for s, col, label in [(36, colors[0], "6:00更新"), (72, colors[1], "12:00更新"),
                          (108, colors[3], "18:00更新")]:
        fc = intraday_update(base, price[i], s)
        ax.plot(hour[s:], fc[s:], color=col, lw=.9, label=label)
        ax.axvline(s/6, color="gray", ls=":", lw=.55)
    ax.set(xlabel="时刻/h", ylabel="电价/(元·kWh$^{-1}$)")
    ax.legend(frameon=False, ncol=3)
    export_figure(fig, FIG / "process_q4_causal_price_updates", dpi=300)

    # process 2: four-time confirmation and SOC for one representative day.
    x = seg[(seg.date == day) & (seg.variant == "4-3")].sort_values("slot")
    h = (x.slot - 1) / 6
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.1), sharex=True)
    axes[0].step(h, x.plan_kwh*6, where="post", color=colors[2], ls="--", label="0:00计划")
    axes[0].step(h, x.adjusted_kwh*6, where="post", color=colors[0], label="执行前最后确认量")
    axes[0].fill_between(h, 0, x.emergency_kwh*6, color=colors[1], alpha=.5, label="紧急购电")
    axes[0].set_ylabel("购电功率/kW"); axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(h, x.soc_end_kwh, color=colors[3], lw=1.0)
    axes[1].axhline(1200, color="gray", ls=":", lw=.7)
    axes[1].axhline(10800, color="gray", ls=":", lw=.7)
    axes[1].set(xlabel="时刻/h", ylabel="储电量/kWh")
    for ax in axes:
        for k in (6, 12, 18): ax.axvline(k, color="gray", ls=":", lw=.55)
    export_figure(fig, FIG / "process_q4_four_time_execution", dpi=300)

    # process 3: settlement-derived weighted quantiles, using real paired samples.
    # 取18:00前一段在334个样本外日期上的实际购电量与实现价，展示价格加权经验分布。
    hist = seg[(seg.variant == "4-3") & (seg.slot == 108)].copy()
    hist = hist.sort_values("adjusted_kwh")
    y = hist.adjusted_kwh.to_numpy(float)
    weight = hist.price.to_numpy(float)
    cdf = np.cumsum(weight); cdf /= cdf[-1]
    q50 = y[np.searchsorted(cdf, .5)]; q80 = y[np.searchsorted(cdf, .8)]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.step(y, cdf, where="post", color=colors[0], lw=1.2)
    ax.axhline(.5, color=colors[2], ls="--", lw=.8); ax.axvline(q50, color=colors[2], ls="--", lw=.8,
               label="4-3理论：加权中位数")
    ax.axhline(.8, color=colors[1], ls="--", lw=.8); ax.axvline(q80, color=colors[1], ls="--", lw=.8,
               label="4-2正式：加权4/5分位")
    ax.set(xlabel="历史配对样本购电量/kWh", ylabel="累计价格权重")
    ax.legend(frameon=False)
    export_figure(fig, FIG / "process_q4_weighted_quantiles", dpi=300)

    # result 1: official routes and executable ablations.
    labels = ["4-2固定价", "4-2点预测", "4-2风险计划", "4-3固定形状", "4-3因果点预测", "4-3加权中位数"]
    vals = [summary["total_yuan_42"][k]/1e4 for k in ("Q42_fixed","Q42_point","Q42_nb")]
    vals += [summary["total_yuan"][k]/1e4 for k in ("Q0_fixed","Q1_point","Q2_nb")]
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    bars = ax.bar(np.arange(6), vals, color=[colors[2],colors[2],colors[0],colors[0],colors[2],colors[2]])
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set(xticks=np.arange(6), xticklabels=labels, ylabel="全年费用/万元", ylim=(1200, 1800))
    ax.tick_params(axis="x", rotation=16)
    export_figure(fig, FIG / "result_q4_route_costs", dpi=300)

    # result 2: monthly official cost comparison.
    z = daily.copy(); z["month"] = z.date.dt.month
    monthly = z.groupby("month")[["Q0_fixed_total", "Q42_nb_total"]].sum()/1e4
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.plot(monthly.index, monthly.Q42_nb_total, marker="o", color=colors[1], label="4-2风险计划")
    ax.plot(monthly.index, monthly.Q0_fixed_total, marker="s", color=colors[0], label="4-3稳健滚动")
    ax.set(xlabel="月份", ylabel="月费用/万元", xticks=range(2,13))
    ax.legend(frameon=False)
    export_figure(fig, FIG / "result_q4_monthly_costs", dpi=300)

    # result 3: official dates, purchase and inventory separated.
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.0), sharex=True)
    for row, d in enumerate(REPS):
        xx = seg[(seg.date == d) & (seg.variant == "4-3")].sort_values("slot")
        hh = (xx.slot - 1) / 6
        axes[row,0].step(hh, xx.adjusted_kwh*6, where="post", color=colors[0], label="调整购电")
        axes[row,0].fill_between(hh, 0, xx.emergency_kwh*6, color=colors[1], alpha=.55, label="紧急购电")
        axes[row,0].text(.01,.80,d.strftime("%m-%d"),transform=axes[row,0].transAxes)
        axes[row,0].set_ylabel("功率/kW")
        axes[row,1].plot(hh, xx.soc_end_kwh, color=colors[3], lw=1.0)
        axes[row,1].set_ylabel("储电量/kWh")
    axes[0,0].legend(frameon=False,ncol=2); axes[-1,0].set_xlabel("时刻/h"); axes[-1,1].set_xlabel("时刻/h")
    export_figure(fig, FIG / "result_q4_representative_dispatch", dpi=300)
    plt.close("all")


if __name__ == "__main__":
    main()
