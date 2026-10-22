from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_style_path  # noqa: E402
ROOT = find_project_root(__file__)
add_style_path()
from plot_style import apply_publication_style, export_figure


def main():
    out = ROOT / "figures"
    r = pd.read_csv(ROOT / "results" / "q1_dispatch_10min.csv")
    r["net_load_kwh"] = r["load_kwh"] - r["pv_forecast_kwh"]
    style = apply_publication_style(journal="nature", lang="zh")
    colors = style["colors"]
    t = np.arange(144) / 6
    ticks = np.arange(0, 25, 4)

    # raw 1: source profiles
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(t, r.load_kw, color=colors[0], lw=1.7, label="负荷")
    axes[0].plot(t, r.pv_forecast_kw, color=colors[1], lw=1.7, ls="--", label="光伏预测")
    axes[0].set_ylabel("功率/kW"); axes[0].legend(ncol=2, frameon=False)
    axes[1].fill_between(t, r.net_load_kwh, 0, color=colors[2], alpha=.65)
    axes[1].axhline(0, color="0.35", lw=.8); axes[1].set_ylabel("净负荷电量/kWh")
    axes[2].step(t, r.price, where="post", color=colors[3], lw=1.5)
    axes[2].set_ylabel("电价/(元/kWh)"); axes[2].set_xlabel("时刻/h")
    axes[2].set_xticks(ticks)
    export_figure(fig, out / "raw_q1_source_profiles", dpi=300); plt.close(fig)

    # raw 2: load-PV phase relation
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    sc = ax.scatter(r.pv_forecast_kw, r.load_kw, c=t, cmap="viridis", s=22, alpha=.8,
                    edgecolors="none")
    lim = max(r.pv_forecast_kw.max(), r.load_kw.max())
    ax.plot([0, lim], [0, lim], color="0.35", ls="--", lw=1, label="光伏=负荷")
    ax.set_xlabel("光伏预测功率/kW"); ax.set_ylabel("负荷功率/kW")
    ax.legend(frameon=False); cbar=fig.colorbar(sc, ax=ax); cbar.set_label("时刻/h")
    export_figure(fig, out / "raw_q1_load_pv_relation", dpi=300); plt.close(fig)

    # raw 3: distributions
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    sns.histplot(r.net_load_kwh, bins=18, kde=True, color=colors[0], ax=axes[0])
    axes[0].axvline(0, color=colors[1], ls="--", lw=1); axes[0].set_xlabel("净负荷电量/kWh"); axes[0].set_ylabel("时段数")
    sns.histplot(r.price, bins=14, kde=True, color=colors[1], ax=axes[1])
    axes[1].set_xlabel("电价/(元/kWh)"); axes[1].set_ylabel("时段数")
    export_figure(fig, out / "raw_q1_netload_price_distribution", dpi=300); plt.close(fig)

    # process 1: actions
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True)
    axes[0].step(t, r.buy_kwh, where="post", color=colors[0], lw=1.5, label="计划购电")
    axes[0].step(t, r.load_kwh, where="post", color="0.35", lw=1.0, ls="--", label="负荷电量")
    axes[0].set_ylabel("电量/kWh"); axes[0].legend(ncol=2, frameon=False)
    axes[1].bar(t, r.charge_kwh, width=1/6, color=colors[2], label="充电")
    axes[1].bar(t, -r.discharge_kwh, width=1/6, color=colors[1], label="放电")
    axes[1].axhline(0, color="0.3", lw=.8); axes[1].set_ylabel("储能动作/kWh")
    axes[1].set_xlabel("时刻/h"); axes[1].set_xticks(ticks); axes[1].legend(ncol=2, frameon=False)
    export_figure(fig, out / "process_q1_dispatch_actions", dpi=300); plt.close(fig)

    # process 2: SOC and bounds
    soc_t = np.arange(145) / 6
    soc = np.r_[r.energy_start_kwh.iloc[0], r.energy_end_kwh.to_numpy()]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.fill_between(soc_t, 1200, 10800, color=colors[2], alpha=.18, label="允许运行区间")
    ax.plot(soc_t, soc, color=colors[0], lw=1.8, label="储电量")
    ax.axhline(6000, color=colors[1], ls="--", lw=1, label="初末储电量")
    ax.set_xlim(0,24); ax.set_xticks(ticks); ax.set_xlabel("时刻/h"); ax.set_ylabel("储电量/kWh")
    ax.legend(ncol=3, frameon=False)
    export_figure(fig, out / "process_q1_soc_constraints", dpi=300); plt.close(fig)

    # process 3: active constraints heatmap
    active = np.vstack([
        (r.charge_kwh > 1e-6).astype(int),
        (r.discharge_kwh > 1e-6).astype(int),
        (r.energy_end_kwh <= 1200 + 1e-5).astype(int),
        (r.energy_end_kwh >= 10800 - 1e-5).astype(int),
    ])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.heatmap(active, cmap=sns.color_palette(["#F3F5F7", colors[0]], as_cmap=True),
                cbar=False, yticklabels=["充电", "放电", "SOC下限", "SOC上限"], ax=ax)
    ax.set_xticks(np.arange(0,145,24)); ax.set_xticklabels([str(x) for x in ticks])
    ax.set_xlabel("时刻/h"); ax.set_ylabel(""); ax.tick_params(axis="y", labelsize=8.5)
    export_figure(fig, out / "process_q1_active_constraints", dpi=300); plt.close(fig)

    # result 1: supply-demand balance
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.stackplot(t, r.pv_forecast_kwh, r.buy_kwh, r.discharge_kwh,
                 colors=[colors[2], colors[0], colors[1]], alpha=.82,
                 labels=["光伏", "计划购电", "储能放电"])
    ax.plot(t, r.load_kwh + r.charge_kwh, color="#222222", lw=1.3,
            label="负荷+储能充电")
    ax.set_xlim(0,24); ax.set_xticks(ticks); ax.set_xlabel("时刻/h"); ax.set_ylabel("电量/kWh")
    ax.legend(ncol=2, frameon=False)
    export_figure(fig, out / "result_q1_energy_balance", dpi=300); plt.close(fig)

    # result 2: cost comparison
    import json
    s = json.loads((ROOT / "results" / "q1_summary.json").read_text(encoding="utf-8"))
    vals = [s["no_storage_cost_yuan"], s["total_purchase_cost_yuan"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bars = ax.bar(["无储能基线", "LP最优调度"], vals, color=["#9AA0A6", colors[0]], width=.58)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals], padding=3, fontsize=9)
    ax.text(.5, min(vals)*.58, f"节省 {s['cost_saving_ratio']:.2%}", ha="center", color=colors[1], fontsize=10)
    ax.set_ylabel("全天购电费/元"); ax.set_ylim(0, max(vals)*1.17)
    export_figure(fig, out / "result_q1_cost_comparison", dpi=300); plt.close(fig)

    # result 3: four-hour storage aggregation
    block = pd.read_csv(ROOT / "results" / "q1_table2_storage.csv")
    y = np.arange(len(block)); h=.34
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh(y-h/2, block.charge_kwh, height=h, color=colors[2], label="充电量")
    ax.barh(y+h/2, block.discharge_kwh, height=h, color=colors[1], label="放电量")
    ax.set_yticks(y); ax.set_yticklabels(block.interval); ax.invert_yaxis()
    ax.set_xlabel("电量/kWh"); ax.set_ylabel("时间段"); ax.legend(frameon=False)
    export_figure(fig, out / "result_q1_storage_blocks", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
