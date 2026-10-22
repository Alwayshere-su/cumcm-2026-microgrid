from __future__ import annotations

import json
import sys
import importlib.metadata
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_style_path, STYLE_DIR  # noqa: E402
ROOT = find_project_root(__file__)
ATT = ROOT / "附件"
OUT = ROOT / "data_processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STYLE_ROOT = STYLE_DIR
add_style_path()
from plot_style import apply_publication_style, export_figure  # noqa: E402


@dataclass
class QualityReport:
    checks: dict[str, bool]
    metrics: dict[str, int | float | str]
    issues: list[str]

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.issues


def _read_wide(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=0)


def build_q1_typical_day(a1: pd.DataFrame) -> pd.DataFrame:
    if a1.shape != (144, 4):
        raise ValueError(f"附件1应为144行×4列，实际为{a1.shape}")
    q1 = pd.DataFrame(
        {
            "slot": np.arange(1, 145),
            "source_time": a1.iloc[:, 0].astype(str),
            "price": pd.to_numeric(a1.iloc[:, 1], errors="coerce"),
            "load_kw": pd.to_numeric(a1.iloc[:, 2], errors="coerce"),
            "pv_forecast_kw": pd.to_numeric(a1.iloc[:, 3], errors="coerce"),
        }
    )
    q1["load_kwh"] = q1["load_kw"] / 6
    q1["pv_forecast_kwh"] = q1["pv_forecast_kw"] / 6
    q1["template_interval"] = _template_intervals()
    return q1


def _wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    dates = pd.to_datetime(df.iloc[:, 0], errors="raise")
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if values.shape[1] != 144:
        raise ValueError(f"{value_name}应有144个时段，实际为{values.shape[1]}")
    arr = values.to_numpy(float).reshape(-1)
    slot = np.tile(np.arange(1, 145), len(df))
    day = np.repeat(dates.to_numpy(), 144)
    result = pd.DataFrame({"date": day, "slot": slot, value_name: arr})
    result["timestamp_end"] = result["date"] + pd.to_timedelta(result["slot"] * 10, unit="m")
    return result


def _template_intervals() -> list[str]:
    template = pd.read_excel(ATT / "附件5" / "result2.xlsx", sheet_name="计划购电量", nrows=0)
    labels = [str(x) for x in template.columns[1:145]]
    if len(labels) != 144:
        raise ValueError("结果模板的10分钟时段列不是144列")
    return labels


def build_time_mapping(source_headers: list[object]) -> pd.DataFrame:
    labels = _template_intervals()
    intervals = pd.DataFrame(
        {
            "template_column": np.arange(2, 146),
            "column_type": "interval",
            "slot": np.arange(1, 145, dtype=float),
            "source_time": [str(x) for x in source_headers],
            "template_interval": labels,
            "physical_start_minute": np.arange(0, 144) * 10,
            "physical_end_minute": np.arange(1, 145) * 10,
            "minutes_from_midnight": np.arange(1, 145) * 10,
        }
    )
    summaries = pd.DataFrame(
        {
            "template_column": [146, 147],
            "column_type": ["daily_total_energy", "daily_total_cost"],
            "slot": [np.nan, np.nan],
            "source_time": [None, None],
            "template_interval": ["全天购电量", "全天购电费"],
            "physical_start_minute": [np.nan, np.nan],
            "physical_end_minute": [np.nan, np.nan],
            "minutes_from_midnight": [np.nan, np.nan],
        }
    )
    return pd.concat([intervals, summaries], ignore_index=True)


def build_forecast_hourly(actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    raw = pd.read_excel(ATT / "附件3.xlsx", sheet_name=0, header=0)
    raw.iloc[:, 0] = raw.iloc[:, 0].replace("", np.nan).ffill()
    issue_date = pd.to_datetime(raw.iloc[:, 0], errors="raise")
    issue_time = pd.to_timedelta(raw.iloc[:, 1].astype(str) + ":00")
    issue_at = issue_date + issue_time
    values = raw.iloc[:, 2:26].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    records = []
    for row_no, issued in enumerate(issue_at):
        for horizon in range(1, 25):
            records.append((issued, horizon, values[row_no, horizon - 1]))
    fc = pd.DataFrame(records, columns=["issue_at", "horizon_h", "pv_forecast_kw"])
    fc["target_end"] = fc["issue_at"] + pd.to_timedelta(fc["horizon_h"], unit="h")

    actual_hour = (
        actual.set_index("timestamp_end")["pv_actual_kw"]
        .resample("1h", closed="right", label="right")
        .mean()
        .rename("actual_kw")
    )
    end_join = fc.join(actual_hour, on="target_end")
    end_mae = float((end_join["pv_forecast_kw"] - end_join["actual_kw"]).abs().mean())
    start_target = fc["target_end"] - pd.Timedelta(hours=1)
    start_actual = actual_hour.reindex(start_target).to_numpy()
    start_mae = float(np.nanmean(np.abs(fc["pv_forecast_kw"].to_numpy() - start_actual)))
    alignment = "forecast_h_targets_issue_plus_h" if end_mae <= start_mae else "forecast_h_targets_issue_plus_h_minus_1"
    return fc, {"alignment": alignment, "mae_end_kw": end_mae, "mae_start_kw": start_mae}


def quality_harness(
    q1: pd.DataFrame,
    load_raw: pd.DataFrame,
    pv_raw: pd.DataFrame,
    price_raw: pd.DataFrame,
    microgrid: pd.DataFrame,
    forecast: pd.DataFrame,
) -> QualityReport:
    checks = {
        "q1_shape_144x8": q1.shape == (144, 8),
        "q1_no_missing": not q1[["price", "load_kw", "pv_forecast_kw"]].isna().any().any(),
        "q1_nonnegative": bool((q1[["price", "load_kw", "pv_forecast_kw"]] >= 0).all().all()),
        "load_shape_365x145": load_raw.shape == (365, 145),
        "pv_shape_365x145": pv_raw.shape == (365, 145),
        "price_shape_365x145": price_raw.shape == (365, 145),
        "long_shape_52560": len(microgrid) == 365 * 144,
        "date_range_complete": microgrid["date"].nunique() == 365,
        "each_day_144": bool((microgrid.groupby("date").size() == 144).all()),
        "forecast_rows_35040": len(forecast) == 365 * 4 * 24,
        "forecast_four_issues_daily": forecast["issue_at"].nunique() == 365 * 4,
        "no_missing_core": not microgrid[["load_kw", "pv_actual_kw", "price_dynamic"]].isna().any().any(),
        "nonnegative_core": bool((microgrid[["load_kw", "pv_actual_kw", "price_dynamic"]] >= 0).all().all()),
        "energy_conversion_exact": bool(np.allclose(microgrid["load_kwh"] * 6, microgrid["load_kw"])),
        "template_144_intervals": len(_template_intervals()) == 144,
        "template_first_is_0010_0020": _template_intervals()[0] == "0:10-0:20",
        "template_last_is_nextday_0000_0010": _template_intervals()[-1] == "0:00-0:10+1",
        "slot_144_ends_next_midnight": bool(
            (microgrid.loc[microgrid["slot"] == 144, "timestamp_end"].dt.normalize()
             == microgrid.loc[microgrid["slot"] == 144, "date"] + pd.Timedelta(days=1)).all()
        ),
    }
    metrics = {
        "openpyxl_version": importlib.metadata.version("openpyxl"),
        "q1_rows": len(q1),
        "q1_load_kw_min": float(q1["load_kw"].min()),
        "q1_load_kw_max": float(q1["load_kw"].max()),
        "q1_pv_kw_max": float(q1["pv_forecast_kw"].max()),
        "rows_microgrid": len(microgrid),
        "dates": microgrid["date"].nunique(),
        "missing_core": int(microgrid[["load_kw", "pv_actual_kw", "price_dynamic"]].isna().sum().sum()),
        "negative_core": int((microgrid[["load_kw", "pv_actual_kw", "price_dynamic"]] < 0).sum().sum()),
        "zero_pv_share": float((microgrid["pv_actual_kw"] == 0).mean()),
        "load_kw_min": float(microgrid["load_kw"].min()),
        "load_kw_max": float(microgrid["load_kw"].max()),
        "pv_kw_max": float(microgrid["pv_actual_kw"].max()),
        "price_min": float(microgrid["price_dynamic"].min()),
        "price_max": float(microgrid["price_dynamic"].max()),
        "pv_exceeds_load_share": float((microgrid["pv_actual_kw"] > microgrid["load_kw"]).mean()),
        "load_jump_rate_abs_p99": float(microgrid["load_jump_rate"].abs().quantile(0.99)),
        "pv_jump_kw_abs_p99": float(microgrid["pv_jump_kw"].abs().quantile(0.99)),
    }
    issues = [name for name, passed in checks.items() if not passed]
    return QualityReport(checks=checks, metrics=metrics, issues=issues)


def make_figures(microgrid: pd.DataFrame, forecast_meta: dict[str, float | str], report: QualityReport) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    style = apply_publication_style(journal="nature", lang="zh")
    colors = style["colors"]

    # raw 1: four representative days, load and PV in separate panels.
    days = pd.to_datetime(["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"])
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.2), sharex=True)
    styles = ["-", "--", "-.", ":"]
    for d, ls in zip(days, styles):
        x = microgrid[microgrid["date"] == d]
        hour = x["slot"] / 6
        axes[0].plot(hour, x["load_kw"], ls=ls, label=d.strftime("%m-%d"))
        axes[1].plot(hour, x["pv_actual_kw"], ls=ls, label=d.strftime("%m-%d"))
    axes[0].set_ylabel("负荷/kW"); axes[1].set_ylabel("光伏/kW"); axes[1].set_xlabel("时刻/h")
    axes[0].legend(ncol=4, frameon=False)
    export_figure(fig, FIGURES / "raw_preprocess_representative_days", dpi=300)

    # raw 2: monthly distributions.
    daily = microgrid.groupby("date", as_index=False).agg(load_kwh=("load_kwh", "sum"), pv_kwh=("pv_kwh", "sum"))
    daily["month"] = daily["date"].dt.month
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    monthly_load = [daily.loc[daily["month"] == m, "load_kwh"].to_numpy() for m in range(1, 13)]
    monthly_pv = [daily.loc[daily["month"] == m, "pv_kwh"].to_numpy() for m in range(1, 13)]
    bp1 = axes[0].boxplot(monthly_load, patch_artist=True, showfliers=True, flierprops={"markersize": 1.5})
    bp2 = axes[1].boxplot(monthly_pv, patch_artist=True, showfliers=True, flierprops={"markersize": 1.5})
    for patch in bp1["boxes"]: patch.set_facecolor(colors[0])
    for patch in bp2["boxes"]: patch.set_facecolor(colors[1])
    axes[0].set(xlabel="月份", ylabel="日负荷电量/kWh"); axes[1].set(xlabel="月份", ylabel="日光伏电量/kWh")
    export_figure(fig, FIGURES / "raw_preprocess_monthly_distribution", dpi=300)

    # raw 3: annual dynamic-price heatmap.
    mat = microgrid.pivot(index="date", columns="slot", values="price_dynamic")
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set(xlabel="时刻/h", ylabel="日期序号", xticks=[0, 35, 71, 107, 143], xticklabels=[0, 6, 12, 18, 24])
    fig.colorbar(im, ax=ax, label="电价/(元/kWh)")
    export_figure(fig, FIGURES / "raw_preprocess_dynamic_price_heatmap", dpi=300)

    # process 1: quality contract matrix.
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    vals = np.array([[1 if v else 0] for v in report.checks.values()])
    ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set(
        yticks=np.arange(len(report.checks)),
        yticklabels=list(report.checks),
        xticks=[0],
        xticklabels=["通过状态"],
        ylabel="数据契约检查项",
    )
    export_figure(fig, FIGURES / "process_preprocess_quality_contract", dpi=300)

    # process 2: forecast alignment comparison.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    names = ["预测第h小时对应发布后第h小时", "预测第h小时对应发布后第h-1小时"]
    vals = [forecast_meta["mae_end_kw"], forecast_meta["mae_start_kw"]]
    ax.barh(names, vals, color=[colors[0], colors[1]])
    ax.set_xlabel("与实际小时均值的MAE/kW")
    export_figure(fig, FIGURES / "process_preprocess_forecast_alignment", dpi=300)

    # process 3: unit-conversion identity.
    sample = microgrid.iloc[::180]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(sample["load_kw"], sample["load_kwh"], s=8, alpha=.35, color=colors[0])
    xx = np.array([sample["load_kw"].min(), sample["load_kw"].max()])
    ax.plot(xx, xx / 6, "--", color=colors[1])
    ax.set(xlabel="负荷功率/kW", ylabel="10分钟负荷电量/kWh")
    export_figure(fig, FIGURES / "process_preprocess_unit_conversion", dpi=300)

    # result 1: daily energy trend.
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.plot(daily["date"], daily["load_kwh"], label="负荷", color=colors[0], lw=.9)
    ax.plot(daily["date"], daily["pv_kwh"], label="光伏", color=colors[1], lw=.9, ls="--")
    ax.set(xlabel="日期", ylabel="日电量/kWh"); ax.legend(frameon=False)
    export_figure(fig, FIGURES / "result_preprocess_daily_energy", dpi=300)

    # result 2: core-variable correlation.
    corr = microgrid[["load_kw", "pv_actual_kw", "price_dynamic", "net_load_kwh"]].corr()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["负荷", "光伏", "电价", "净负荷"]
    ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    export_figure(fig, FIGURES / "result_preprocess_core_correlation", dpi=300)

    # result 3: monthly net-load distribution.
    daily_net = microgrid.groupby("date", as_index=False)["net_load_kwh"].sum()
    daily_net["month"] = daily_net["date"].dt.month
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    monthly_net = [daily_net.loc[daily_net["month"] == m, "net_load_kwh"].to_numpy() for m in range(1, 13)]
    parts = ax.violinplot(monthly_net, positions=range(1, 13), showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(colors[2]); body.set_edgecolor(colors[0]); body.set_alpha(.75)
    ax.set_xticks(range(1, 13))
    ax.set(xlabel="月份", ylabel="日净负荷电量/kWh")
    export_figure(fig, FIGURES / "result_preprocess_monthly_netload", dpi=300)
    plt.close("all")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    a1 = pd.read_excel(ATT / "附件1.xlsx", sheet_name=0, header=0)
    sheets = pd.ExcelFile(ATT / "附件2.xlsx").sheet_names
    expected_sheets = ["小区负载", "光伏发电实际功率"]
    if sheets != expected_sheets:
        raise ValueError(f"附件2工作表结构异常：期望{expected_sheets}，实际{sheets}")
    load_raw = _read_wide(ATT / "附件2.xlsx", sheet_name="小区负载")
    pv_raw = _read_wide(ATT / "附件2.xlsx", sheet_name="光伏发电实际功率")
    price_raw = _read_wide(ATT / "附件4.xlsx", sheet_name=0)

    load = _wide_to_long(load_raw, "load_kw")
    pv = _wide_to_long(pv_raw, "pv_actual_kw")
    price = _wide_to_long(price_raw, "price_dynamic")
    microgrid = load.merge(pv, on=["date", "slot", "timestamp_end"], validate="one_to_one")
    microgrid = microgrid.merge(price, on=["date", "slot", "timestamp_end"], validate="one_to_one")
    q1 = build_q1_typical_day(a1)
    fixed = q1["price"].to_numpy(float)
    if len(fixed) != 144:
        raise ValueError("附件1固定电价不是144个时段")
    microgrid["price_fixed"] = np.tile(fixed, 365)
    microgrid["load_kwh"] = microgrid["load_kw"] / 6
    microgrid["pv_kwh"] = microgrid["pv_actual_kw"] / 6
    microgrid["net_load_kwh"] = microgrid["load_kwh"] - microgrid["pv_kwh"]
    microgrid["pv_load_ratio"] = microgrid["pv_actual_kw"] / microgrid["load_kw"]
    microgrid["load_jump_rate"] = microgrid.groupby("date")["load_kw"].pct_change(fill_method=None)
    microgrid["pv_jump_kw"] = microgrid.groupby("date")["pv_actual_kw"].diff()
    microgrid["month"] = microgrid["date"].dt.month
    microgrid["weekday"] = microgrid["date"].dt.weekday
    microgrid["is_weekend"] = (microgrid["weekday"] >= 5).astype(int)
    angle = 2 * np.pi * (microgrid["slot"] - 1) / 144
    microgrid["tod_sin"] = np.sin(angle); microgrid["tod_cos"] = np.cos(angle)

    mapping = build_time_mapping(list(load_raw.columns[1:]))
    forecast, forecast_meta = build_forecast_hourly(microgrid)
    report = quality_harness(q1, load_raw, pv_raw, price_raw, microgrid, forecast)
    forecast_meta["selected_alignment"] = forecast_meta["alignment"]

    q1.to_csv(OUT / "q1_typical_day.csv", index=False, encoding="utf-8-sig")
    microgrid.to_csv(OUT / "microgrid_10min.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(OUT / "pv_forecast_hourly.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUT / "time_mapping.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "q1_storage_parameters.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "initial_energy_kwh": 6000,
                "terminal_energy_kwh": 6000,
                "operating_min_kwh": 1200,
                "operating_max_kwh": 10800,
                "rated_capacity_kwh": 12000,
                "max_charge_discharge_power_kw": 5000,
                "charge_efficiency": 0.9,
                "discharge_efficiency": 0.9,
                "interval_hours": 1 / 6,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(RESULTS / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(report), "passed": report.passed, "forecast_alignment": forecast_meta}, f, ensure_ascii=False, indent=2)
    summary = pd.DataFrame([
        {"item": "问题1典型日记录", "before": 144, "after": q1[["price", "load_kw", "pv_forecast_kw"]].dropna().shape[0]},
        {"item": "原始负荷记录", "before": 365 * 144, "after": microgrid["load_kw"].notna().sum()},
        {"item": "原始光伏记录", "before": 365 * 144, "after": microgrid["pv_actual_kw"].notna().sum()},
        {"item": "原始动态电价记录", "before": 365 * 144, "after": microgrid["price_dynamic"].notna().sum()},
        {"item": "小时光伏预报记录", "before": 365 * 4 * 24, "after": forecast["pv_forecast_kw"].notna().sum()},
    ])
    summary["retention_rate"] = summary["after"] / summary["before"]
    summary.to_csv(RESULTS / "preprocessing_summary.csv", index=False, encoding="utf-8-sig")
    make_figures(microgrid, forecast_meta, report)
    print(json.dumps({"passed": report.passed, "metrics": report.metrics, "forecast_alignment": forecast_meta}, ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
