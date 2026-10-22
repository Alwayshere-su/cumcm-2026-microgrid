# -*- coding: utf-8 -*-
"""论文数字的单一事实源（single source of truth）。

把散落在 results/*.json 与 csv 中的关键数字集中抽取，并计算少数派生量
（如价格形状相关系数），统一写入 results/paper_metrics.json。

论文脚本 build_paper_draft.py 与校验脚本 verify_paper_numbers.py 都从这里取值，
避免正文手写数字与结果文件漂移。

用法：
    python scripts/paper_metrics.py            # 生成 results/paper_metrics.json
    python scripts/paper_metrics.py --show     # 同时打印全部指标
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RES = ROOT / "results"
DATA = ROOT / "data_processed"


def _load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise FileNotFoundError(f"缺少结果文件：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def price_shape_metrics() -> dict:
    """价格形状相关系数：给出多种定义，避免正文表述与算法不一致。"""
    d = pd.read_csv(DATA / "microgrid_10min.csv", parse_dates=["date"])
    P = d.pivot(index="date", columns="slot", values="price_dynamic")
    F = d.pivot(index="date", columns="slot", values="price_fixed")
    test = P.loc[P.index >= pd.Timestamp("2025-02-01")]
    tv = test.to_numpy()
    fshape = (F.iloc[0] / F.iloc[0].mean()).to_numpy()
    fraw = F.iloc[0].to_numpy()
    norm = tv / np.maximum(tv.mean(axis=1, keepdims=True), 1e-8)
    med = np.median(norm, axis=0)
    dev = tv - tv.mean(axis=1, keepdims=True)
    fd = fraw - fraw.mean()
    return {
        # 正文采用：扣除各自日均价后的 Pearson 相关（衡量相对日内骨架的偏离）
        "corr_price_shape_excess_pearson": float(
            np.corrcoef(dev.ravel(), np.tile(fd, len(test)))[0, 1]),
        "corr_price_shape_median_curve": float(np.corrcoef(med, fshape)[0, 1]),
        "corr_price_shape_daily_norm_mean": float(np.mean(
            [np.corrcoef(norm[i], fshape)[0, 1] for i in range(len(norm))])),
        "corr_price_shape_pooled": float(
            np.corrcoef(tv.ravel(), np.tile(fshape, len(test)))[0, 1]),
        "corr_price_excess_autocorr_lag1": float(
            pd.Series(dev.ravel()).autocorr(1)),
        "corr_price_excess_autocorr_lag6": float(
            pd.Series(dev.ravel()).autocorr(6)),
    }


def price_stats() -> dict:
    """附件4 电价结构统计。"""
    d = pd.read_csv(DATA / "microgrid_10min.csv", parse_dates=["date"])
    P = d.pivot(index="date", columns="slot", values="price_dynamic")
    F = d.pivot(index="date", columns="slot", values="price_fixed")
    tm = P.mean(axis=0).to_numpy()
    dev = P.to_numpy() - tm[None, :]
    peak = tm[107:124].mean()
    valley = tm[0:33].mean()
    f = F.iloc[0].to_numpy()
    fd = f - f.mean()
    corr_excess = float(np.corrcoef(dev.ravel(), np.tile(fd, len(P)))[0, 1])
    return {
        "price_dynamic_min": float(P.to_numpy().min()),
        "price_dynamic_max": float(P.to_numpy().max()),
        "price_dynamic_mean": float(P.to_numpy().mean()),
        "price_peak_valley_ratio_dynamic": float(peak / valley),
        "price_peak_valley_ratio_fixed": float(
            f[107:124].mean() / f[0:33].mean()),
        "price_within_day_std_mean": float(P.std(axis=1).mean()),
        "price_daily_level_std": float(P.mean(axis=1).std()),
        "price_excess_std": float(dev.std()),
        "price_daily_range_mean": float(
            (P.max(axis=1) - P.min(axis=1)).mean()),
        "price_spike_slot_ratio": float((dev > 0.3).mean()),
        "price_spike_days": int((dev > 0.3).any(axis=1).sum()),
        "price_corr_excess_vs_att1": corr_excess,
    }


def preprocess_stats() -> dict:
    d = _load("data_quality_report.json")
    keys = ("missing_core", "negative_core", "pv_zero_ratio",
            "pv_gt_load_ratio", "load_jump_p99", "pv_jump_p99", "records")
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def vif_metrics() -> dict:
    """从证据包读取 VIF、条件数与系数稳定性。"""
    base = ROOT / "岭回归相关性证据" / "data"
    vl = pd.read_csv(base / "q2_reg_vif_load.csv")
    vp = pd.read_csv(base / "q2_reg_vif_pv.csv")
    cn = pd.read_csv(base / "q2_reg_condition_number.csv")
    sm = json.loads((base / "q2_reg_collinearity_summary.json")
                    .read_text(encoding="utf-8"))
    cs = sm.get("coef_stability", {})
    out = {
        "n_features": int(len(vl)),
        "load_vif_max": float(vl.VIF.max()),
        "pv_vif_max": float(vp.VIF.max()),
        "load_vif_mean": float(vl.VIF.mean()),
        "pv_vif_mean": float(vp.VIF.mean()),
        "n_vif_gt_10": int((vl.VIF > 10).sum()),
        "load_cond_X": float(cn.loc[cn.target == "load", "condition_number_X"].iloc[0]),
        "pv_cond_X": float(cn.loc[cn.target == "pv", "condition_number_X"].iloc[0]),
    }
    # 正文 VIF 简表：按光伏 VIF 降序取前 8（含两列 VIF 与中文含义）
    top = vl[["feature", "feature_cn", "VIF"]].merge(
        vp[["feature", "VIF"]], on="feature", suffixes=("_load", "_pv"))
    top = top.sort_values("VIF_pv", ascending=False).head(8)
    out["table_top8"] = [
        {"feature": r.feature, "cn": r.feature_cn,
         "vif_load": float(r.VIF_load), "vif_pv": float(r.VIF_pv)}
        for r in top.itertuples()
    ]
    for t in ("load", "pv"):
        if t in cs:
            out[f"ols_coef_std_mean_{t}"] = cs[t]["ols_coef_std_mean"]
            out[f"ridge_coef_std_mean_{t}"] = cs[t]["ridge_coef_std_mean"]
    return out


def lambda_metrics(cv: dict) -> dict:
    """λ 前向交叉验证：最优 λ 与主模型 λ=1 的 MAE 差距。"""
    out = {"protocol": cv.get("protocol")}
    for key, lab in (("load_kwh", "load"), ("pv_kwh", "pv")):
        d = cv.get(key)
        if not d:
            continue
        curve = d["curve"]

        def mae_at(a):
            for r in curve:
                if r["alpha"] == a:
                    return r["mae_mean"]
            return None

        m1 = mae_at(1.0)
        out[lab] = {
            "best_alpha": d["best_alpha"],
            "best_mae": d["best_mae"],
            "one_se_alpha": d.get("one_se_alpha"),
            "mae_at_1": m1,
            "gap_pct_at_1": ((m1 - d["best_mae"]) / d["best_mae"] * 100)
            if m1 is not None else None,
            "curve": curve,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    q1 = _load("q1_summary.json")
    q2 = _load("q2_summary.json")
    q2acc = _load("q2_accuracy_comparison.json")
    q3 = _load("q3_final_summary.json")
    q3ab = _load("q3_ablation_2x3.json")
    q2r = _load("q2_ridge_analysis.json")
    q2nl = _load("q2_ridge_regstats.json")
    q2fa = _load("q2_ridge_feature_ablation.json")
    q2cv = _load("q2_ridge_cv.json")
    q4 = _load("harness_q4_summary.json")
    q4off = _load("q4_offline_bound.json")
    q4fd = _load("q4_four_days.json") if (RES / "q4_four_days.json").exists() else {}

    m: dict = {
        "_sources": {
            "q1": "results/q1_summary.json",
            "q2": "results/q2_summary.json",
            "q2_accuracy": "results/q2_accuracy_comparison.json",
            "q2_ridge": "results/q2_ridge_analysis.json",
            "q3": "results/q3_final_summary.json",
            "q3_ablation": "results/q3_ablation_2x3.json",
            "q4": "results/harness_q4_summary.json",
            "q4_offline": "results/q4_offline_bound.json",
        },
        "q1": {
            "cost_yuan": q1.get("primary_optimal_cost_yuan"),
            "throughput_kwh": q1.get("battery_throughput_kwh"),
            "no_storage_cost_yuan": q1.get("no_storage_cost_yuan"),
            "no_storage_purchase_kwh": q1.get("no_storage_purchase_kwh"),
            "saving_yuan": q1.get("cost_saving_yuan"),
            "saving_pct": (q1.get("cost_saving_ratio") or 0) * 100,
            "purchase_kwh": q1.get("total_purchase_kwh"),
            "total_load_kwh": q1.get("total_load_kwh"),
            "total_pv_kwh": q1.get("total_pv_kwh"),
            "max_charge_kwh": q1.get("max_charge_kwh"),
            "max_discharge_kwh": q1.get("max_discharge_kwh"),
        },
        "q2": {
            "days": q2.get("days"),
            "scenarios": q2.get("scenarios"),
            "total_cost_yuan": q2.get("total_cost_yuan"),
            "plan_cost_yuan": q2.get("plan_cost_yuan"),
            "emergency_cost_yuan": q2.get("emergency_cost_yuan"),
            "emergency_kwh": q2.get("total_emergency_kwh"),
            "plan_purchase_kwh": q2.get("total_plan_purchase_kwh"),
        },
        "q2_accuracy": q2acc,
        "q2_regstats": {
            "protocol": q2nl.get("protocol"),
            "load": q2nl.get("regstats", {}).get("load_kwh"),
            "pv": q2nl.get("regstats", {}).get("pv_kwh"),
        },
        "q2_nonlinear": q2nl.get("nonlinear"),
        "q2_feature_ablation": q2fa.get("ablation_drop_group"),
        "q2_ridge_cv": lambda_metrics(q2cv),        "q2_feature_subsets": q2fa.get("subset_performance"),
        "q2_vif": vif_metrics(),
        "q2_blend": _load("q2_forecast_accuracy.json"),
        "q2_ridge": {
            "alpha_main": q2r.get("alpha_main"),
            "n_features": q2r.get("n_features"),
            "retrain_days": q2r.get("retrain_days"),
            "main_metrics": q2r.get("main_metrics"),
            "ols_metrics": q2r.get("ols_metrics"),
            "intercept_load": q2r.get("coefficients", {}).get("load", {}).get(
                "intercept_raw"),
            "intercept_pv": q2r.get("coefficients", {}).get("pv", {}).get(
                "intercept_raw"),
        },
        "q3": {
            "days": q3.get("days"),
            "total_yuan": q3.get("total"),
            "within_cost_yuan": q3.get("within_cost"),
            "under_cost_yuan": q3.get("under_cost"),
            "over_cost_yuan": q3.get("over_cost"),
            "emergency_cost_yuan": q3.get("emergency_cost"),
            "emergency_kwh": q3.get("emergency_kwh"),
            "surplus_kwh": q3.get("surplus_kwh"),
            "charge_kwh": q3.get("charge_kwh"),
            "discharge_kwh": q3.get("discharge_kwh"),
            "under_kwh": q3.get("under_kwh"),
            "over_kwh": q3.get("over_kwh"),
            "cvar95": q3.get("cvar95"),
            "max_day": q3.get("max_day"),
            "terminal_soc_kwh": q3.get("terminal_soc_kwh"),
            "A_total_yuan": q3.get("A_total_yuan"),
            "B_total_yuan": q3.get("B_total_yuan"),
            "C_total_yuan": q3.get("C_total_yuan"),
            "forecast_accuracy": q3.get("forecast_accuracy"),
            "max_balance_residual_kwh": q3.get("max_balance_residual_kwh"),
            "max_soc_residual_kwh": q3.get("max_soc_residual_kwh"),
        },
        "q3_ablation": q3ab,
        "q4": {
            "days": q4.get("days"),
            "scenarios": q4.get("scenarios"),
            "rule": q4.get("rule"),
            "grid_charge": q4.get("grid_charge"),
            "shape_window": q4.get("shape_window"),
            "level_window": q4.get("level_window"),
            "execution": q4.get("execution"),
            "commitment": q4.get("commitment"),
            "total_yuan": q4.get("total_yuan"),
            "total_yuan_42": q4.get("total_yuan_42"),
            "wan_yuan": q4.get("wan_yuan"),
            "effects_yuan": q4.get("effects_yuan"),
            "effects_42_yuan": q4.get("effects_42_yuan"),
            "emergency_kwh": q4.get("emergency_kwh"),
            "under_kwh": q4.get("under_kwh"),
            "over_kwh": q4.get("over_kwh"),
            "unused_kwh": q4.get("unused_kwh"),
            "cvar95_daily": q4.get("cvar95_daily"),
            "worst_day": q4.get("worst_day"),
            "terminal_soc": q4.get("terminal_soc"),
            "offline_bound_yuan": q4off.get("cost"),
        },
        "q4_four_days": q4fd,
    }
    m.update(price_shape_metrics())
    m.update(price_stats())
    m["preprocess"] = preprocess_stats()

    (RES / "paper_metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {RES / 'paper_metrics.json'}")

    if args.show:
        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                print(f"  {prefix:<52} {obj}")
        walk("", m)


if __name__ == "__main__":
    main()
