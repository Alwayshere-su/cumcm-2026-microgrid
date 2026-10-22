"""问题二 方法3：回归(ridge)预测 + 历史情景随机规划 + DP 价值执行器（含储备 R_t）。

与方法2 的唯一差别是"预测器"：方法2 用统计均值（同星期 35 天均值 / 近 7 天光伏均值），
方法3 换成方法1 的回归预测（滞后/滚动特征 + 日内/年内多阶谐波，非前视、每 7 天重训、
光伏非负并截断到历史 0.995 分位包络）。其余（残差整日重采样情景、M=30、两阶段随机规划、
DP 价值执行器与保留水平 R_t）与方法2 完全相同。

为给残差库提供回归预测，回归 walk-forward 从 2025-01-22 开始（此前特征含 lag14 尚不可用）；
因此回测前期（约 2/1—2/20）残差库不足 30 天，按实际可用天数取情景。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_q2 import feature_panel  # noqa: E402
from solve_q2 import DATA, write_template  # noqa: E402
from solve_q2_twolayer import (RESID_WINDOW, V_CONT, residuals_from_forecasts,
                               run_backtest)  # noqa: E402
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
RIDGE_PANEL = ROOT / "data_processed" / "q2_forecasts_reg_ext.csv"

HARM = [f"tod_sin{k}" for k in range(1, 5)] + [f"tod_cos{k}" for k in range(1, 5)] \
     + ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
BASE = ["slot", "dow", "month", "lag1", "lag2", "lag7", "lag14",
        "mean3", "std3", "mean7", "std7", "mean14", "std14"]
FEATURES = BASE + HARM


def official_summary(summary: dict, forecast_model: str = "ridge") -> dict:
    """Attach the frozen modeling contract to a computed method-3 summary."""
    names = {
        "ridge": "Ridge回归预测",
        "decomp": "日电量水平×日内形状分解预测",
        "blend": "分解⊕Ridge线性加权融合预测",
        "mult": "分解水平×Ridge形状乘法融合预测",
    }
    forecast_name = names.get(forecast_model, forecast_model)
    return {
        **summary,
        "model_id": f"q2_method3_{forecast_model}_scenario_sp_dp",
        "model": f"{forecast_name} + 历史情景两阶段随机规划 + DP保留水位因果执行器",
        "forecast_model": forecast_model,
        "ridge_alpha": 1.0,
        "retrain_days": 7,
        "pv_historical_quantile_cap": 0.995,
        "residual_window_days": RESID_WINDOW,
        "continuation_value_yuan_per_kwh": V_CONT,
        "emergency_price_multiplier": 5.0,
        "backtest_start": "2025-02-01",
        "backtest_end": "2025-12-31",
    }


def save_method3_outputs(result: dict) -> None:
    """Publish method 3 as the sole official Q2 result and fill result2.xlsx."""
    RESULTS.mkdir(exist_ok=True)
    dispatch = result["dispatch"].copy()
    summary = result["summary"].copy()
    dispatch.to_csv(RESULTS / "q2_method3_dispatch.csv", index=False, encoding="utf-8-sig")
    dispatch.to_csv(RESULTS / "q2_dispatch_10min.csv", index=False, encoding="utf-8-sig")
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    (RESULTS / "q2_method3_summary.json").write_text(payload, encoding="utf-8")
    (RESULTS / "q2_summary.json").write_text(payload, encoding="utf-8")
    write_template(dispatch)


def add_harmonics(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    s = (p["slot"] - 1) / 144.0
    for k in range(1, 5):
        p[f"tod_sin{k}"] = np.sin(2 * np.pi * k * s)
        p[f"tod_cos{k}"] = np.cos(2 * np.pi * k * s)
    d = p["doy"] / 365.0
    for k in (1, 2):
        p[f"doy_sin{k}"] = np.sin(2 * np.pi * k * d)
        p[f"doy_cos{k}"] = np.cos(2 * np.pi * k * d)
    return p


def reg_forecasts(start: str, end: str, alpha: float = 1.0) -> dict:
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    test_dates = pd.date_range(start, end, freq="D")
    models, out = {}, {}
    t0 = time.perf_counter()
    for i, day in enumerate(test_dates):
        if i % 7 == 0:
            for target in ("load_kwh", "pv_kwh"):
                tr = panel[(panel.target == target) & (panel.date < day)].dropna(subset=FEATURES + [target])
                if len(tr) < 200:
                    continue
                models[target] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(tr[FEATURES], tr[target])
        daily = panel[panel.date == day]
        lh = np.maximum(models["load_kwh"].predict(
            daily[daily.target == "load_kwh"].sort_values("slot")[FEATURES]), 0.0)
        ph = np.maximum(models["pv_kwh"].predict(
            daily[daily.target == "pv_kwh"].sort_values("slot")[FEATURES]), 0.0)
        hist = data[data.date < day]
        cap = hist.groupby("slot").pv_kwh.quantile(.995).to_numpy()
        ph = np.minimum(ph, cap)
        out[day] = (lh, ph)
    print(f"  回归 walk-forward 完成（{len(out)} 天，{time.perf_counter()-t0:.1f}s）", flush=True)
    return out


def decomp_forecasts(data: pd.DataFrame) -> dict:
    """日电量水平 × 日内归一化形状分解预测（非前视）。

    类型：周五、周六为低负载日；水平与形状取自此前最近 3 个同类型日，
    光伏取此前最近 min(3, d-1) 天均值。所有量只用 date < d 的样本。
    """
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    dates = sorted(by_day)
    is_low = {d: d.weekday() in (4, 5) for d in dates}
    out = {}
    for i, day in enumerate(dates):
        same = [d for d in dates[:i] if is_low[d] == is_low[day]][-3:]
        if not same:
            continue
        Lm = np.mean([by_day[d]["load_kwh"].to_numpy(float) for d in same], axis=0)
        lev = float(np.mean([by_day[d]["load_kwh"].sum() for d in same]))
        lh = np.maximum(Lm / Lm.sum() * lev, 0.0)
        prev = dates[max(0, i - 3):i]
        ph = (np.mean([by_day[d]["pv_kwh"].to_numpy(float) for d in prev], axis=0)
              if prev else np.zeros_like(lh))
        out[day] = (lh, np.maximum(ph, 0.0))
    return out


def mult_forecasts(data: pd.DataFrame, start: str, end: str) -> dict:
    """融合预测：分解给出水平，形状取"归一化分解形状"与"归一化 Ridge 形状"的均值。

    回归在此扮演形状修正项，而非独立预测器；水平仍由近 3 个同类型日均电量给出。
    全部非前视。
    """
    dec = decomp_forecasts(data)
    reg = reg_forecasts(start, end)
    out = {}
    for day, (ld, pv_d) in dec.items():
        lr = reg.get(day)
        if lr is None:
            out[day] = (ld, pv_d)
            continue
        sh_d = ld / ld.sum() if ld.sum() > 0 else ld
        sh_r = lr[0] / lr[0].sum() if lr[0].sum() > 0 else lr[0]
        out[day] = (np.maximum((sh_d + sh_r) / 2 * ld.sum(), 0.0), pv_d)
    return out


def decomposition_trend(data: pd.DataFrame) -> dict:
    """问题三式分解：日电量水平（含工作日/周末切换倍率 β 的趋势外推）× 近 3 个同类型日归一化形状。

    类型日：周五、周六为低负荷日。光伏沿用问题二口径（此前最近 3 天实测均值）。非前视。
    """
    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}
    dates = sorted(by_day)
    L = np.array([by_day[d]["load_kwh"].to_numpy(float) for d in dates])
    B = L.sum(axis=1)
    dow = np.array([d.weekday() for d in dates])
    k = np.where((dow == 4) | (dow == 5), 0, 1)
    out = {}
    for i, day in enumerate(dates):
        ki = k[i]
        same = [j for j in range(i) if k[j] == ki]
        Id = same[-3:] if len(same) >= 3 else same
        if not Id:
            continue
        shape = np.mean(L[Id], axis=0) / np.mean(B[Id])
        Jd = [j for j in range(max(1, i - 35), i) if k[j] != k[j - 1]]
        beta = (float(np.median([np.log(B[j] / B[j - 1]) / (k[j] - k[j - 1])
                                 for j in Jd])) if Jd else 0.0)
        level = B[i - 1] * np.exp(beta * (ki - k[i - 1]))
        lh = np.maximum(level * shape, 0.0)
        prev = dates[max(0, i - 3):i]
        ph = (np.mean([by_day[d]["pv_kwh"].to_numpy(float) for d in prev], axis=0)
              if prev else np.zeros_like(lh))
        out[day] = (lh, np.maximum(ph, 0.0))
    return out


def fused_forecasts(data: pd.DataFrame) -> dict:
    """问题三式线性加权融合（正式）：负荷 = w·DEC + (1−w)·Ridge。

    DEC 取问题三式分解（β 趋势水平 × 同类型日形状）；Ridge 支取官方面板
    ``q2_forecasts_reg_ext.csv``；w 由此前 ≤30 日残差含交叉项最小二乘滚动确定。
    光伏沿用问题二口径（近 3 天实测均值）。全部非前视。
    """
    dec = decomposition_trend(data)
    piv = data.pivot(index="date", columns="slot", values="load_kwh")
    dates = list(piv.index)
    L = piv.to_numpy(float)
    ridge = pd.read_csv(RIDGE_PANEL, parse_dates=["date"])
    RL = (ridge.pivot(index="date", columns="slot", values="lgb_load_kwh")
          .reindex(dates).to_numpy(float))
    cm = np.nanmean(RL, axis=0)
    msk = np.isnan(RL)
    RL[msk] = np.take(cm, np.nonzero(msk)[1])
    DEC = np.array([dec[d][0] if d in dec else RL[i] for i, d in enumerate(dates)])

    out = {}
    for i, day in enumerate(dates):
        if day not in dec:
            continue
        pv = dec[day][1]
        if i < 14:
            out[day] = (DEC[i], pv)
            continue
        a = max(0, i - 30)
        eD = (DEC[a:i] - L[a:i]).ravel()
        eR = (RL[a:i] - L[a:i]).ravel()
        den = float(np.sum((eD - eR) ** 2))
        w = 0.75 if den <= 0 else float(np.clip(np.sum(eR * (eR - eD)) / den, 0, 1))
        out[day] = (np.maximum(w * DEC[i] + (1 - w) * RL[i], 0.0), pv)
    return out


def save_forecast_panel(fc: dict, forecast_model: str) -> Path:
    """导出问题四可复用的权威逐段预测接口。"""
    rows = []
    for day, (load, pv) in sorted(fc.items()):
        for slot, (lv, vv) in enumerate(zip(load, pv), 1):
            rows.append({"date": day, "slot": slot,
                         "load_forecast_kwh": float(lv),
                         "pv_forecast_kwh": float(vv),
                         "forecast_model": forecast_model})
    out = pd.DataFrame(rows)
    path = ROOT / "data_processed" / f"q2_forecasts_{forecast_model}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--forecast", default="blend",
                    choices=["ridge", "decomp", "blend", "mult"],
                    help="预测层：blend=分解⊕Ridge线性加权融合（正式）；"
                         "mult=分解水平×Ridge形状乘法融合（对照）；"
                         "decomp=日电量水平×日内形状分解；ridge=Ridge回归")
    ap.add_argument("--plan-uplift", type=float, default=1.0,
                    help="敏感性实验：计划量整体乘系数（默认 1.0）")
    args = ap.parse_args()

    data = pd.read_csv(DATA, parse_dates=["date"])
    if args.forecast == "decomp":
        print("方法3：生成日电量水平×日内归一化形状分解预测（非前视）...", flush=True)
        fc = decomp_forecasts(data)
    elif args.forecast == "blend":
        print("方法3：生成分解⊕Ridge 线性加权融合预测（非前视）...", flush=True)
        fc = fused_forecasts(data)
    elif args.forecast == "mult":
        print("方法3：生成分解水平×Ridge形状乘法融合预测（对照，非前视）...", flush=True)
        fc = mult_forecasts(data, "2025-01-22", "2025-12-31")
    else:
        print("方法3：生成回归预测（非前视、7 天重训、0.995 截断）...", flush=True)
        fc = reg_forecasts("2025-01-22", "2025-12-31")
    fcast, resid = residuals_from_forecasts(data, fc)

    res = run_backtest(args.days, args.scenarios, fcast=fcast, resid=resid,
                       model_name="方法3：回归预测 + 历史情景随机规划 + DP 价值执行器",
                       plan_uplift=args.plan_uplift)
    res["summary"] = official_summary(res["summary"], args.forecast)
    s = res["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.save:
        path = save_forecast_panel(fc, args.forecast)
        save_method3_outputs(res)
        print(f"已写出问题二正式结果、q2_summary.json、result2.xlsx 与 {path.name}")


if __name__ == "__main__":
    main()
