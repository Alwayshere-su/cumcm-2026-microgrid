# -*- coding: utf-8 -*-
"""问题二：用非前视岭回归/OLS做源荷预测。

严格复用问题二的模型约束：
  * 非前视：每个决策日 d 只用 d 之前的标签训练，当日预测不使用当日实测；
  * 滚动重训：每 7 天重训一次，日间只更新滞后/滚动特征；
  * 光伏非负并截断到仅由历史估计的 0.995 分位晴空包络；
  * 输出 144 段/天，2/1—12/31，与 LightGBM 版本一一对应，便于逐点比较。

特征 = 问题二既有滞后/滚动特征 + 日内/年内多阶谐波（回归模型的常规设计）。
"""
from __future__ import annotations

import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_q2 import feature_panel  # noqa: E402
from solve_q2 import DATA, seasonal_naive  # noqa: E402

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
OUT = ROOT / "data_processed" / "q2_forecasts_reg.csv"
RESULTS = ROOT / "results"
LGB = ROOT / "data_processed" / "q2_forecasts.csv"

HARM = [f"tod_sin{k}" for k in range(1, 5)] + [f"tod_cos{k}" for k in range(1, 5)] \
     + ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
BASE = ["slot", "dow", "month", "lag1", "lag2", "lag7", "lag14",
        "mean3", "std3", "mean7", "std7", "mean14", "std14"]
FEATURES = BASE + HARM


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


def make_model(kind: str):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "ols":
        return make_pipeline(StandardScaler(), LinearRegression())
    raise ValueError("kind must be 'ridge' or 'ols'")


def walk_forward(kind: str = "ridge") -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    test_dates = pd.date_range("2025-02-01", "2025-12-31")
    records, models = [], {}
    for i, day in enumerate(test_dates):
        if i % 7 == 0:
            for target in ("load_kwh", "pv_kwh"):
                tr = panel[(panel.target == target) & (panel.date < day)].dropna(subset=FEATURES + [target])
                models[target] = make_model(kind).fit(tr[FEATURES], tr[target])
        daily = panel[panel.date == day].sort_values(["target", "slot"])
        actual = data[data.date == day].sort_values("slot")
        naive_l, naive_p = seasonal_naive(data, day)
        row = {"date": day, "slot": np.arange(1, 145),
               "naive_load_kwh": naive_l, "naive_pv_kwh": naive_p}
        result = pd.DataFrame(row)
        for target, prefix in (("load_kwh", "load"), ("pv_kwh", "pv")):
            x = daily[daily.target == target].sort_values("slot")
            pred = np.maximum(models[target].predict(x[FEATURES]), 0.0)
            if target == "pv_kwh":
                hist = data[data.date < day]
                cap = hist.groupby("slot").pv_kwh.quantile(.995).to_numpy()
                pred = np.minimum(pred, cap)
            result[f"reg_{prefix}_kwh"] = pred
        result["actual_load_kwh"] = actual.load_kwh.to_numpy()
        result["actual_pv_kwh"] = actual.pv_kwh.to_numpy()
        records.append(result)
    out = pd.concat(records, ignore_index=True)
    metrics = {"model": kind}
    return out, metrics


def overlap_rate(pred: np.ndarray, actual: np.ndarray, tols=(0.05, 0.1, 0.2, 0.3),
                 floor: float = 1e-9) -> dict:
    denom = np.maximum(np.abs(actual), floor)
    rel = np.abs(pred - actual) / denom
    return {f"within_{int(t*100)}pct": float(np.mean(rel <= t)) for t in tols}


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "ridge"
    out, _ = walk_forward(kind)

    lgb = pd.read_csv(LGB, parse_dates=["date"])[
        ["date", "slot", "lgb_load_kwh", "lgb_pv_kwh"]]
    merged = out.merge(lgb, on=["date", "slot"], how="left")
    merged.to_csv(OUT, index=False, encoding="utf-8-sig")

    def kw(x): return x * 6.0
    def mae(p, a): return float(np.mean(np.abs(p - a)))

    al, ap = merged.actual_load_kwh.to_numpy(), merged.actual_pv_kwh.to_numpy()
    daytime = ap * 6.0 > 50.0  # 白昼（光伏>50 kW）
    print(f"===== 回归模型({kind}) vs LightGBM vs 朴素：预测精度（2025-02-01起334天, kWh/10min）=====")
    rows = []
    for name, lp, pp in [("朴素(前一日)", merged.naive_load_kwh, merged.naive_pv_kwh),
                         ("LightGBM P50", merged.lgb_load_kwh, merged.lgb_pv_kwh),
                         (f"回归({kind})", merged.reg_load_kwh, merged.reg_pv_kwh)]:
        lp = lp.to_numpy(); pp = pp.to_numpy()
        rec = {
            "model": name,
            "load_MAE_kW": mae(lp, al) * 6,
            "pv_MAE_kW": mae(pp, ap) * 6,
            "net_MAE_kW": mae(lp - pp, al - ap) * 6,
        }
        ol = overlap_rate(lp, al); op = overlap_rate(pp[daytime], ap[daytime])
        rec["load_重合率@10%"] = ol["within_10pct"]
        rec["load_重合率@20%"] = ol["within_20pct"]
        rec["pv_白昼重合率@20%"] = op["within_20pct"]
        rec["pv_白昼重合率@30%"] = op["within_30pct"]
        rows.append(rec)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.unicode.east_asian_width", True)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}" if v < 1.5 else f"{v:,.1f}"))

    # 两次预测彼此的重合（LGB 与 回归 的一致性）
    agree = np.mean((np.abs(merged.reg_load_kwh - merged.lgb_load_kwh) / np.maximum(merged.actual_load_kwh, 1e-9)) <= 0.2)
    print(f"\n两者负荷预测互相吻合@20%: {agree:.4f}")

    summary = {"model": kind, "predictions": OUT.name, "rows": int(len(merged)),
               "metrics": table.to_dict(orient="records"),
               "agreement_20pct": float(agree)}
    (RESULTS / f"q2_reg_{kind}_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUT} 和 results/q2_reg_{kind}_comparison.json")


if __name__ == "__main__":
    main()
