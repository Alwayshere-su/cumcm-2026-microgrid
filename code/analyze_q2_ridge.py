# -*- coding: utf-8 -*-
"""问题二 岭回归（Ridge）完整分析结果导出。

产出（results/q2_ridge_analysis.json + .md）：
  1. 主模型设定：α=1.0、StandardScaler、每 7 天滚动重训、非前视
  2. α 网格：α ∈ {0.01,0.1,0.3,1,3,10,30,100,300}，给出全年样本外 MAE / RMSE / R²
  3. 最终模型系数 β（标准化尺度 与 原始尺度）与截距 β₀（负荷、光伏各一套）
  4. 全年样本外精度：Ridge vs OLS vs 朴素(前一日)
  5. 复现问题二正式主链所用的 α=1.0 口径

注意：scaler 与 Ridge 均拟合在"预测日之前的全部样本"上（严格非前视）；
"最终模型"取全年最后一个重训日（2025-12-25）所拟合的模型。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_q2 import feature_panel  # noqa: E402
from forecast_q2_reg import FEATURES, add_harmonics  # noqa: E402
from solve_q2 import DATA  # noqa: E402

# 仅在直接运行时包一层 UTF-8 stdout（被 import 时不要动，否则关闭调用方的流）
if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    _orig_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
ALPHA_MAIN = 1.0
ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
TEST_DATES = pd.date_range("2025-02-01", "2025-12-31")

FEAT_DESC = {
    "slot": "日内时段序号 1–144",
    "dow": "星期（0=周一）",
    "month": "月份",
    "lag1": "滞后 1 日同时段",
    "lag2": "滞后 2 日同时段",
    "lag7": "滞后 7 日同时段",
    "lag14": "滞后 14 日同时段",
    "mean3": "此前 3 日同时段均值",
    "std3": "此前 3 日同时段标准差",
    "mean7": "此前 7 日同时段均值",
    "std7": "此前 7 日同时段标准差",
    "mean14": "此前 14 日同时段均值",
    "std14": "此前 14 日同时段标准差",
    "tod_sin1": "日内 1 阶正弦",
    "tod_cos1": "日内 1 阶余弦",
    "tod_sin2": "日内 2 阶正弦",
    "tod_cos2": "日内 2 阶余弦",
    "tod_sin3": "日内 3 阶正弦",
    "tod_cos3": "日内 3 阶余弦",
    "tod_sin4": "日内 4 阶正弦",
    "tod_cos4": "日内 4 阶余弦",
    "doy_sin1": "年内 1 阶正弦",
    "doy_cos1": "年内 1 阶余弦",
    "doy_sin2": "年内 2 阶正弦",
    "doy_cos2": "年内 2 阶余弦",
}


def mae(p, a):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(a))))


def rmse(p, a):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(a)) ** 2)))


def r2(p, a):
    a = np.asarray(a, float)
    return float(1 - np.sum((np.asarray(p) - a) ** 2) / np.sum((a - a.mean()) ** 2))


def fit_on(panel, target, day, alpha):
    tr = panel[(panel.target == target) & (panel.date < day)].dropna(
        subset=FEATURES + [target])
    if alpha is None:
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(tr[FEATURES], tr[target])
    return model, len(tr)


def walk_forward(panel, data, alpha, model_step="ridge"):
    """全年滚动：每 7 天重训一次，非前视。返回逐 10min 预测与实测。

    alpha=None 时用 OLS（作正则强度对照）。
    """
    a_use = None if model_step == "ols" else alpha
    models, recs = {}, []
    for i, day in enumerate(TEST_DATES):
        if i % 7 == 0:
            for target in ("load_kwh", "pv_kwh"):
                models[target], _ = fit_on(panel, target, day, a_use)
        daily = panel[panel.date == day]
        actual = data[data.date == day].sort_values("slot")
        row = {"date": day, "slot": np.arange(1, 145)}
        for target, prefix in (("load_kwh", "load"), ("pv_kwh", "pv")):
            x = daily[daily.target == target].sort_values("slot")
            pred = np.maximum(models[target].predict(x[FEATURES]), 0.0)
            if target == "pv_kwh":
                cap = data[data.date < day].groupby("slot").pv_kwh.quantile(
                    .995).to_numpy()
                pred = np.minimum(pred, cap)
            row[f"{prefix}_pred_kwh"] = pred
        row["load_act_kwh"] = actual.load_kwh.to_numpy()
        row["pv_act_kwh"] = actual.pv_kwh.to_numpy()
        recs.append(pd.DataFrame(row))
    out = pd.concat(recs, ignore_index=True)
    # kWh/10min → kW
    m = {
        "load_MAE_kW": mae(out.load_pred_kwh, out.load_act_kwh) * 6,
        "load_RMSE_kW": rmse(out.load_pred_kwh, out.load_act_kwh) * 6,
        "load_R2": r2(out.load_pred_kwh, out.load_act_kwh),
        "pv_MAE_kW": mae(out.pv_pred_kwh, out.pv_act_kwh) * 6,
        "pv_RMSE_kW": rmse(out.pv_pred_kwh, out.pv_act_kwh) * 6,
        "pv_R2": r2(out.pv_pred_kwh, out.pv_act_kwh),
    }
    net_p = out.load_pred_kwh - out.pv_pred_kwh
    net_a = out.load_act_kwh - out.pv_act_kwh
    m["net_MAE_kW"] = mae(net_p, net_a) * 6
    m["net_RMSE_kW"] = rmse(net_p, net_a) * 6
    return m, out


def coefficients(model, target):
    """返回标准化尺度系数、原始尺度系数、截距。"""
    sc = model.named_steps["standardscaler"]
    rg = model.named_steps["ridge"]
    coef_std = rg.coef_.astype(float)
    coef_raw = coef_std / sc.scale_
    intercept_raw = float(rg.intercept_ - np.sum(coef_std * sc.mean_ / sc.scale_))
    return coef_std, coef_raw, intercept_raw


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    final_day = TEST_DATES[-1]          # 2025-12-31
    refit_day = TEST_DATES[((len(TEST_DATES) - 1) // 7) * 7]  # 最后一个重训日

    print("=" * 78)
    print("问题二 岭回归（Ridge）分析")
    print(f"训练集：2025-01-01 ~ {refit_day.date()} 之前全部样本（严格非前视）")
    print(f"评价期：{TEST_DATES[0].date()} ~ {final_day.date()}，{len(TEST_DATES)} 天 × 144 段")
    print(f"特征数：{len(FEATURES)}   主模型 α = {ALPHA_MAIN}（StandardScaler + L2）")
    print("=" * 78)

    # ---------- 1. α 网格 ----------
    print("\n【1】正则强度 α 网格（全年样本外，kW）")
    grid = []
    for a in ALPHAS:
        m, _ = walk_forward(panel, data, a)
        grid.append({"alpha": a, **m})
        print(f"  α={a:<6} 负荷MAE={m['load_MAE_kW']:8.2f}  光伏MAE={m['pv_MAE_kW']:8.2f}"
              f"  净负荷MAE={m['net_MAE_kW']:8.2f}  负荷R²={m['load_R2']:.5f}"
              f"  光伏R²={m['pv_R2']:.5f}")

    # ---------- 2. 主模型精度 + OLS 对照 ----------
    print(f"\n【2】主模型 α={ALPHA_MAIN} 与 OLS 对照（全年样本外）")
    m_main, pred_main = walk_forward(panel, data, ALPHA_MAIN)
    m_ols, _ = walk_forward(panel, data, 0.0, model_step="ols")
    print(f"  {'模型':<14}{'负荷MAE':>10}{'光伏MAE':>10}{'净负荷MAE':>12}"
          f"{'负荷R²':>10}{'光伏R²':>10}")
    for name, m in (("Ridge α=1.0", m_main), ("OLS(无正则)", m_ols)):
        print(f"  {name:<14}{m['load_MAE_kW']:>10.2f}{m['pv_MAE_kW']:>10.2f}"
              f"{m['net_MAE_kW']:>12.2f}{m['load_R2']:>10.5f}{m['pv_R2']:>10.5f}")

    # ---------- 3. 最终模型系数 ----------
    print(f"\n【3】最终模型系数（重训日 {refit_day.date()}，α={ALPHA_MAIN}）")
    coef_tables = {}
    for target, label, unit in (("load_kwh", "负荷", "kW"), ("pv_kwh", "光伏", "kW")):
        model, ntr = fit_on(panel, target, refit_day, ALPHA_MAIN)
        cs, cr, b0 = coefficients(model, target)
        df = pd.DataFrame({
            "特征": FEATURES,
            "含义": [FEAT_DESC.get(f, f) for f in FEATURES],
            "标准化β": cs,
            "原始尺度β": cr,
        })
        df["|标准化β|"] = df["标准化β"].abs()
        df = df.sort_values("|标准化β|", ascending=False).reset_index(drop=True)
        coef_tables[target] = {"intercept_raw": b0,
                               "intercept_std_space": float(
                                   model.named_steps["ridge"].intercept_),
                               "n_train": ntr,
                               "coef": df.to_dict(orient="records")}
        print(f"\n  ── {label}（{unit}）  β₀ = {b0:,.2f} {unit}   "
              f"训练样本 {ntr:,} 条 ──")
        print(f"  {'特征':<12}{'含义':<26}{'标准化β':>12}{'原始尺度β':>16}")
        for _, r in df.iterrows():
            print(f"  {r['特征']:<12}{r['含义']:<26}{r['标准化β']:>12.4f}"
                  f"{r['原始尺度β']:>16.6f}")

    # ---------- 4. 落盘 ----------
    payload = {
        "model": "Ridge (StandardScaler + L2)",
        "alpha_main": ALPHA_MAIN,
        "alpha_grid": ALPHAS,
        "retrain_days": 7,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "train_rule": "每个决策日只用 date < d 的样本；每 7 天重训",
        "test_window": [str(TEST_DATES[0].date()), str(final_day.date())],
        "refit_day_of_final_model": str(refit_day.date()),
        "alpha_sweep": grid,
        "main_metrics": m_main,
        "ols_metrics": m_ols,
        "coefficients": {
            "load": coef_tables["load_kwh"],
            "pv": coef_tables["pv_kwh"],
        },
    }
    (RESULTS / "q2_ridge_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {RESULTS / 'q2_ridge_analysis.json'}")


if __name__ == "__main__":
    main()
