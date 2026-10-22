# -*- coding: utf-8 -*-
"""问题二岭回归：特征选择依据的实证分析（消融实验）。

回答"为什么选这些特征"：对每个特征组做留一剔除，看全年样本外 MAE/R² 的变化；
再构造几个"看似合理但更差"的替代方案，说明为什么被排除。

协议与主模型完全一致：StandardScaler + Ridge(alpha=1.0)、每 7 天重训、严格非前视、
评价期 2025-02-01 ~ 12-31（334 天 × 144 段）。误差单位 kW。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_q2 import feature_panel  # noqa: E402
from forecast_q2_reg import add_harmonics  # noqa: E402
from solve_q2 import DATA  # noqa: E402

if __name__ == "__main__" and hasattr(sys.stdout, "buffer") \
        and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
TEST = pd.date_range("2025-02-01", "2025-12-31")

FULL = ["slot", "dow", "month",
        "lag1", "lag2", "lag7", "lag14",
        "mean3", "std3", "mean7", "std7", "mean14", "std14",
        "tod_sin1", "tod_cos1", "tod_sin2", "tod_cos2",
        "tod_sin3", "tod_cos3", "tod_sin4", "tod_cos4",
        "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]

GROUPS = {
    "滞后 lag1/2/7/14": ["lag1", "lag2", "lag7", "lag14"],
    "滚动均值 mean3/7/14": ["mean3", "mean7", "mean14"],
    "滚动标准差 std3/7/14": ["std3", "std7", "std14"],
    "日内谐波 tod_*": [f"tod_{k}{i}" for k in ("sin", "cos") for i in (1, 2, 3, 4)],
    "年内谐波 doy_*": ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"],
    "日历 slot/dow/month": ["slot", "dow", "month"],
}


def add_extra(panel: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """构造三个候选替代特征，用于说明为何未纳入主模型。"""
    p = panel.copy()
    for target in ("load_kwh", "pv_kwh"):
        wide = data.pivot(index="date", columns="slot", values=target)
        # 同时段一阶差分（趋势）与斜率
        d1 = wide.diff(1)
        p.loc[p.target == target, "diff1"] = d1.stack(future_stack=True).to_numpy()
        slope3 = (wide - wide.shift(3)) / 3.0
        p.loc[p.target == target, "slope3"] = slope3.stack(future_stack=True).to_numpy()
    # 星期×时段交互
    p["dow_x_slot"] = p["dow"] * p["slot"]
    return p


def run(panel: pd.DataFrame, data: pd.DataFrame, feats: list[str]) -> dict:
    models, recs = {}, []
    for i, day in enumerate(TEST):
        if i % 7 == 0:
            for target in ("load_kwh", "pv_kwh"):
                tr = panel[(panel.target == target) & (panel.date < day)].dropna(
                    subset=feats + [target])
                models[target] = make_pipeline(
                    StandardScaler(), Ridge(alpha=1.0)).fit(tr[feats], tr[target])
        daily = panel[panel.date == day]
        act = data[data.date == day].sort_values("slot")
        row = {}
        for target, pre in (("load_kwh", "load"), ("pv_kwh", "pv")):
            x = daily[daily.target == target].sort_values("slot")
            pr = np.maximum(models[target].predict(x[feats]), 0.0)
            if target == "pv_kwh":
                cap = data[data.date < day].groupby("slot").pv_kwh.quantile(.995).to_numpy()
                pr = np.minimum(pr, cap)
            row[f"{pre}_p"] = pr
        row["l"] = act.load_kwh.to_numpy()
        row["v"] = act.pv_kwh.to_numpy()
        recs.append(pd.DataFrame(row))
    o = pd.concat(recs, ignore_index=True)

    def mae(a, b):
        return float(np.mean(np.abs(a - b))) * 6

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) * 6

    def r2(a, b):
        a = np.asarray(a, float)
        return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))

    return {"load_MAE": mae(o.load_p, o.l), "pv_MAE": mae(o.pv_p, o.v),
            "net_MAE": mae(o.load_p - o.pv_p, o.l - o.v),
            "load_RMSE": rmse(o.load_p, o.l), "pv_RMSE": rmse(o.pv_p, o.v),
            "load_R2": r2(o.load_p, o.l), "pv_R2": r2(o.pv_p, o.v)}


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    panel_x = add_extra(panel, data)

    print("=" * 84)
    print("问题二岭回归 特征选择消融（α=1.0，334 天，非前视，kW）")
    print("=" * 84)

    base = run(panel, data, FULL)
    print(f"\n【基准】全部 {len(FULL)} 个特征")
    print(f"  负荷MAE={base['load_MAE']:.2f}  光伏MAE={base['pv_MAE']:.2f}  "
          f"净负荷MAE={base['net_MAE']:.2f}  负荷R²={base['load_R2']:.5f}")

    print("\n【A】留一剔除各组：ΔMAE = 剔除后 − 基准（正号表示该组有贡献）")
    print(f"  {'剔除组':<24}{'负荷ΔMAE':>12}{'光伏ΔMAE':>12}{'净负荷ΔMAE':>14}"
          f"{'负荷ΔR²':>12}")
    abl = []
    for name, cols in GROUPS.items():
        sub = [c for c in FULL if c not in cols]
        r = run(panel, data, sub)
        d = {"组": name, "n_drop": len(cols),
             "load_dMAE": r["load_MAE"] - base["load_MAE"],
             "pv_dMAE": r["pv_MAE"] - base["pv_MAE"],
             "net_dMAE": r["net_MAE"] - base["net_MAE"],
             "load_dR2": r["load_R2"] - base["load_R2"]}
        abl.append(d)
        print(f"  {name:<24}{d['load_dMAE']:>+12.2f}{d['pv_dMAE']:>+12.2f}"
              f"{d['net_dMAE']:>+14.2f}{d['load_dR2']:>+12.5f}")

    print("\n【B】只用子集：看最小可用特征集")
    print(f"  {'特征子集':<32}{'负荷MAE':>11}{'光伏MAE':>11}{'净负荷MAE':>13}")
    subsets = {
        "仅滞后+滚动均值(7个)": ["lag1", "lag7", "mean3", "mean7", "mean14"],
        "滞后+滚动(13个,无谐波)": [c for c in FULL if not c.startswith(("tod_", "doy_"))],
        "滚动+谐波(无滞后)": [c for c in FULL if not c.startswith("lag")],
        "仅日历+谐波(无历史)": ["slot", "dow", "month"] + [
            c for c in FULL if c.startswith(("tod_", "doy_"))],
    }
    srows = []
    for name, cols in subsets.items():
        r = run(panel, data, cols)
        srows.append({"子集": name, "n": len(cols), **r})
        print(f"  {name:<32}{r['load_MAE']:>11.2f}{r['pv_MAE']:>11.2f}"
              f"{r['net_MAE']:>13.2f}")

    print("\n【C】候选替代特征（加入后是否更好）")
    print(f"  {'方案':<32}{'负荷MAE':>11}{'光伏MAE':>11}{'净负荷MAE':>13}")
    alts = {
        "主模型(25)": FULL,
        "＋diff1 日间一阶差分": FULL + ["diff1"],
        "＋slope3 三日斜率": FULL + ["slope3"],
        "＋dow×slot 交互": FULL + ["dow_x_slot"],
        "－lag2（删除二阶滞后）": [c for c in FULL if c != "lag2"],
    }
    arows = []
    for name, cols in alts.items():
        r = run(panel_x, data, cols)
        arows.append({"方案": name, "n": len(cols), **r})
        print(f"  {name:<32}{r['load_MAE']:>11.2f}{r['pv_MAE']:>11.2f}"
              f"{r['net_MAE']:>13.2f}")

    print("\n【D】训练样本量随 `lag14` 的代价")
    print("  lag14 需要 14 天前置数据，使回归 walk-forward 只能从 2025-01-22 起算；")
    print("  若去掉 lag14，训练可用样本提前，但需实测 MAE 是否变化。")
    r_nolag14 = run(panel, data, [c for c in FULL if c != "lag14"])
    print(f"  －lag14：负荷MAE={r_nolag14['load_MAE']:.2f}  "
          f"光伏MAE={r_nolag14['pv_MAE']:.2f}  "
          f"净负荷MAE={r_nolag14['net_MAE']:.2f}")

    payload = {
        "protocol": {"alpha": 1.0, "retrain_days": 7, "test": [str(TEST[0].date()),
                    str(TEST[-1].date())], "n_features_full": len(FULL)},
        "features_full": FULL, "baseline": base,
        "ablation_drop_group": abl,
        "subset_performance": srows,
        "alternative_features": arows,
        "drop_lag14": r_nolag14,
    }
    (RESULTS / "q2_ridge_feature_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {RESULTS / 'q2_ridge_feature_ablation.json'}")


if __name__ == "__main__":
    main()
