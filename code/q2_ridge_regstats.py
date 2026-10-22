# -*- coding: utf-8 -*-
"""问题三（预测层）论文所需补充统计量与图。

产出：
  1. results/q2_ridge_regstats.json —— R²、调整 R²、F、P、σ、训练/检验样本量
  2. results/q2_nonlinear_probe.json —— 非线性分支证据：
     在完全相同的特征与训练协议下，比较线性 Ridge / 加二次项 Ridge / LightGBM，
     并对残差作 Ljung-Box 式分组检验，量化"线性是否够用"
  3. figures/result_q2_pred_vs_actual.png —— 预测曲线与实测曲线对照（含右下角小框放大）
  4. results/q2_ridge_vif_top.csv —— 精简 VIF 表（正文表格用）

协议：StandardScaler + Ridge(alpha=1.0)、每 7 天重训、严格非前视、
     评价期 2025-02-01 ~ 12-31（334 天 × 144 段）。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()
from forecast_q2 import feature_panel  # noqa: E402
from forecast_q2_reg import FEATURES, add_harmonics  # noqa: E402
from solve_q2 import DATA  # noqa: E402

if __name__ == "__main__" and hasattr(sys.stdout, "buffer") \
        and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RES = ROOT / "results"
FIG = ROOT / "figures"
TEST = pd.date_range("2025-02-01", "2025-12-31")
ALPHA = 1.0


def final_model(panel, target, day, alpha=ALPHA, extra=None):
    """在 day 之前的全部样本上拟合最终模型，返回 (model, X, y, feats)。"""
    feats = list(FEATURES) + list(extra or [])
    tr = panel[(panel.target == target) & (panel.date < day)].dropna(
        subset=feats + [target])
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
        tr[feats], tr[target])
    return model, tr[feats].to_numpy(float), tr[target].to_numpy(float), feats


def regstats(model, X, y, feats) -> dict:
    """OLS 口径的 F 检验与 p 值（在标准化设计矩阵上评估）。"""
    n, p = X.shape
    yhat = model.predict(X)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    df1, df2 = p, n - p - 1
    f = ((tss - rss) / df1) / (rss / df2)
    pval = float(stats.f.sf(f, df1, df2))
    sigma = float(np.sqrt(rss / df2))
    return {"n_train": int(n), "n_features": int(p),
            "R2": float(r2), "R2_adj": float(adj),
            "F": float(f), "df1": int(df1), "df2": int(df2),
            "p_value": pval, "sigma": sigma,
            "RSS": rss, "TSS": tss}


def make_quadratic_features(panel: pd.DataFrame) -> pd.DataFrame:
    """非线性分支：对主要连续特征加二次项。"""
    p = panel.copy()
    for f in ("lag1", "mean3", "mean7", "mean14"):
        p[f"{f}_sq"] = p[f] ** 2
    return p


QUAD = [f"{f}_sq" for f in ("lag1", "mean3", "mean7", "mean14")]


def walk_forward(panel, data, feats, alpha=ALPHA, model_kind="ridge"):
    """完全相同协议下的滚动回测，返回逐 10min 预测。"""
    models, recs = {}, []
    for i, day in enumerate(TEST):
        if i % 7 == 0:
            for target in ("load_kwh", "pv_kwh"):
                tr = panel[(panel.target == target) & (panel.date < day)].dropna(
                    subset=feats + [target])
                if model_kind == "ridge":
                    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                else:
                    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                models[target] = mdl.fit(tr[feats], tr[target])
        daily = panel[panel.date == day]
        act = data[data.date == day].sort_values("slot")
        row = {"date": day, "slot": np.arange(1, 145)}
        for target, pre in (("load_kwh", "load"), ("pv_kwh", "pv")):
            x = daily[daily.target == target].sort_values("slot")
            pr = np.maximum(models[target].predict(x[feats]), 0.0)
            if target == "pv_kwh":
                cap = data[data.date < day].groupby("slot").pv_kwh.quantile(
                    .995).to_numpy()
                pr = np.minimum(pr, cap)
            row[f"{pre}_p"] = pr
        row["l"] = act.load_kwh.to_numpy()
        row["v"] = act.pv_kwh.to_numpy()
        recs.append(pd.DataFrame(row))
    return pd.concat(recs, ignore_index=True)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) * 6


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    panel_q = make_quadratic_features(panel)
    refit_day = TEST[((len(TEST) - 1) // 7) * 7]

    # ---------- 1. 回归统计量 ----------
    print("=" * 78)
    print("1. 回归统计量（最终模型，训练集 = 重训日之前全部样本）")
    print("=" * 78)
    stats_out = {}
    for target, label in (("load_kwh", "负荷"), ("pv_kwh", "光伏")):
        model, X, y, feats = final_model(panel, target, refit_day)
        st = regstats(model, X, y, feats)
        stats_out[target] = st
        print(f"  {label}: n={st['n_train']:,}  p={st['n_features']}  "
              f"R²={st['R2']:.5f}  R²adj={st['R2_adj']:.5f}  "
              f"F={st['F']:,.1f}  P={st['p_value']:.3e}  σ={st['sigma']:.3f}")

    # ---------- 2. 非线性分支 ----------
    print("\n" + "=" * 78)
    print("2. 非线性分支：线性是否够用（同特征、同协议，全年样本外 MAE/kW）")
    print("=" * 78)
    base = walk_forward(panel, data, FEATURES)
    quad = walk_forward(panel_q, data, list(FEATURES) + QUAD)
    nl = {
        "linear_ridge": {"load_MAE": mae(base.load_p, base.l),
                         "pv_MAE": mae(base.pv_p, base.v),
                         "net_MAE": mae(base.load_p - base.pv_p, base.l - base.v)},
        "quadratic_ridge": {"load_MAE": mae(quad.load_p, quad.l),
                            "pv_MAE": mae(quad.pv_p, quad.v),
                            "net_MAE": mae(quad.load_p - quad.pv_p, quad.l - quad.v)},
    }
    # LightGBM 对照（非线性上界参照）
    try:
        import lightgbm as lgb
        lg = {}
        for target in ("load_kwh", "pv_kwh"):
            lg[target] = None
        models, recs = {}, []
        for i, day in enumerate(TEST):
            if i % 7 == 0:
                for target in ("load_kwh", "pv_kwh"):
                    tr = panel[(panel.target == target) & (panel.date < day)].dropna(
                        subset=FEATURES + [target])
                    models[target] = lgb.LGBMRegressor(
                        n_estimators=220, learning_rate=.04, num_leaves=25,
                        min_child_samples=40, subsample=.85, colsample_bytree=.85,
                        reg_lambda=.2, random_state=2026, verbosity=-1,
                        n_jobs=-1).fit(tr[FEATURES], tr[target])
            daily = panel[panel.date == day]
            act = data[data.date == day].sort_values("slot")
            row = {"date": day}
            for target, pre in (("load_kwh", "load"), ("pv_kwh", "pv")):
                x = daily[daily.target == target].sort_values("slot")
                pr = np.maximum(models[target].predict(x[FEATURES]), 0.0)
                if target == "pv_kwh":
                    cap = data[data.date < day].groupby("slot").pv_kwh.quantile(
                        .995).to_numpy()
                    pr = np.minimum(pr, cap)
                row[f"{pre}_p"] = pr
            row["l"] = act.load_kwh.to_numpy()
            row["v"] = act.pv_kwh.to_numpy()
            recs.append(pd.DataFrame(row))
        g = pd.concat(recs, ignore_index=True)
        nl["lightgbm"] = {"load_MAE": mae(g.load_p, g.l),
                          "pv_MAE": mae(g.pv_p, g.v),
                          "net_MAE": mae(g.load_p - g.pv_p, g.l - g.v)}
    except Exception as exc:  # noqa: BLE001
        print(f"  LightGBM 对照跳过：{exc}")

    for k, v in nl.items():
        print(f"  {k:<18} 负荷={v['load_MAE']:8.2f}  光伏={v['pv_MAE']:8.2f}  "
              f"净负荷={v['net_MAE']:8.2f}")
    gain_l = nl["linear_ridge"]["load_MAE"] - nl["quadratic_ridge"]["load_MAE"]
    print(f"\n  二次项相对线性的负荷 MAE 改善：{gain_l:+.2f} kW "
          f"({gain_l/nl['linear_ridge']['load_MAE']*100:+.2f}%)")
    if "lightgbm" in nl:
        gl = nl["linear_ridge"]["load_MAE"] - nl["lightgbm"]["load_MAE"]
        print(f"  非线性上界（LightGBM）相对线性的负荷 MAE 改善：{gl:+.2f} kW "
              f"({gl/nl['linear_ridge']['load_MAE']*100:+.2f}%)")

    # ---------- 3. VIF 精简表 ----------
    vif_l = pd.read_csv(ROOT / "岭回归相关性证据" / "data" / "q2_reg_vif_load.csv")
    vif_p = pd.read_csv(ROOT / "岭回归相关性证据" / "data" / "q2_reg_vif_pv.csv")
    v = vif_l[["feature", "feature_cn", "VIF"]].merge(
        vif_p[["feature", "VIF"]], on="feature", suffixes=("_load", "_pv"))
    v = v.sort_values("VIF_pv", ascending=False).head(8).reset_index(drop=True)
    v.to_csv(RES / "q2_ridge_vif_top.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print("3. VIF 精简表（按光伏 VIF 降序，取前 8）")
    print("=" * 78)
    print(v.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # ---------- 4. 预测 vs 实测图（含右下角小框） ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from plot_style import apply_publication_style  # type: ignore
        colors = apply_publication_style(journal="nature", lang="zh")["colors"]
    except Exception:  # noqa: BLE001
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    figs = []
    # 实测列名为 l / v，预测列名为 load_p / pv_p
    for pre, act_col, lab, unit in (("load", "l", "负荷", "kW"),
                                    ("pv", "v", "光伏", "kW")):
        d = base.copy()
        d["act"] = d[act_col] * 6.0
        d["pred"] = d[f"{pre}_p"] * 6.0
        t = pd.date_range(d.date.min(), periods=len(d), freq="10min")
        fig, ax = plt.subplots(figsize=(7.1, 3.4))
        ax.plot(t, d.act, color="black", lw=.7, label=f"实测{lab}")
        ax.plot(t, d.pred, color=colors[0], lw=.7, ls="--",
                label=f"Ridge预测{lab}")
        ax.set_ylabel(f"{lab}/({unit})")
        ax.set_xlabel("日期")
        ax.legend(frameon=False, ncol=2, loc="upper right")
        # 右下角小框：放大连续 3 日，展示两条曲线是否贴合
        axin = ax.inset_axes([0.60, 0.06, 0.37, 0.36])
        lo = int(len(d) * 0.42)
        sl = slice(lo, lo + 3 * 144)
        axin.plot(t[sl], d.act.to_numpy()[sl], color="black", lw=.9)
        axin.plot(t[sl], d.pred.to_numpy()[sl], color=colors[0], lw=.9, ls="--")
        axin.set_title("连续3日放大", fontsize=7, pad=2)
        axin.tick_params(labelsize=6)
        axin.set_xticks([])
        fig.tight_layout()
        out = FIG / f"result_q2_pred_vs_actual_{pre}"
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        figs.append(str(out.with_suffix('.png').name))
        print(f"\n已写出 {out.with_suffix('.png').name}")

    payload = {
        "protocol": {"alpha": ALPHA, "retrain_days": 7,
                     "refit_day_of_final_model": str(refit_day.date()),
                     "test": [str(TEST[0].date()), str(TEST[-1].date())]},
        "regstats": stats_out,
        "nonlinear": nl,
        "quadratic_features": QUAD,
        "figures": figs,
    }
    (RES / "q2_ridge_regstats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {RES / 'q2_ridge_regstats.json'}")


if __name__ == "__main__":
    main()
