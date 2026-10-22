# -*- coding: utf-8 -*-
"""问题二：岭回归的"多重共线性诊断"证据包。

目的：给出数据，证明问题二预测层所用回归特征之间存在显著多重共线性，
因而采用岭回归（Ridge, alpha=1.0）而非普通最小二乘（OLS）是必要的、
有依据的。全部指标均基于问题二实际使用的同一套特征：

    FEATURES = BASE(滞后/滚动/日历) + HARM(日内/年内多阶谐波)

输出（results/ 与 figures/）：
  * q2_reg_feature_corr_<target>.csv        特征 Pearson 相关矩阵
  * q2_reg_high_corr_pairs.csv              |r|>=0.7 的高相关特征对（降序）
  * q2_reg_vif_<target>.csv                 方差膨胀因子 VIF（按目标分别计算）
  * q2_reg_collinearity_summary.json        条件数、VIF、稳定性等摘要
  * q2_reg_ridge_trace_<target>.csv         岭迹（系数随 alpha 变化）
  * q2_reg_ols_vs_ridge_coef_<target>.csv   OLS 与 Ridge 系数对比（标准化尺度）
  * q2_reg_coef_stability.csv               系数跨训练窗口的稳定性（OLS vs Ridge）
  * figures/q2_reg_feature_correlation.png  特征相关热力图
  * figures/q2_reg_vif.png                  VIF 柱状图
  * figures/q2_reg_ridge_trace.png          岭迹图
  * figures/q2_reg_coef_stability.png       系数稳定性对比图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_q2 import feature_panel          # noqa: E402
from forecast_q2_reg import FEATURES, add_harmonics  # noqa: E402
from solve_q2 import DATA                       # noqa: E402

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

TARGETS = {"load": "load_kwh", "pv": "pv_kwh"}
SHOW = {  # 便于中文报告阅读的特征名
    "slot": "时段slot", "dow": "星期dow", "month": "月份month",
    "lag1": "滞后1日", "lag2": "滞后2日", "lag7": "滞后7日", "lag14": "滞后14日",
    "mean3": "滚动均值3", "mean7": "滚动均值7", "mean14": "滚动均值14",
    "std3": "滚动标准差3", "std7": "滚动标准差7", "std14": "滚动标准差14",
    "tod_sin1": "日内sin1", "tod_cos1": "日内cos1", "tod_sin2": "日内sin2", "tod_cos2": "日内cos2",
    "tod_sin3": "日内sin3", "tod_cos3": "日内cos3", "tod_sin4": "日内sin4", "tod_cos4": "日内cos4",
    "doy_sin1": "年内sin1", "doy_cos1": "年内cos1", "doy_sin2": "年内sin2", "doy_cos2": "年内cos2",
}


def build_design(target_col: str):
    """构造与问题二完全一致的 (X标准化前, y) 设计矩阵。"""
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel = add_harmonics(feature_panel(data))
    sub = panel[panel.target == target_col].dropna(subset=FEATURES + [target_col])
    X = sub[FEATURES].to_numpy(float)
    y = sub[target_col].to_numpy(float)
    return X, y, len(sub)


def safe_standardize(X: np.ndarray) -> np.ndarray:
    """与 sklearn StandardScaler 一致的标准化：标准差为 0 的列按 1 处理。"""
    mu = X.mean(0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd


def corr_matrix(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def vif_values(X: np.ndarray) -> np.ndarray:
    """VIF_j = 1/(1-R_j^2)，R_j^2 为特征 j 对其余特征回归的拟合优度。已对 X 标准化。"""
    Xc = safe_standardize(X)
    n, p = Xc.shape
    vifs = np.empty(p)
    for j in range(p):
        others = np.delete(Xc, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(A, Xc[:, j], rcond=None)
        resid = Xc[:, j] - A @ coef
        sse = float(resid @ resid)
        sst = float(((Xc[:, j] - Xc[:, j].mean()) ** 2).sum())
        r2 = 1.0 - sse / sst
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def coef_stability(data: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """沿问题二'每7天重训'的窗口重复估计，比较 OLS 与 Ridge 系数的跨窗口波动。

    共线性会使 OLS 系数在不同训练窗口间剧烈跳动；岭回归的收缩能把这种
    跳动压下来——这是'为什么选岭回归而非 OLS'最直接的经验证据。
    """
    train_dates = pd.date_range("2025-02-01", "2025-12-31", freq="7D")
    rows = []
    for tag, col in TARGETS.items():
        sub = panel[panel.target == col]
        for d in train_dates:
            s = sub[sub.date < d].dropna(subset=FEATURES + [col])
            if len(s) < 500:
                continue
            X = s[FEATURES].to_numpy(float)
            y = s[col].to_numpy(float)
            Xc = safe_standardize(X)
            ols = LinearRegression().fit(Xc, y)
            rdg = Ridge(alpha=1.0, fit_intercept=True).fit(Xc, y)
            rows.append({
                "target": tag, "date": d.date(),
                **{f"ols_{f}": c for f, c in zip(FEATURES, ols.coef_)},
                **{f"rdg_{f}": c for f, c in zip(FEATURES, rdg.coef_)},
            })
    df = pd.DataFrame(rows)
    out = []
    for tag in TARGETS:
        sub = df[df.target == tag]
        ols_std = sub[[f"ols_{f}" for f in FEATURES]].std(axis=0)
        rdg_std = sub[[f"rdg_{f}" for f in FEATURES]].std(axis=0)
        for f in FEATURES:
            out.append({"target": tag, "feature": f, "feature_cn": SHOW.get(f, f),
                        "n_windows": int(len(sub)),
                        "ols_coef_std": float(ols_std[f"ols_{f}"]),
                        "ridge_coef_std": float(rdg_std[f"rdg_{f}"]),
                        "ratio_ols_over_ridge": float(ols_std[f"ols_{f}"] / rdg_std[f"rdg_{f}"]) if rdg_std[f"rdg_{f}"] > 0 else np.nan})
    return pd.DataFrame(out)


def ridge_trace(Xc: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    coefs = []
    for a in alphas:
        m = Ridge(alpha=float(a), fit_intercept=True).fit(Xc, y)
        coefs.append(m.coef_)
    return np.vstack(coefs)


def main() -> None:
    alphas = np.logspace(-3, 4, 60)
    summary = {"features": FEATURES, "n_features": len(FEATURES), "targets": {}}
    all_pairs = []
    cond_rows = []
    data = pd.read_csv(DATA, parse_dates=["date"])
    panel_full = add_harmonics(feature_panel(data))

    for tag, col in TARGETS.items():
        X, y, n = build_design(col)
        Xc = safe_standardize(X)          # 标准化（与问题二 StandardScaler 一致）

        # ---- 1) 相关矩阵 ----
        C = corr_matrix(X)
        corr_df = pd.DataFrame(C, index=FEATURES, columns=FEATURES)
        corr_df.to_csv(RESULTS / f"q2_reg_feature_corr_{tag}.csv", encoding="utf-8-sig")

        # ---- 2) 高相关特征对 ----
        pairs = []
        for i in range(len(FEATURES)):
            for j in range(i + 1, len(FEATURES)):
                r = C[i, j]
                if abs(r) >= 0.7:
                    pairs.append({"target": tag, "feat_i": FEATURES[i], "feat_j": FEATURES[j],
                                  "r": float(r), "abs_r": float(abs(r))})
        all_pairs += pairs

        # ---- 3) VIF ----
        vifs = vif_values(X)

        # ---- 4) 条件数（标准化设计矩阵）----
        cond = float(np.linalg.cond(Xc))
        eig = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
        cond_xtx = float(eig.max() / eig.min())

        # ---- 5) 岭迹 + OLS/Ridge 系数对比 ----
        trace = ridge_trace(Xc, y, alphas)
        ols = LinearRegression().fit(Xc, y)
        ridge = Ridge(alpha=1.0, fit_intercept=True).fit(Xc, y)

        summary["targets"][tag] = {
            "samples": int(n),
            "condition_number_X": cond,
            "condition_number_XtX": cond_xtx,
            "n_high_corr_pairs_abs_r_ge_0.7": int(len(pairs)),
            "vif_max": float(vifs.max()),
            "vif_min": float(vifs.min()),
            "vif_mean": float(vifs.mean()),
            "n_vif_gt_10": int((vifs > 10).sum()),
            "n_vif_gt_5": int((vifs > 5).sum()),
            "ols_coef_absmax": float(np.abs(ols.coef_).max()),
            "ridge_coef_absmax": float(np.abs(ridge.coef_).max()),
            "ols_ridge_coef_corr": float(np.corrcoef(ols.coef_, ridge.coef_)[0, 1]),
        }

        trace_df = pd.DataFrame(trace, columns=FEATURES)
        trace_df.insert(0, "alpha", alphas)
        trace_df.to_csv(RESULTS / f"q2_reg_ridge_trace_{tag}.csv", index=False, encoding="utf-8-sig")

        coef_df = pd.DataFrame({
            "feature": FEATURES,
            "feature_cn": [SHOW.get(f, f) for f in FEATURES],
            "ols_coef": ols.coef_,
            "ridge_coef_a1": ridge.coef_,
            "abs_shrink": np.abs(ols.coef_) - np.abs(ridge.coef_),
        })
        coef_df.to_csv(RESULTS / f"q2_reg_ols_vs_ridge_coef_{tag}.csv", index=False, encoding="utf-8-sig")

        # 保存 VIF（分目标）
        pd.DataFrame({"feature": FEATURES, "feature_cn": [SHOW.get(f, f) for f in FEATURES],
                      "VIF": vifs, "target": tag}).to_csv(
            RESULTS / f"q2_reg_vif_{tag}.csv", index=False, encoding="utf-8-sig")

        # 保存条件数（分目标）
        cond_rows.append({
            "target": tag, "samples": int(n), "n_features": len(FEATURES),
            "condition_number_X": cond, "condition_number_XtX": cond_xtx,
            "vif_max": float(vifs.max()), "n_vif_gt_10": int((vifs > 10).sum()),
        })

        # ---- 打印 ----
        print(f"\n================ 目标 = {col}  (样本 n={n}) ================")
        print(f"标准化设计矩阵条件数 kappa(X) = {cond:,.1f}   kappa(X'X) = {cond_xtx:,.3e}")
        print(f"VIF: 最大 {vifs.max():.1f} / 均值 {vifs.mean():.1f} / 最小 {vifs.min():.2f}"
              f"   (VIF>10 的特征数 = {(vifs > 10).sum()} / {len(FEATURES)})")

        top = sorted(zip(FEATURES, vifs), key=lambda t: -t[1])
        print("\n-- VIF 最高的 10 个特征 --")
        for f, v in top[:10]:
            print(f"   {SHOW.get(f, f):<12} VIF = {v:12,.1f}")

        print("\n-- |r| 最高的 12 个特征对 --")
        pl = sorted(pairs, key=lambda d: -d["abs_r"])[:12]
        for d in pl:
            print(f"   {SHOW.get(d['feat_i'], d['feat_i']):<12} ~ {SHOW.get(d['feat_j'], d['feat_j']):<12} r = {d['r']:+.3f}")

        print(f"\nOLS 系数最大绝对值 = {np.abs(ols.coef_).max():,.1f}   "
              f"Ridge(alpha=1) 系数最大绝对值 = {np.abs(ridge.coef_).max():,.1f}")

    # ---- 汇总所有高相关对 ----
    if all_pairs:
        pdf = pd.DataFrame(all_pairs).sort_values("abs_r", ascending=False)
        pdf["feat_i_cn"] = pdf.feat_i.map(lambda f: SHOW.get(f, f))
        pdf["feat_j_cn"] = pdf.feat_j.map(lambda f: SHOW.get(f, f))
        pdf = pdf[["target", "feat_i", "feat_i_cn", "feat_j", "feat_j_cn", "r", "abs_r"]]
        pdf.to_csv(RESULTS / "q2_reg_high_corr_pairs.csv", index=False, encoding="utf-8-sig")

    # ---- 条件数汇总 CSV ----
    pd.DataFrame(cond_rows).to_csv(
        RESULTS / "q2_reg_condition_number.csv", index=False, encoding="utf-8-sig")

    # ---- 6) 系数跨窗口稳定性（OLS vs Ridge）----
    stab = coef_stability(data, panel_full)
    stab.to_csv(RESULTS / "q2_reg_coef_stability.csv", index=False, encoding="utf-8-sig")
    summary["coef_stability"] = {}
    print("\n================ 系数跨窗口稳定性（问题二'每7天重训'共约48个窗口）================")
    for tag in TARGETS:
        s = stab[stab.target == tag]
        ols_mean = float(s.ols_coef_std.mean()); rdg_mean = float(s.ridge_coef_std.mean())
        summary["coef_stability"][tag] = {
            "ols_coef_std_mean": ols_mean, "ridge_coef_std_mean": rdg_mean,
            "ols_coef_std_max": float(s.ols_coef_std.max()),
            "ridge_coef_std_max": float(s.ridge_coef_std.max()),
            "mean_instability_reduction": float(ols_mean / rdg_mean) if rdg_mean > 0 else None,
        }
        print(f"\n-- 目标={tag}：各特征系数的跨窗口标准差（越小越稳）--")
        top = s.sort_values("ols_coef_std", ascending=False).head(8)
        for _, r_ in top.iterrows():
            print(f"   {r_.feature_cn:<12} OLS std = {r_.ols_coef_std:9.2f}   Ridge std = {r_.ridge_coef_std:9.2f}"
                  f"   倍数 = {r_.ratio_ols_over_ridge:5.2f}")
        print(f"   均值: OLS {ols_mean:,.2f} vs Ridge {rdg_mean:,.2f}  ->  平均波动降至 1/{ols_mean/rdg_mean:.2f}")

    (RESULTS / "q2_reg_collinearity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 results/q2_reg_* 相关证据文件。")

    # ---- 图 ----
    try:
        _plots(alphas, stab)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 绘图跳过：{exc}")


def _plots(alphas: np.ndarray, stab: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 相关热力图（负荷）
    X, _, _ = build_design("load_kwh")
    C = corr_matrix(X)
    labels = [SHOW.get(f, f) for f in FEATURES]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and abs(C[i, j]) >= 0.7:
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title("问题二回归特征相关矩阵（|r|≥0.7 标注）—— 负荷")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGDIR / "q2_reg_feature_correlation.png", dpi=150)
    plt.close(fig)

    # VIF 柱状图（负荷）
    vifs = vif_values(X)
    order = np.argsort(-vifs)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(vifs)), vifs[order], color="#c0392b")
    ax.set_xticks(range(len(vifs)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=8)
    ax.axhline(10, color="k", ls="--", lw=1); ax.axhline(5, color="gray", ls=":", lw=1)
    ax.text(len(vifs) - 1, 10, " VIF=10", va="bottom", fontsize=8)
    ax.set_ylabel("方差膨胀因子 VIF")
    ax.set_yscale("log")
    ax.set_title("问题二回归特征 VIF（>10 视为严重共线）—— 负荷")
    fig.tight_layout()
    fig.savefig(FIGDIR / "q2_reg_vif.png", dpi=150)
    plt.close(fig)

    # 岭迹图（负荷）
    Xl, yl, _ = build_design("load_kwh")
    Xc = safe_standardize(Xl)
    trace = ridge_trace(Xc, yl, alphas)
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, f in enumerate(FEATURES):
        ax.plot(alphas, trace[:, k], lw=1)
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.text(1.05, ax.get_ylim()[1], " 问题二取 k=1", fontsize=9, va="top")
    ax.set_xlabel("岭参数 k 值"); ax.set_ylabel("岭估计值")
    ax.set_title("岭迹图：岭估计值随岭参数 k 的变化（负荷）")
    fig.tight_layout()
    fig.savefig(FIGDIR / "q2_reg_ridge_trace.png", dpi=150)
    plt.close(fig)

    # 系数稳定性对比（负荷）
    s = stab[stab.target == "load"].sort_values("ols_coef_std", ascending=False)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 0.2, s.ols_coef_std, width=0.4, label="OLS", color="#c0392b")
    ax.bar(x + 0.2, s.ridge_coef_std, width=0.4, label="Ridge (alpha=1)", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(s.feature_cn, rotation=90, fontsize=8)
    ax.set_ylabel("系数跨训练窗口标准差")
    ax.set_title("系数估计稳定性：OLS vs Ridge（负荷，约48个重训窗口）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "q2_reg_coef_stability.png", dpi=150)
    plt.close(fig)
    print("已写出 figures/q2_reg_feature_correlation.png / q2_reg_vif.png / q2_reg_ridge_trace.png / q2_reg_coef_stability.png")


if __name__ == "__main__":
    main()
