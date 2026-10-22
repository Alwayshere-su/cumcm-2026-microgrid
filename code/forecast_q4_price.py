"""问题四严格非前视价格预测：日水平×日内形状 + 日内残差修正。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _level_features(levels: np.ndarray, dates: pd.DatetimeIndex, i: int) -> np.ndarray:
    """日价格水平的严格滞后特征；调用前要求 i>=28。"""
    x = [levels[i - k] for k in (1, 2, 3, 7, 14, 21, 28)]
    for w in (3, 7, 14, 28):
        h = levels[i - w:i]
        x.extend((float(np.mean(h)), float(np.std(h))))
    doy = dates[i].dayofyear
    x.extend((np.sin(2 * np.pi * doy / 365.25),
              np.cos(2 * np.pi * doy / 365.25),
              np.sin(4 * np.pi * doy / 365.25),
              np.cos(4 * np.pi * doy / 365.25)))
    x.extend(float(dates[i].dayofweek == k) for k in range(7))
    return np.asarray(x, float)


def walk_forward_hybrid_prices(P: np.ndarray, dates: pd.DatetimeIndex,
                               fixed_curve: np.ndarray, start_i: int = 31,
                               retrain_days: int = 7, ridge_alpha: float = 10.0,
                               shape_window: int = 35,
                               calibration_window: int = 28) -> np.ndarray:
    """Ridge水平 + 稳健形状融合 + 正斜率保序校准的非前视价格预测。

    模型参数每 ``retrain_days`` 日重训一次；预测第 i 日时，训练标签与校准
    样本均严格来自 ``date < dates[i]``。返回矩阵中预热期为 NaN。
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    P = np.asarray(P, float)
    fixed = np.asarray(fixed_curve, float)
    n, t = P.shape
    if fixed.shape != (t,):
        raise ValueError("fixed_curve shape mismatch")
    levels = P.mean(axis=1)
    fixed_shape = fixed / max(float(fixed.mean()), 1e-8)
    pred = np.full_like(P, np.nan, dtype=float)
    raw = np.full_like(P, np.nan, dtype=float)
    model = None
    last_fit = -10**9

    for i in range(max(28, start_i), n):
        # 周期重训；标签 j 的全部特征只使用 j 以前信息，且 j<i。
        if model is None or i - last_fit >= retrain_days:
            js = np.arange(28, i)
            X = np.vstack([_level_features(levels, dates, j) for j in js])
            model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
            model.fit(X, levels[js])
            last_fit = i
        level = max(float(model.predict(_level_features(levels, dates, i)[None])[0]),
                    1e-4)

        hs = P[max(0, i - shape_window):i]
        hlev = np.maximum(hs.mean(axis=1, keepdims=True), 1e-8)
        hist_shape = np.median(hs / hlev, axis=0)
        # 仅用历史日比较两种形状；误差倒数权重，避免硬切换。
        ef = float(np.mean(np.abs(hs / hlev - fixed_shape)))
        eh = float(np.mean(np.abs(hs / hlev - hist_shape)))
        wf = (1 / max(ef, 1e-8)) / (1 / max(ef, 1e-8) + 1 / max(eh, 1e-8))
        raw[i] = level * (wf * fixed_shape + (1 - wf) * hist_shape)

        # 正斜率仿射校准保持价格排序；校准也只读取过去的预测和实现值。
        lo = max(start_i, i - calibration_window)
        ok = np.arange(lo, i)
        ok = ok[np.isfinite(raw[ok]).all(axis=1)]
        if len(ok) >= 7:
            x = raw[ok].ravel()
            y = P[ok].ravel()
            vx = float(np.var(x))
            b = max(float(np.cov(x, y, ddof=0)[0, 1] / vx), 1e-4) if vx > 0 else 1.0
            a = float(y.mean() - b * x.mean())
            pred[i] = np.maximum(a + b * raw[i], 1e-4)
        else:
            pred[i] = np.maximum(raw[i], 1e-4)
    return pred


def day_ahead_price(P: np.ndarray, i: int, shape_window: int = 35,
                    level_window: int = 14) -> np.ndarray:
    """只使用 P[:i] 预测第 i 日；P 的形状为 (days, 144)。"""
    if i <= 0:
        raise ValueError("price forecast needs at least one completed day")
    hs = P[max(0, i - shape_window):i]
    hl = P[max(0, i - level_window):i]
    levels = np.maximum(hs.mean(axis=1, keepdims=True), 1e-6)
    shape = np.median(hs / levels, axis=0)
    # 稳健水平：近期中位数；后续可在验证集门禁后替换为 AR-X Ridge。
    level = float(np.median(hl.mean(axis=1)))
    return np.maximum(level * shape, 1e-4)


def intraday_update(base: np.ndarray, observed: np.ndarray, issue_slot: int,
                    rho: float = 0.96, lookback: int = 6) -> np.ndarray:
    """用 issue_slot 前已观测价格修正未来；绝不读取 observed[issue_slot:]。"""
    base = np.asarray(base, float)
    observed = np.asarray(observed, float)
    out = base.copy()
    if issue_slot <= 0:
        return out
    lo = max(0, issue_slot - lookback)
    bias = float(np.median(observed[lo:issue_slot] - base[lo:issue_slot]))
    h = np.arange(len(base) - issue_slot)
    out[issue_slot:] = np.maximum(base[issue_slot:] + (rho ** h) * bias, 1e-4)
    return out


def walk_forward_prices(P: np.ndarray, start_i: int, end_i: int,
                        issues=(0, 36, 72, 108)) -> dict:
    """返回 {(day_index, issue_slot): 144段预测}；更新仅使用已实现前缀。"""
    out = {}
    for i in range(start_i, end_i + 1):
        base = day_ahead_price(P, i)
        out[(i, 0)] = base
        for s in issues[1:]:
            out[(i, s)] = intraday_update(base, P[i], s)
    return out


def price_metrics(P: np.ndarray, forecasts: dict) -> dict:
    err, rel, peak_hits = [], [], []
    for (i, s), fc in forecasts.items():
        if s != 0:
            continue
        y = P[i]
        err.extend(np.abs(y - fc))
        rel.extend(np.abs((y - y.mean()) - (fc - fc.mean())))
        true_peak = set(np.argsort(y)[-24:])
        pred_peak = set(np.argsort(fc)[-24:])
        peak_hits.append(len(true_peak & pred_peak) / 24.0)
    return {"mae_yuan_per_kwh": float(np.mean(err)),
            "relative_mae_yuan_per_kwh": float(np.mean(rel)),
            "peak_recall": float(np.mean(peak_hits))}
