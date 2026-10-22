# -*- coding: utf-8 -*-
"""问题三：附件3光伏预报的对齐口径与精度审计。

用途：在正式建模前，用数据判定三件事，避免方案预设错误口径。
  1. 附件3的整点预报值在10分钟粒度上究竟对应哪一个时段（对齐口径）；
  2. "整点值当小时平均再均分6段" 与 "整点锚点+插值" 两种降尺度方式的精度差异；
  3. 6:00/12:00/18:00 更新预报相对沿用0:00预报的剩余区间MAE改善（VOI的预测侧证据）。

非前视：仅做预测值—实测值的统计比较，不产出任何调度决策。
"""
from __future__ import annotations

import sys
import io

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MICRO = "data_processed/microgrid_10min.csv"
FCST = "data_processed/pv_forecast_hourly.csv"


def load():
    m = pd.read_csv(MICRO, parse_dates=["date", "timestamp_end"])
    f = pd.read_csv(FCST, parse_dates=["issue_at", "target_end"])
    return m, f


def alignment_10min(m: pd.DataFrame, f: pd.DataFrame) -> None:
    """把附件3的值与不同10分钟偏移的实测比较，定位其真实语义。"""
    m = m.sort_values("timestamp_end").reset_index(drop=True)
    idx = {t: i for i, t in enumerate(m["timestamp_end"])}
    pv = m["pv_actual_kw"].to_numpy()
    print("== 1) 附件3值 vs 各10分钟偏移的实测 MAE (offset=0 表示覆盖 target_end 的那一格) ==")
    best = None
    for off in range(-6, 7):
        vals = np.array([
            pv[idx[te + pd.Timedelta(minutes=10 * off)]] if (te + pd.Timedelta(minutes=10 * off)) in idx else np.nan
            for te in f["target_end"]
        ])
        e = f["pv_forecast_kw"].to_numpy() - vals
        msk = ~np.isnan(e)
        mae = float(np.abs(e[msk]).mean())
        if best is None or mae < best[1]:
            best = (off, mae)
        print("   offset=%+d (%+3d min)  MAE=%7.2f" % (off, 10 * off, mae))
    print("   -> 最优偏移 = %+d (MAE=%.2f)\n" % best)


def downscale_compare(m: pd.DataFrame, f: pd.DataFrame) -> None:
    """0:00预报重建全天10分钟光伏：小时均分 / 整点插值 / 朴素基线。"""
    pv = m.assign(hour=(m["slot"] - 1) // 6).pivot(index="date", columns="slot", values="pv_actual_kw")
    dates = sorted(pv.index)
    dates_set = set(dates)
    f0 = f[f["issue_at"].dt.hour == 0]
    by_day = {}
    for r in f0.itertuples():
        by_day.setdefault(r.issue_at.normalize(), []).append((r.horizon_h, r.pv_forecast_kw))

    def hourmean(d):
        arr = np.zeros(144)
        for h, kw in by_day.get(d, []):
            if 1 <= h <= 24:
                arr[(h - 1) * 6: h * 6] = kw
        return arr

    def interp(d):
        pts = sorted((h * 6, kw) for h, kw in by_day.get(d, []) if 1 <= h * 6 <= 144)
        if len(pts) < 2:
            return None
        xs, ys = zip(*pts)
        return np.interp(np.arange(1, 145), xs, ys)

    def naive(d):
        prev = d - pd.Timedelta(days=1)
        return pv.loc[prev].to_numpy() if prev in dates_set else None

    res = {"A_小时均分": [], "B_朴素(前一日)": [], "C_整点插值": []}
    day = {"A_小时均分": [], "B_朴素(前一日)": [], "C_整点插值": []}
    for d in dates:
        if d < pd.Timestamp("2025-02-01"):
            continue
        t = pv.loc[d].to_numpy()
        mask = t > 50
        for name, fn in (("A_小时均分", hourmean), ("B_朴素(前一日)", naive), ("C_整点插值", interp)):
            p = fn(d)
            if p is None:
                continue
            res[name].append(np.abs(p - t).mean())
            if mask.sum():
                day[name].append(np.abs(p[mask] - t[mask]).mean())
    print("== 2) 0:00预报重建全天10分钟光伏 MAE (kW, 2025-02-01起334天) ==")
    for name in res:
        print("   %-14s 全天=%6.1f  白昼(>50kW)=%6.1f" % (name, np.mean(res[name]), np.mean(day[name])))
    print()


def voi_forecast_side(m: pd.DataFrame, f: pd.DataFrame) -> None:
    """各发布时刻剩余区间MAE，以及相对沿用0:00预报的改善。"""
    m = m.assign(hour=(m["slot"] - 1) // 6)
    obs = m.groupby(["date", "hour"], as_index=False)["pv_actual_kw"].mean()
    obs_map = {(r.date, r.hour): r.pv_actual_kw for r in obs.itertuples()}
    f = f.copy()
    f["cover"] = f["issue_at"] + pd.to_timedelta(f["horizon_h"] - 1, unit="h")
    f["obs"] = [obs_map.get((x.normalize(), x.hour), np.nan) for x in f["cover"]]
    f = f.dropna(subset=["obs"])
    f["iss"] = f["issue_at"].dt.hour
    f["day"] = f["issue_at"].dt.normalize()
    same = f[f["cover"].dt.normalize() == f["day"]]
    print("== 3) 更新预报的预测侧价值（整点锚点口径, 仅当天剩余整点）==")
    for k in (0, 6, 12, 18):
        s = same[same["iss"] == k]
        print("   issue %2d:00   MAE=%6.1f  bias=%7.1f  n=%d" % (
            k, np.abs(s["pv_forecast_kw"] - s["obs"]).mean(), (s["pv_forecast_kw"] - s["obs"]).mean(), len(s)))
    f0 = f[f["iss"] == 0][["day", "cover", "pv_forecast_kw"]].rename(columns={"pv_forecast_kw": "kw0"})
    print("   相对沿用0:00预报的改善：")
    for k in (6, 12, 18):
        s = same[same["iss"] == k][["day", "cover", "pv_forecast_kw", "obs"]]
        mg = s.merge(f0, on=["day", "cover"], how="inner")
        cur = np.abs(mg["pv_forecast_kw"] - mg["obs"]).mean()
        old = np.abs(mg["kw0"] - mg["obs"]).mean()
        print("   issue %2d:00  当次=%.1f  沿用0:00=%.1f  改善=%.1f kW" % (k, cur, old, old - cur))


if __name__ == "__main__":
    micro, fcst = load()
    alignment_10min(micro, fcst)
    downscale_compare(micro, fcst)
    voi_forecast_side(micro, fcst)
