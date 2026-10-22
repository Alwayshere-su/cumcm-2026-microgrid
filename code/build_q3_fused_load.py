# -*- coding: utf-8 -*-
"""生成问题三「分解 ⊕ Ridge 线性融合」负荷预测面板（非前视）。

融合：L̂ = w·分解 + (1−w)·Ridge，w 由此前≤30日残差含交叉项最小二乘滚动确定。
输出 data_processed/q3_forecasts_fused.csv（列：date, slot, lgb_load_kwh=融合负荷, lgb_pv_kwh=Ridge光伏）。
不改动 q2_forecasts_reg_ext.csv（问题二/四不受影响）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
DATA = ROOT / "data_processed/microgrid_10min.csv"
RIDGE = ROOT / "data_processed/q2_forecasts_reg_ext.csv"
OUT = ROOT / "data_processed/q3_forecasts_fused.csv"

m = pd.read_csv(DATA, parse_dates=["date"])
piv = m.pivot(index="date", columns="slot", values="load_kwh")
dates = piv.index
L = piv.values
B = L.sum(axis=1)
dow = np.array([d.weekday() for d in dates])
k = np.where((dow == 4) | (dow == 5), 0, 1)

reg = pd.read_csv(RIDGE, parse_dates=["date"])
RL = reg.pivot(index="date", columns="slot", values="lgb_load_kwh").reindex(dates).to_numpy(float, copy=True)
cm = np.nanmean(RL, axis=0); msk = np.isnan(RL)
RL[msk] = np.take(cm, np.nonzero(msk)[1])


def dec(i):
    ki = k[i]
    same = [j for j in range(i) if k[j] == ki]
    Id = same[-3:] if len(same) >= 3 else same
    if not Id:
        return RL[i]
    shape = np.mean(L[Id], axis=0) / np.mean(B[Id])
    Jd = [j for j in range(max(1, i - 35), i) if k[j] != k[j - 1]]
    beta = float(np.median([np.log(B[j] / B[j - 1]) / (k[j] - k[j - 1]) for j in Jd])) if Jd else 0.0
    return (B[i - 1] * np.exp(beta * (k[i] - k[i - 1]))) * shape


DEC = np.array([dec(i) for i in range(len(dates))])
FUSED = RL.copy()
for i in range(len(dates)):
    if i < 14:
        continue
    a = max(0, i - 30)
    eD = (DEC[a:i] - L[a:i]).ravel(); eR = (RL[a:i] - L[a:i]).ravel()
    den = float(np.sum((eD - eR) ** 2))
    w = 0.75 if den <= 0 else float(np.clip(np.sum(eR * (eR - eD)) / den, 0, 1))
    FUSED[i] = w * DEC[i] + (1 - w) * RL[i]

slot_cols = list(reg.pivot(index="date", columns="slot", values="lgb_load_kwh").columns)
fdf = pd.DataFrame(FUSED, index=dates, columns=slot_cols).stack().rename("lgb_load_kwh").reset_index()
fdf.columns = ["date", "slot", "lgb_load_kwh"]

pv = reg.pivot(index="date", columns="slot", values="lgb_pv_kwh").reindex(dates)
PV = pv.to_numpy(float, copy=True)
pm = np.nanmean(PV, axis=0); pmask = np.isnan(PV)
PV[pmask] = np.take(pm, np.nonzero(pmask)[1])
pvdf = pd.DataFrame(PV, index=dates, columns=slot_cols).stack().rename("lgb_pv_kwh").reset_index()
pvdf.columns = ["date", "slot", "lgb_pv_kwh"]

out = fdf.merge(pvdf, on=["date", "slot"]).sort_values(["date", "slot"])
out.to_csv(OUT, index=False, encoding="utf-8-sig")

lo = dates.get_loc(pd.Timestamp("2025-02-01")); hi = dates.get_loc(pd.Timestamp("2025-12-31"))
mae = lambda a: float(np.mean(np.abs(a[lo:hi + 1] - L[lo:hi + 1]))) * 6
print(f"输出 {OUT.name}  行数 {len(out)}")
print(f"  Ridge  MAE = {mae(RL):.2f} kW")
print(f"  融合   MAE = {mae(FUSED):.2f} kW")
