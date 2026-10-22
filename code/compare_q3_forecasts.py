# -*- coding: utf-8 -*-
"""问题三预测层候选（非ML为主）在 2025-02-01~12-31 上的真实精度：MAE / 偏差 / 重合率。

负荷候选：朴素、统计均值(35同周日)、岭回归P50、LightGBM P50(参考)
光伏候选：朴素、岭回归P50、附件3(0:00整点锚点插值)、LightGBM P50(参考)
偏差 bias = mean(预测-实际)，>0 表示高估（负荷高估=安全，光伏高估=危险）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_q2_twolayer import stat_forecast  # noqa: E402
from solve_q3 import _official_current_day  # noqa: E402

from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)


def kw(x):
    return x * 6.0


def mae(p, a):
    return float(np.mean(np.abs(p - a))) * 6


def bias(p, a):
    return float(np.mean(p - a)) * 6


def overlap(p, a, tol):
    return float(np.mean(np.abs(p - a) / np.maximum(np.abs(a), 1e-9) <= tol))


def main() -> None:
    data = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv", parse_dates=["date"])
    reg = pd.read_csv(ROOT / "data_processed/q2_forecasts_reg.csv", parse_dates=["date"])
    official = pd.read_csv(ROOT / "data_processed/pv_forecast_hourly.csv",
                           parse_dates=["issue_at", "target_end"])
    dates = pd.date_range("2025-02-01", "2025-12-31")
    reg = reg[reg.date.isin(dates)].sort_values(["date", "slot"]).reset_index(drop=True)

    by_day = {d: x.sort_values("slot") for d, x in data.groupby("date")}

    # 非前视统计均值预测（逐日）
    stat_l, stat_p = [], []
    for d in dates:
        lh, ph = stat_forecast(by_day, d)
        stat_l.append(lh)
        stat_p.append(ph)
    reg["stat_load_kwh"] = np.concatenate(stat_l)
    reg["stat_pv_kwh"] = np.concatenate(stat_p)

    # 附件3 0:00 整点锚点插值（逐日）
    anch_p = []
    for d in dates:
        anch_p.append(_official_current_day(official, data, d, 0))
    reg["anchor_pv_kwh"] = np.concatenate(anch_p)

    al = reg.actual_load_kwh.to_numpy(float)
    ap = reg.actual_pv_kwh.to_numpy(float)
    daytime = ap * 6.0 > 50.0

    load_cand = [("朴素", reg.naive_load_kwh), ("统计均值", reg.stat_load_kwh),
                 ("岭回归P50", reg.reg_load_kwh), ("LightGBM P50", reg.lgb_load_kwh)]
    pv_cand = [("朴素", reg.naive_pv_kwh), ("岭回归P50", reg.reg_pv_kwh),
               ("附件3锚点", reg.anchor_pv_kwh), ("LightGBM P50", reg.lgb_pv_kwh)]

    print("===== 负荷预测（kW，2025-02-01~12-31） =====")
    print(f"{'模型':<14}{'MAE':>9}{'偏差bias':>10}{'重合@10%':>10}{'重合@20%':>10}")
    for name, s in load_cand:
        p = s.to_numpy(float)
        print(f"{name:<14}{mae(p, al):>9.2f}{bias(p, al):>10.2f}"
              f"{overlap(p, al, .1):>10.4f}{overlap(p, al, .2):>10.4f}")

    print("\n===== 光伏预测（kW，2025-02-01~12-31） =====")
    print(f"{'模型':<14}{'MAE':>9}{'偏差bias':>10}{'白昼重合@20%':>12}{'白昼重合@30%':>12}")
    for name, s in pv_cand:
        p = s.to_numpy(float)
        print(f"{name:<14}{mae(p, ap):>9.2f}{bias(p, ap):>10.2f}"
              f"{overlap(p[daytime], ap[daytime], .2):>12.4f}{overlap(p[daytime], ap[daytime], .3):>12.4f}")

    print("\n===== 净负荷(负荷-光伏) 预测 =====")
    print(f"{'模型':<14}{'MAE':>9}{'偏差bias':>10}")
    nets = [("朴素", reg.naive_load_kwh - reg.naive_pv_kwh),
            ("统计均值+附件3", reg.stat_load_kwh - reg.anchor_pv_kwh),
            ("岭回归(负荷+光伏)", reg.reg_load_kwh - reg.reg_pv_kwh),
            ("LightGBM(负荷+光伏)", reg.lgb_load_kwh - reg.lgb_pv_kwh)]
    an = al - ap
    for name, s in nets:
        p = s.to_numpy(float)
        print(f"{name:<14}{mae(p, an):>9.2f}{bias(p, an):>10.2f}")


if __name__ == "__main__":
    main()
