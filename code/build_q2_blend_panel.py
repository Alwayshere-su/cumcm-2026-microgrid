# -*- coding: utf-8 -*-
"""落盘问题二「融合预测」面板，供论文绘图与数字核对（非前视，不改动任何正式结果）。

融合定义（与 scripts/solve_q2_method3.py 的 fused_forecasts 完全一致）：
  负荷 = 分解⊕Ridge 线性加权融合 = w·DEC + (1−w)·Ridge
         DEC = 日电量水平（含工作日/周末切换倍率 β 的趋势外推）× 近 3 个同类型日形状
         w   由此前 ≤30 日残差含交叉项最小二乘滚动确定
  光伏 = 分解光伏（此前最近 min(3,d-1) 日均值，非负）
输出 data_processed/q2_forecasts_blend.csv（date, slot, lgb_load_kwh, lgb_pv_kwh）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()

from solve_q2_method3 import fused_forecasts  # noqa: E402
from solve_q2 import DATA  # noqa: E402

OUT = ROOT / "data_processed" / "q2_forecasts_blend.csv"


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"])
    fc = fused_forecasts(data)
    records = []
    for day in sorted(fc):
        lh, ph = fc[day]
        records.append(pd.DataFrame({
            "date": day, "slot": np.arange(1, 145),
            "lgb_load_kwh": np.asarray(lh, float),
            "lgb_pv_kwh": np.asarray(ph, float),
        }))
    panel = pd.concat(records, ignore_index=True)
    panel.to_csv(OUT, index=False, encoding="utf-8-sig")

    actual = data.pivot(index="date", columns="slot", values="load_kwh")
    lo, hi = pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")
    p = panel.pivot(index="date", columns="slot", values="lgb_load_kwh")
    dates = [d for d in p.index if lo <= d <= hi]
    mae = np.concatenate([np.abs(p.loc[d].to_numpy(float)
                                 - actual.loc[d].to_numpy(float)) for d in dates]).mean() * 6
    print(f"已写出 {OUT.name}: {len(panel)} 行 / {panel.date.nunique()} 天 "
          f"({panel.date.min().date()} ~ {panel.date.max().date()})")
    print(f"面板自检：全年(334天)负荷 MAE = {mae:.2f} kW（应为 131.80 kW）")


if __name__ == "__main__":
    main()
