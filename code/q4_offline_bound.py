"""计算全年离线下界（整年一次 LP，跨日 SOC 连续、无终端约束）。

这是任意可行因果策略费用的下界，用于量化遗憾值。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()
from q4_models import offline_annual  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid-charge", action="store_true",
                    help="禁止网电充电（须与 harness 同设定，否则下界不可比）")
    args = ap.parse_args()
    gc = not args.no_grid_charge
    print(f"网电充电：{'允许' if gc else '禁止'}", flush=True)
    m = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                    parse_dates=["date"])
    piv = {k: m.pivot(index="date", columns="slot", values=k)
           for k in ("price_dynamic", "load_kwh", "pv_kwh")}
    dates = piv["price_dynamic"].index
    P, L, V = (piv[k].values for k in
               ("price_dynamic", "load_kwh", "pv_kwh"))
    lo = dates.get_loc(pd.Timestamp("2025-02-01"))
    hi = dates.get_loc(pd.Timestamp("2025-12-31"))
    p, l, v = P[lo:hi + 1], L[lo:hi + 1], V[lo:hi + 1]
    print(f"窗口 {dates[lo].date()} ~ {dates[hi].date()}，{len(p)} 天", flush=True)
    t0 = time.perf_counter()
    out = offline_annual(p, l, v, grid_charge=gc)
    el = time.perf_counter() - t0
    print(f"离线下界 = {out['cost']:,.2f} 元   ({el:.0f}s)", flush=True)
    daily = np.sum(p * out["buy"], axis=1)
    (ROOT / "results" / "q4_offline_bound.json").write_text(
        json.dumps({"cost": out["cost"], "days": len(p), "grid_charge": gc,
                    "window": [str(dates[lo].date()), str(dates[hi].date())],
                    "daily_cost": daily.tolist(),
                    "terminal_soc": float(out["soc"][-1, -1]),
                    "elapsed_s": el}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"末态 SOC = {out['soc'][-1, -1]:.1f} kWh", flush=True)


if __name__ == "__main__":
    main()
