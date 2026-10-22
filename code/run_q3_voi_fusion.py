# -*- coding: utf-8 -*-
"""按「融合负荷」口径重跑问题三的四条 VOI 策略（P0/P06/P0612），使 §11.2 表口径统一。

根因：`report_q3_final.py` 目前把 P0/P06/P0612 指向
`abl_sp_ridge_fused_i0[-6][-6-12]`（ablation_q3_forecast.py 的 "ridge_fused" 是
「负荷=纯岭回归(MAE 162.23)+ 光伏=校正融合」，见其脚本第 11 行注释），
而 P061218 指向 `q3_final`（负荷=分解⊕Ridge 融合，MAE 131.80）。四条策略负荷口径不一致。

本脚本完全复用 `solve_q3_final.py` 的正式管线（读 `q3_forecasts_fused.csv` 融合负荷 +
融合光伏 + 30 日成对整日残差 + SP+DP 价值执行 + 口径C），仅改变允许使用的发布时刻
issues，得到与正式模型同口径的 P0/P06/P0612。P061218 直接用已生成的 `q3_final`。

输出 stems：q3_voi_fused_p0 / p06 / p0612（summary/daily/dispatch）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
RESULTS = ROOT / "results"
add_script_path()

from ablation_q3_forecast import (build_library, forecast_accuracy,  # noqa: E402
                                  run_sp, summarize)
from solve_q3_final import (MODEL_ID, add_settlement_audit,  # noqa: E402
                            load_inputs, validate_dispatch)

VOI_RUNS = {
    "p0": (0,),
    "p06": (0, 6),
    "p0612": (0, 6, 12),
}


def save(stem: str, out: pd.DataFrame, daily: pd.DataFrame, summary: dict) -> None:
    out.to_csv(RESULTS / f"{stem}_dispatch.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(RESULTS / f"{stem}_daily.csv", index=False, encoding="utf-8-sig")
    (RESULTS / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="仅用于冒烟测试：限制回测天数")
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--only", choices=list(VOI_RUNS), default=None)
    args = ap.parse_args()

    runs = {args.only: VOI_RUNS[args.only]} if args.only else VOI_RUNS
    cfg = {"name": "decomposition_ridge_fused", "load": "ridge", "pv": "fused"}

    data, official, stat, ridge, prm = load_inputs()
    lib = build_library("ridge", "fused", data, official, stat, ridge)

    full_dates = pd.date_range("2025-02-01", "2025-12-31")
    print("全口径融合负荷预测精度（全年）:",
          json.dumps(forecast_accuracy(lib, full_dates), ensure_ascii=False), flush=True)

    dates = full_dates if args.days is None else full_dates[: args.days]
    for stem, issues in runs.items():
        out, daily = run_sp(cfg, lib, dates, prm, args.scenarios, issues)
        audit = add_settlement_audit(out)
        summary = summarize(f"q3_voi_fused_{stem}", out, daily)
        summary.update(validate_dispatch(out))
        summary["forecast_accuracy"] = forecast_accuracy(lib, dates)
        summary["A_total_yuan"] = float(audit.A_total.sum())
        summary["B_total_yuan"] = float(audit.B_total.sum())
        summary["C_total_yuan"] = float(audit.C_total.sum())
        summary["model_id"] = MODEL_ID
        summary["issues"] = list(issues)
        save(f"q3_voi_fused_{stem}", out, daily, summary)
        print(json.dumps({"stem": stem, "issues": list(issues),
                          "total": summary["total"],
                          "emergency_cost": summary["emergency_cost"],
                          "emergency_kwh": summary["emergency_kwh"],
                          "load_mae_kw": summary["forecast_accuracy"]["load_mae_kw"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
