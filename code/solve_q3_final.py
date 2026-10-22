"""问题三正式主模型与唯一交付入口。

模型：分解⊕Ridge 融合负荷 + 附件3偏差校正/Ridge光伏融合 + 30日成对整日残差
情景两阶段随机规划 + CPWL-DP价值执行器 + 0/6/12/18四时刻滚动。
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
add_script_path()

from ablation_q3_forecast import (build_library, forecast_accuracy, run_sp,
                                  summarize)
from solve_q3 import write_result3
from solve_q2_twolayer import V_CONT

MODEL_ID = "q3_decomposition_ridge_fused_sp_dp_rolling4"
ISSUES = (0, 6, 12, 18)
RESULTS = ROOT / "results"


def official_summary(summary: dict) -> dict:
    return {
        **summary,
        "model_id": MODEL_ID,
        "forecast_load": "decomposition_ridge_fusion",
        "ridge_alpha": 1.0,
        "ridge_retrain_days": 7,
        "forecast_pv": "official_bias_corrected_ridge_fusion",
        "fusion_history_days": 28,
        "scenario_model": "paired_full_day_residuals",
        "residual_window_days": 30,
        "issues": list(ISSUES),
        "continuation_value_yuan_per_kwh": V_CONT,
        "official_settlement": "C",
        "backtest_start": "2025-02-01",
        "backtest_end": "2025-12-31",
    }


def settlement_abc(original, adjusted, price, emergency) -> dict:
    original = np.asarray(original, float)
    adjusted = np.asarray(adjusted, float)
    price = np.asarray(price, float)
    emergency = np.asarray(emergency, float)
    under = np.maximum(original - adjusted, 0.0)
    over = np.maximum(adjusted - original, 0.0)
    within = np.minimum(original, adjusted)
    urgent = float(np.sum(5.0 * price * emergency))
    a_total = float(np.sum(price * original + .5 * price * under
                           + 1.5 * price * over) + urgent)
    b_total = float(np.sum(price * adjusted + .5 * price * under
                           + 1.5 * price * over) + urgent)
    c_total = float(np.sum(price * within + .5 * price * under
                           + 1.5 * price * over) + urgent)
    return {"official_settlement": "C", "A_total": a_total,
            "B_total": b_total, "C_total": c_total,
            "under_kwh": float(under.sum()), "over_kwh": float(over.sum()),
            "emergency_cost": urgent}


def validate_dispatch(out: pd.DataFrame) -> dict:
    x = out.sort_values(["date", "slot"]).reset_index(drop=True)
    sizes = x.groupby("date").size()
    bal = (x.buy_kwh + x.pv_kwh + x.discharge_kwh + x.emergency_kwh
           - x.load_kwh - x.charge_kwh - x.surplus_kwh)
    eta = .9
    soc = x.energy_end_kwh - x.energy_start_kwh - eta * x.charge_kwh \
        + x.discharge_kwh / eta
    ends = x.groupby("date").energy_end_kwh.last().to_numpy()
    starts = x.groupby("date").energy_start_kwh.first().to_numpy()
    gaps = np.abs(ends[:-1] - starts[1:]) if len(ends) > 1 else np.array([0.0])
    return {"days": int(sizes.size), "slots_per_day_min": int(sizes.min()),
            "slots_per_day_max": int(sizes.max()),
            "max_balance_residual_kwh": float(np.abs(bal).max()),
            "max_soc_residual_kwh": float(np.abs(soc).max()),
            "max_cross_day_soc_gap_kwh": float(gaps.max()),
            "min_soc_kwh": float(min(x.energy_start_kwh.min(), x.energy_end_kwh.min())),
            "max_soc_kwh": float(max(x.energy_start_kwh.max(), x.energy_end_kwh.max())),
            "max_simultaneous_charge_discharge_kwh": float(
                np.minimum(x.charge_kwh, x.discharge_kwh).max())}


def add_settlement_audit(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in out.groupby("date", sort=True):
        ans = settlement_abc(g.original_plan_kwh, g.buy_kwh, g.price,
                             g.emergency_kwh)
        rows.append({"date": pd.Timestamp(day), **ans})
    return pd.DataFrame(rows)


def load_inputs():
    data = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                       parse_dates=["date", "timestamp_end"])
    official = pd.read_csv(ROOT / "data_processed/pv_forecast_hourly.csv",
                           parse_dates=["issue_at", "target_end"])
    stat = pd.read_csv(ROOT / "data_processed/q3_forecasts_noml.csv",
                       parse_dates=["date"])
    ridge = pd.read_csv(ROOT / "data_processed/q3_forecasts_fused.csv",
                        parse_dates=["date"])
    prm = json.loads((ROOT / "data_processed/q1_storage_parameters.json")
                     .read_text(encoding="utf-8"))
    return data, official, stat, ridge, prm


def run(days: int | None = None, scenarios: int = 30):
    data, official, stat, ridge, prm = load_inputs()
    lib = build_library("ridge", "fused", data, official, stat, ridge)
    dates = pd.date_range("2025-02-01", "2025-12-31")
    if days is not None:
        dates = dates[:days]
    # build_library 的内部键仍沿用 ridge，以读取历史兼容列名；实际输入文件
    # q3_forecasts_fused.csv 的 lgb_load_kwh 已是分解⊕Ridge 融合预测。
    cfg = {"name": "decomposition_ridge_fused", "load": "ridge", "pv": "fused"}
    out, daily = run_sp(cfg, lib, dates, prm, scenarios, ISSUES)
    audit = add_settlement_audit(out)
    summary = summarize(MODEL_ID, out, daily)
    summary.update(validate_dispatch(out))
    summary["forecast_accuracy"] = forecast_accuracy(lib, dates)
    summary["A_total_yuan"] = float(audit.A_total.sum())
    summary["B_total_yuan"] = float(audit.B_total.sum())
    summary["C_total_yuan"] = float(audit.C_total.sum())
    return out, daily, audit, official_summary(summary)


def save_outputs(out, daily, audit, summary) -> None:
    RESULTS.mkdir(exist_ok=True)
    out = out.copy()
    out["adjusted_buy_kwh"] = out["buy_kwh"]
    out.to_csv(RESULTS / "q3_final_dispatch.csv", index=False,
               encoding="utf-8-sig")
    out.to_csv(RESULTS / "q3_dispatch_10min.csv", index=False,
               encoding="utf-8-sig")
    daily.to_csv(RESULTS / "q3_final_daily.csv", index=False,
                 encoding="utf-8-sig")
    audit.to_csv(RESULTS / "q3_settlement_abc.csv", index=False,
                 encoding="utf-8-sig")
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    (RESULTS / "q3_final_summary.json").write_text(payload, encoding="utf-8")
    (RESULTS / "q3_summary.json").write_text(payload, encoding="utf-8")
    write_result3(out, ROOT / "附件/附件5/result3.xlsx", ROOT / "result3.xlsx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    out, daily, audit, summary = run(args.days, args.scenarios)
    if args.save:
        save_outputs(out, daily, audit, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
