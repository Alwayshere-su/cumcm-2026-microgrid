"""Aggregate the official Q3 full-year run into VOI and representative tables."""
from __future__ import annotations

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

from solve_q3 import _emergency_ranges
from solve_q3_final import settlement_abc

# 四条策略统一为「融合负荷」口径（负荷=分解⊕Ridge 融合，MAE 131.80）。
# P0/P06/P0612 由 run_q3_voi_fusion.py 以与 q3_final 完全相同的管线重跑，
# 仅改变允许使用的发布时刻；不再复用旧 Ridge 负荷的 abl_sp_ridge_fused_i0[-6][-6-12]。
POLICIES = {
    "P0": "q3_voi_fused_p0",
    "P06": "q3_voi_fused_p06",
    "P0612": "q3_voi_fused_p0612",
    "P061218": "q3_final",
}
REQUIRED = pd.to_datetime(["2025-03-20", "2025-06-21",
                           "2025-09-23", "2025-12-21"])
SLOTS = [61, 73, 85, 97, 109, 121]


def _load_policy(name: str, stem: str):
    summary = json.loads((RESULTS / f"{stem}_summary.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(RESULTS / f"{stem}_daily.csv", parse_dates=["date"])
    dispatch = pd.read_csv(RESULTS / f"{stem}_dispatch.csv", parse_dates=["date"])
    return summary, daily, dispatch


def build_policy_tables():
    summaries, daily = [], {}
    for policy, stem in POLICIES.items():
        s, d, _ = _load_policy(policy, stem)
        summaries.append({"policy": policy, "issues": {"P0": "0", "P06": "0/6",
                          "P0612": "0/6/12", "P061218": "0/6/12/18"}[policy],
                          "total_cost_yuan": s["total"],
                          "emergency_cost_yuan": s["emergency_cost"],
                          "emergency_kwh": s["emergency_kwh"],
                          "cvar95_yuan": s["cvar95"], "max_day_yuan": s["max_day"]})
        daily[policy] = d.set_index("date")["total"].sort_index()
    table = pd.DataFrame(summaries)
    voi = pd.DataFrame({"voi_6_yuan": daily["P0"] - daily["P06"],
                        "voi_12_yuan": daily["P06"] - daily["P0612"],
                        "voi_18_yuan": daily["P0612"] - daily["P061218"]}).reset_index()
    return table, voi


def representative_tables(out: pd.DataFrame):
    purchases, storage, emergencies, fees = [], [], [], []
    for day in REQUIRED:
        x = out[out.date == day].sort_values("slot").copy()
        if len(x) != 144:
            raise ValueError(f"missing official day {day.date()}")
        abc = settlement_abc(x.original_plan_kwh, x.buy_kwh, x.price, x.emergency_kwh)
        plan_cost = float(np.sum(x.price * x.original_plan_kwh))
        fees.append({"date": day, "plan_kwh": x.original_plan_kwh.sum(),
                     "adjusted_kwh": x.buy_kwh.sum(), "plan_cost_yuan": plan_cost,
                     **abc, "emergency_kwh": x.emergency_kwh.sum(),
                     "soc_0_kwh": x.energy_start_kwh.iloc[0],
                     "soc_24_kwh": x.energy_end_kwh.iloc[-1]})
        for slot in SLOTS:
            r = x[x.slot == slot].iloc[0]
            purchases.append({"date": day, "slot": slot,
                              "interval": f"{(slot-1)*10//60}:{(slot-1)*10%60:02d}-"
                                          f"{slot*10//60}:{slot*10%60:02d}",
                              "original_plan_kwh": r.original_plan_kwh,
                              "adjusted_buy_kwh": r.buy_kwh})
        x["bin"] = (x.slot - 1) // 24
        agg = x.groupby("bin").agg(charge_kwh=("charge_kwh", "sum"),
                                    discharge_kwh=("discharge_kwh", "sum"))
        for b, r in agg.iterrows():
            storage.append({"date": day, "interval": f"{4*b}:00-{4*(b+1)}:00",
                            **r.to_dict(), "soc_0_kwh": x.energy_start_kwh.iloc[0] if b == 0 else np.nan,
                            "soc_24_kwh": x.energy_end_kwh.iloc[-1] if b == 0 else np.nan})
        for interval, amount in _emergency_ranges(x):
            emergencies.append({"date": day, "interval": interval,
                                "emergency_kwh": amount})
    return (pd.DataFrame(purchases), pd.DataFrame(storage),
            pd.DataFrame(emergencies), pd.DataFrame(fees))


def main():
    policy, voi = build_policy_tables()
    out = pd.read_csv(RESULTS / "q3_final_dispatch.csv", parse_dates=["date"])
    purchase, storage, emergency, fees = representative_tables(out)
    policy.to_csv(RESULTS / "q3_policy_summary.csv", index=False, encoding="utf-8-sig")
    voi.to_csv(RESULTS / "q3_voi_daily.csv", index=False, encoding="utf-8-sig")
    purchase.to_csv(RESULTS / "q3_representative_purchase.csv", index=False, encoding="utf-8-sig")
    storage.to_csv(RESULTS / "q3_representative_storage.csv", index=False, encoding="utf-8-sig")
    emergency.to_csv(RESULTS / "q3_representative_emergency.csv", index=False, encoding="utf-8-sig")
    fees.to_csv(RESULTS / "q3_representative_fees.csv", index=False, encoding="utf-8-sig")
    marginal = {
        "voi6_total_yuan": float(policy.loc[0, "total_cost_yuan"] - policy.loc[1, "total_cost_yuan"]),
        "voi12_total_yuan": float(policy.loc[1, "total_cost_yuan"] - policy.loc[2, "total_cost_yuan"]),
        "voi18_total_yuan": float(policy.loc[2, "total_cost_yuan"] - policy.loc[3, "total_cost_yuan"]),
        "voi6_positive_days": int((voi.voi_6_yuan > 0).sum()),
        "voi12_positive_days": int((voi.voi_12_yuan > 0).sum()),
        "voi18_positive_days": int((voi.voi_18_yuan > 0).sum()),
    }
    (RESULTS / "q3_voi_summary.json").write_text(
        json.dumps(marginal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(policy.to_string(index=False))
    print(json.dumps(marginal, ensure_ascii=False, indent=2))
    print(fees.to_string(index=False))


if __name__ == "__main__":
    main()
