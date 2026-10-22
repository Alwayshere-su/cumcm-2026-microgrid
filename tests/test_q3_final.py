import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_source_path  # noqa: E402
ROOT = find_project_root(__file__)
add_source_path()

from solve_q3_final import (MODEL_ID, ISSUES, official_summary,
                            settlement_abc, validate_dispatch)
from report_q3_final import build_policy_tables


def synthetic_dispatch():
    rows = []
    energy = 6000.0
    for day in pd.date_range("2025-02-01", periods=2):
        for slot in range(1, 145):
            start = energy
            rows.append({
                "date": day, "slot": slot,
                "original_plan_kwh": 10.0, "buy_kwh": 10.0,
                "charge_kwh": 0.0, "discharge_kwh": 0.0,
                "surplus_kwh": 0.0, "emergency_kwh": 0.0,
                "energy_start_kwh": start, "energy_end_kwh": start,
                "load_kwh": 10.0, "pv_kwh": 0.0, "price": 1.0,
            })
    return pd.DataFrame(rows)


def test_official_model_identity_and_four_issue_times():
    assert MODEL_ID == "q3_decomposition_ridge_fused_sp_dp_rolling4"
    assert ISSUES == (0, 6, 12, 18)
    meta = official_summary({"days": 334})
    assert meta["forecast_load"] == "decomposition_ridge_fusion"
    assert meta["forecast_pv"] == "official_bias_corrected_ridge_fusion"
    assert meta["scenario_model"] == "paired_full_day_residuals"
    assert meta["residual_window_days"] == 30


def test_settlement_abc_has_c_as_official_and_no_double_counting():
    original = np.array([10.0, 10.0])
    adjusted = np.array([8.0, 13.0])
    price = np.ones(2)
    emergency = np.zeros(2)
    ans = settlement_abc(original, adjusted, price, emergency)
    assert abs(ans["C_total"] - 23.5) < 1e-9
    assert ans["official_settlement"] == "C"
    assert abs(ans["under_kwh"] - 2.0) < 1e-9
    assert abs(ans["over_kwh"] - 3.0) < 1e-9


def test_dispatch_validator_checks_144_slots_physics_and_cross_day_soc():
    report = validate_dispatch(synthetic_dispatch())
    assert report["days"] == 2
    assert report["slots_per_day_min"] == 144
    assert report["slots_per_day_max"] == 144
    assert report["max_balance_residual_kwh"] < 1e-9
    assert report["max_soc_residual_kwh"] < 1e-9
    assert report["max_cross_day_soc_gap_kwh"] < 1e-9


def test_saved_official_summary_is_not_an_old_q3_model_when_present():
    path = ROOT / "results/q3_summary.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model_id"] == MODEL_ID
    assert saved["issues"] == [0, 6, 12, 18]


def test_saved_dispatch_contains_all_required_dates_when_present():
    path = ROOT / "results/q3_dispatch_10min.csv"
    if not path.exists():
        return
    out = pd.read_csv(path, parse_dates=["date"])
    required = {pd.Timestamp(x) for x in
                ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")}
    assert required.issubset(set(out.date.unique()))
    assert out.groupby("date").size().eq(144).all()


def test_full_year_policy_report_has_unique_date_column_when_inputs_exist():
    needed = ROOT / "results/abl_sp_ridge_fused_i0_summary.json"
    if not needed.exists():
        return
    policy, voi = build_policy_tables()
    assert policy.policy.tolist() == ["P0", "P06", "P0612", "P061218"]
    assert voi.columns.tolist().count("date") == 1
    assert len(voi) == 334
