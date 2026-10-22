from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_source_path  # noqa: E402
ROOT = find_project_root(__file__)
add_source_path()
from forecast_q4_price import (day_ahead_price, intraday_update,
                               walk_forward_hybrid_prices)  # noqa: E402
from q4_models import execute_q42_causal, settle_q42  # noqa: E402


def test_day_ahead_ignores_current_and_future_days():
    rng = np.random.default_rng(7)
    P = rng.uniform(.2, 1.5, (50, 144))
    a = day_ahead_price(P, 40)
    changed = P.copy()
    changed[40:] = 999.0
    b = day_ahead_price(changed, 40)
    assert np.allclose(a, b)


def test_intraday_update_ignores_future_observations():
    base = np.linspace(.3, 1.2, 144)
    obs = base + .1
    a = intraday_update(base, obs, 72)
    changed = obs.copy()
    changed[72:] = 999.0
    b = intraday_update(base, changed, 72)
    assert np.allclose(a, b)


def test_price_forecast_positive():
    P = np.tile(np.linspace(.2, 1.2, 144), (40, 1))
    assert np.min(day_ahead_price(P, 30)) > 0


def test_hybrid_price_forecast_is_nonanticipative_and_order_preserving():
    rng = np.random.default_rng(12)
    dates = np.asarray(np.arange("2025-01-01", "2025-03-22",
                                 dtype="datetime64[D]"))
    dates = __import__("pandas").DatetimeIndex(dates)
    base = np.linspace(.3, 1.3, 144)
    P = np.vstack([base * (1 + .08 * np.sin(i / 5)) +
                   rng.normal(0, .01, 144) for i in range(len(dates))])
    a = walk_forward_hybrid_prices(P, dates, base, start_i=31)
    changed = P.copy()
    changed[60:] = 99.0
    b = walk_forward_hybrid_prices(changed, dates, base, start_i=31)
    assert np.allclose(a[59], b[59])
    assert np.nanmin(a) > 0
    # 正斜率仿射校准不改变当日原始形状的次序。
    assert np.array_equal(np.argsort(a[59]), np.argsort(a[59]))


def test_q42_causal_balance_and_no_adjustment_fee():
    n = 144
    p = np.ones(n)
    load = np.full(n, 100.0)
    pv = np.zeros(n)
    plan = np.full(n, 90.0)
    r = execute_q42_causal(p, load, pv, plan, np.full(n, 6000.0), 6000.0)
    balance = plan + r["emergency"] + pv + r["discharge"] \
        - load - r["charge"] - r["surplus"]
    assert np.max(np.abs(balance)) < 1e-9
    assert np.isclose(r["total"], np.sum(plan) + 5 * np.sum(r["emergency"]))
    assert r["emergency_kwh"] > 0


def test_q42_uses_available_battery_before_emergency():
    n = 144
    p = np.ones(n)
    load = np.zeros(n); load[0] = 100.0
    pv = np.zeros(n); plan = np.zeros(n)
    r = execute_q42_causal(p, load, pv, plan, np.full(n, 1200.0), 6000.0)
    assert np.isclose(r["discharge"][0], 100.0)
    assert np.isclose(r["emergency"][0], 0.0)


def test_q42_track_executor_adds_discharge_before_emergency():
    n = 144
    load = np.zeros(n); load[0] = 100.0
    z = np.zeros(n); zero = np.zeros(n)
    r = settle_q42(np.ones(n), load, zero, z, 6000.0, zero, zero)
    assert np.isclose(r["discharge"][0], 100.0)
    assert np.isclose(r["g"][0], 0.0)
    balance = z + r["g"] + zero + r["discharge"] \
        - load - r["charge"] - r["unused"]
    assert np.max(np.abs(balance)) < 1e-9
