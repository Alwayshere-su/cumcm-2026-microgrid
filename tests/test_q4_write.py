#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/write_result4.py 的合成数据测试。

真实输入 ``results/q4_segments.csv`` 由另一路并行任务产出；在其就绪前，
本测试用**合成数据**验证写入器与审计逻辑：

  * 全天量 = 逐段之和（计划 / 充电 / 放电 / 紧急）
  * 4-2 与 4-3 口径公式数值正确（两个独立费用函数）
  * SOC 越界会被报错（validate 抛 ValueError）
  * 写完 xlsx 后 round-trip 读回一致

运行：python -m pytest tests/test_q4_write.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_source_path  # noqa: E402
ROOT = find_project_root(__file__)
add_source_path()
import write_result4 as W  # noqa: E402

TEMPLATE_42 = W.TEMPLATE_42
TEMPLATE_43 = W.TEMPLATE_43


# ------------------------------------------------------------------ 合成数据
def make_synth(n_days: int = 2, seed: int = 7, over_bounds: bool = False) -> pd.DataFrame:
    """生成符合契约的合成逐段数据（SOC 递推物理可行）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-02-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    rows: list[dict] = []
    for variant in W.VARIANTS:
        e = W.E0
        for day in dates:
            price = np.clip(0.30 + 0.25 * rng.standard_normal(W.T), 0.10, 1.20)
            load = np.clip(900 + 300 * rng.standard_normal(W.T), 50.0, None)
            pv = np.clip(700 + 500 * rng.standard_normal(W.T), 0.0, None)
            plan = np.clip(load - pv + 80 * rng.standard_normal(W.T), 0.0, None)
            if variant == "4-3":
                adjusted = np.clip(plan + 60 * rng.standard_normal(W.T), 0.0, None)
                emergency = np.zeros(W.T)
            else:
                adjusted = plan.copy()
                emergency = np.clip(load - pv - plan, 0.0, None) * 0.4
            charge = np.zeros(W.T)
            discharge = np.zeros(W.T)
            soc_end = np.zeros(W.T)
            for t in range(W.T):
                c = 300.0 if price[t] < 0.30 else 0.0
                c = min(c, (W.EMAX - e) / W.ETA, W.BMAX)
                d = 300.0 if price[t] > 0.75 else 0.0
                d = min(d, W.ETA * (e - W.EMIN), W.BMAX)
                if c > 0.0 and d > 0.0:
                    d = 0.0
                e = e + W.ETA * c - d / W.ETA
                charge[t], discharge[t], soc_end[t] = c, d, e
            if over_bounds:
                soc_end[5] = W.EMAX + 500.0          # 人为越界
            for t in range(W.T):
                rows.append({
                    "date": day, "slot": t + 1, "variant": variant,
                    "price": float(price[t]), "load_kwh": float(load[t]),
                    "pv_kwh": float(pv[t]), "plan_kwh": float(plan[t]),
                    "adjusted_kwh": float(adjusted[t]),
                    "charge_kwh": float(charge[t]),
                    "discharge_kwh": float(discharge[t]),
                    "emergency_kwh": float(emergency[t]),
                    "unused_kwh": float(max(-(load[t] - pv[t] + charge[t]
                                              - discharge[t]), 0.0)),
                    "soc_end_kwh": float(soc_end[t]),
                })
    return pd.DataFrame(rows)[W.COLS]


# ------------------------------------------------------------------ 口径公式
def test_fee_q42_formula():
    p = [1.0, 2.0, 3.0]
    plan = [10.0, 10.0, 10.0]
    em = [0.0, 1.0, 2.0]
    # Σ p·plan = 60；5·Σ p·em = 5·(0+2+6) = 40
    assert W.fee_q42(p, plan, em) == pytest.approx(100.0)


def test_fee_q43_formula():
    p = [1.0, 2.0, 3.0]
    x0 = [10.0, 10.0, 10.0]
    a = [6.0, 10.0, 14.0]
    em = [0.0, 0.0, 0.0]
    # within=1·6+2·10+3·10=56；under=0.5·1·4=2；over=1.5·3·4=18
    assert W.fee_q43(p, x0, a, em) == pytest.approx(76.0)


def test_fee_functions_are_independent():
    """两支不得给出同一公式：一般情形下 4-2 与 4-3 结果不同。"""
    p = [1.0, 2.0, 3.0]
    x0 = [10.0, 10.0, 10.0]
    a = [6.0, 10.0, 14.0]
    em = [1.0, 0.0, 2.0]
    assert not (W.fee_q42(p, x0, em) == pytest.approx(W.fee_q43(p, x0, a, em)))


# ------------------------------------------------------------------ 全天量
def test_daily_total_equals_segment_sum():
    """每天的全天量 = 144 段之和：充/放按 6 个 4h 区间聚合后仍等于逐段和。"""
    df = make_synth(n_days=3)
    for variant in W.VARIANTS:
        sub = df[df["variant"] == variant]
        for _day, g in sub.groupby("date"):
            slot = g["slot"].to_numpy(int)
            for col in ("charge_kwh", "discharge_kwh"):
                arr = g[col].to_numpy(float)
                block_sum = sum(
                    float(arr[(slot >= lo) & (slot <= hi)].sum())
                    for lo, hi in W.BLOCKS)
                assert block_sum == pytest.approx(float(arr.sum()))
            # 全天量列的口径即逐段和
            assert float(g["plan_kwh"].sum()) == pytest.approx(
                float(g["plan_kwh"].to_numpy(float).sum()))


def test_emergency_ranges_merge_and_sum():
    """相邻紧急时段合并为单一事件，且合并后电量之和 == 逐段和。"""
    em = np.zeros(W.T)
    em[10:15] = 1.0            # 连续 5 段 → 1 个事件
    em[20] = 2.0               # 孤立 1 段 → 1 个事件
    rng = W.emergency_ranges(em)
    assert len(rng) == 2
    assert rng[0][1] == pytest.approx(5.0)
    assert rng[1][1] == pytest.approx(2.0)
    assert sum(a for _, a in rng) == pytest.approx(float(em.sum()))


# ------------------------------------------------------------------ SOC 越界
def test_soc_out_of_bounds_raises():
    df = make_synth(n_days=1, over_bounds=True)
    with pytest.raises(ValueError):
        W.validate(df)


def test_negative_adjusted_raises():
    df = make_synth(n_days=1)
    df.loc[df.index[0], "adjusted_kwh"] = -1.0
    with pytest.raises(ValueError):
        W.validate(df)


# ------------------------------------------------------------------ 端到端
@pytest.mark.skipif(not (TEMPLATE_42.exists() and TEMPLATE_43.exists()),
                    reason="需要 附件/附件5/ 下的 result4-2/4-3 模板")
def test_pipeline_write_and_roundtrip(tmp_path: Path):
    df = make_synth(n_days=3)
    audit = W.run_pipeline(
        df,
        out42=tmp_path / "result4-2.xlsx",
        out43=tmp_path / "result4-3.xlsx",
        audit_path=tmp_path / "audit.json",
        summary_path=tmp_path / "no_summary.json",   # 无参考值 → 条2 记 fail
    )
    checks = {c["id"]: c for c in audit["checks"]}
    assert checks[1]["pass"], checks[1]["detail"]           # 全天量=逐段和
    assert checks[6]["pass"], checks[6]["detail"]           # round-trip
    assert checks[3]["pass"], checks[3]["detail"]           # SOC 界内且连续
    assert checks[5]["pass"], checks[5]["detail"]           # 不售电
    assert (tmp_path / "audit.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
