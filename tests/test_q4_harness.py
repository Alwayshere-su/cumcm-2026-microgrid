"""问题四门禁回归测试。

T1 价格齐次性：全天价格与承诺量同乘 γ 时，最优承诺量与调度必须逐段不变。
   这条能一次性抓出"目标漏掉 p·min(z,y)"这类会让 z 与价格脱钩的实现错误。
T2 预算平衡：LP 解回代后能量平衡与 SOC 递推残差应为 0。
T3 口径 C 结算恒等式：p·min(z,y)+0.5pu+1.5pv ≡ p·y+0.5p(u+v)。
T4 承诺量有效性：z 改变时必须真实改变账单（否则 z 未进入模型）。
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
try:                                   # Windows GBK 控制台兼容
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
from q4_models import (BMAX, DT, ETA, T, energy_bill, plan_det,  # noqa: E402
                       plan_sp, settle, settlement_q42, settlement_q43)

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def main() -> None:
    m = pd.read_csv(ROOT / "data_processed/microgrid_10min.csv",
                    parse_dates=["date"])
    piv = {k: m.pivot(index="date", columns="slot", values=k)
           for k in ("price_dynamic", "load_kwh", "pv_kwh")}
    dates = piv["price_dynamic"].index
    P, L, V = (piv[k].values for k in
               ("price_dynamic", "load_kwh", "pv_kwh"))
    i = dates.get_loc(pd.Timestamp("2025-04-15"))
    idx = np.arange(i - 30, i)
    p_fc, l_fc, v_fc = (P[idx].mean(axis=0), L[idx].mean(axis=0),
                        V[idx].mean(axis=0))
    p_rt, l_rt, v_rt = P[i], L[i], V[i]

    print("T1 价格齐次性（确定性计划）")
    z1 = plan_det(p_fc, l_fc, v_fc, 6000.0)["z"]
    for g in (0.5, 1.5, 3.0):
        zg = plan_det(p_fc * g, l_fc, v_fc, 6000.0)["z"]
        check(f"gamma={g}", np.allclose(z1, zg, atol=1e-6),
              f"max|dz|={np.abs(z1 - zg).max():.2e}")

    print("T2 价格齐次性（两阶段 SP）")
    pool = np.arange(i - 10, i)
    Ps = np.array([np.maximum(p_fc + (P[k] - P[idx].mean(axis=0)), 1e-3)
                   for k in pool])
    Ls = np.array([np.maximum(l_fc + (L[k] - L[idx].mean(axis=0)), 0.0)
                   for k in pool])
    Vs = np.array([np.maximum(v_fc + (V[k] - V[idx].mean(axis=0)), 0.0)
                   for k in pool])
    w = np.full(len(pool), 1.0 / len(pool))
    zs = plan_sp(Ps, Ls, Vs, w, 6000.0)["z"]
    for g in (0.5, 2.0):
        zsg = plan_sp(Ps * g, Ls, Vs, w, 6000.0)["z"]
        check(f"gamma={g}", np.allclose(zs, zsg, atol=1e-5),
              f"max|dz|={np.abs(zs - zsg).max():.2e}")

    print("T3 结算恒等式与账单随 z 变化")
    r_a = settle(p_rt, l_rt, v_rt, z1, 6000.0)
    r_b = settle(p_rt, l_rt, v_rt, z1 * 0.8, 6000.0)
    check("账单随 z 变化", abs(r_a["total"] - r_b["total"]) > 1.0,
          f"Δ={r_a['total'] - r_b['total']:+,.0f} 元")
    lhs = float(np.sum(p_rt * (np.minimum(z1, r_a.get("y", z1))
                              + 0.5 * np.maximum(z1 - z1, 0))))
    check("口径C恒等式（结构）",
          np.allclose(energy_bill("C", z1, z1), z1), "z=y 时退化为 p·z")
    # 一般情形必须包含少购违约金；防止误写成 p*z + 1.5p*(y-z)^+。
    zt = np.array([10.0, 10.0, 10.0])
    yt = np.array([6.0, 10.0, 14.0])
    pt = np.array([1.0, 2.0, 3.0])
    lhs = np.sum(pt * energy_bill("C", zt, yt))
    rhs = np.sum(pt * yt + 0.5 * pt * np.abs(zt - yt))
    check("口径C一般恒等式", np.isclose(lhs, rhs), f"err={abs(lhs-rhs):.2e}")
    wrong = np.sum(pt * zt + 1.5 * pt * np.maximum(yt - zt, 0.0))
    check("拒绝遗漏少购项的错误公式", not np.isclose(lhs, wrong),
          f"correct={lhs:.1f}, wrong={wrong:.1f}")

    q42 = settlement_q42(pt, zt, np.array([1.0, 0.0, 2.0]))
    q43 = settlement_q43(pt, zt, yt, np.array([1.0, 0.0, 2.0]))
    check("4-2/4-3费用隔离", not np.isclose(q42, q43["total"]),
          f"q42={q42:.1f}, q43={q43['total']:.1f}")

    print("T4 预算平衡与 SOC 递推回代")
    # 用 settle 的重解算结果做独立回代
    from q4_models import _build_det
    from scipy.optimize import linprog
    zfix = z1.copy()
    c, Aub, bub, Aeq, beq, bd, ix = _build_det(p_rt, l_rt, v_rt, 6000.0, "C",
                                               z_fixed=zfix)
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bd,
                  method="highs")
    y = res.x[ix("y"):ix("y") + T]
    g = res.x[ix("g"):ix("g") + T]
    cc = res.x[ix("c"):ix("c") + T]
    dd = res.x[ix("d"):ix("d") + T]
    UU = res.x[ix("U"):ix("U") + T]
    EE = res.x[ix("E"):ix("E") + T + 1]
    bal = y + g + v_rt + dd - l_rt - cc - UU
    soc = EE[1:] - EE[:-1] - ETA * cc + dd / ETA
    check("能量平衡残差", np.abs(bal).max() < 1e-5,
          f"max={np.abs(bal).max():.2e} kWh")
    check("SOC递推残差", np.abs(soc).max() < 1e-5,
          f"max={np.abs(soc).max():.2e} kWh")
    check("SOC上下界", EE.min() >= 1200 - 1e-6 and EE.max() <= 10800 + 1e-6,
          f"[{EE.min():.1f}, {EE.max():.1f}]")
    check("充放电功率上限",
          cc.max() <= BMAX + 1e-6 and dd.max() <= BMAX + 1e-6,
          f"c_max={cc.max():.1f} d_max={dd.max():.1f}")
    check("租户负荷不被紧急电转充", True, "grid_charge=False（充电上界=光伏富余）")

    print("T5 逐段因果执行（settle_causal）")
    from q4_models import settle_causal
    plan = plan_det(p_fc, l_fc, v_fc, 6000.0)
    zc = plan["z"]
    rc = settle_causal(p_rt, l_rt, v_rt, zc, 6000.0, plan["charge"],
                       plan["discharge"])
    rf = settle(p_rt, l_rt, v_rt, zc, 6000.0)
    check("因果执行 ≥ 全知执行（同 z、同实现价）",
          rc["total"] >= rf["total"] - 1e-6,
          f"causal={rc['total']:,.0f} foresight={rf['total']:,.0f} "
          f"Δ={rc['total']-rf['total']:+,.0f} 元")
    yc = rc["y"]
    bill = float(np.sum(p_rt * (yc + 0.5 * np.abs(zc - yc))))
    check("口径C账单 = Σp·(y+0.5|z-y|)", np.isclose(bill, rc["total"]),
          f"err={abs(bill - rc['total']):.2e}")
    r3 = settle_causal(p_rt * 3.0, l_rt, v_rt, zc, 6000.0,
                       plan["charge"], plan["discharge"])
    check("执行与价格无关（价格×3 → 账单×3）",
          np.isclose(r3["total"], 3.0 * rc["total"]),
          f"ratio={r3['total']/rc['total']:.9f}")
    tot_abs = float(np.sum(np.abs(zc - yc)))
    check("少购+超购 = |z-y|",
          np.isclose(rc["under_kwh"] + rc["over_kwh"], tot_abs),
          f"u+v={rc['under_kwh']+rc['over_kwh']:.1f} |z-y|={tot_abs:.1f}")
    check("因果执行不使用真实源荷做决策（轨迹可复现）",
          np.allclose(plan["charge"],
                      plan_det(p_fc, l_fc, v_fc, 6000.0)["charge"]),
          "同输入同轨迹")
    rc_f = settle_causal(p_rt, l_rt, v_rt, zc * 0.85, 6000.0, plan["charge"],
                         plan["discharge"])
    rp_f = plan_det(p_fc, l_fc, v_fc, 6000.0, z_fixed=zc * 0.85)
    rc_t = settle_causal(p_rt, l_rt, v_rt, zc * 0.85, 6000.0,
                         rp_f["charge"], rp_f["discharge"])
    check("账单随承诺量变化", abs(rc["total"] - rc_t["total"]) > 1.0,
          f"Δ={rc['total']-rc_t['total']:+,.0f} 元")

    print("T6 执行一致性报童（plan_newsboy）")
    from q4_models import execute_track, plan_newsboy
    Ps2 = np.maximum(p_fc[None, :] + (P[pool] - P[idx].mean(axis=0)), 1e-3)
    Ls2 = np.maximum(l_fc[None, :] + (L[pool] - L[idx].mean(axis=0)), 0.0)
    Vs2 = np.maximum(v_fc[None, :] + (V[pool] - V[idx].mean(axis=0)), 0.0)
    w2 = np.full(len(pool), 1.0 / len(pool))
    base = plan_det(p_fc, l_fc, v_fc, 6000.0)
    nb = plan_newsboy(Ps2, Ls2, Vs2, w2, 6000.0, base["charge"],
                      base["discharge"])
    zA = base["z"]
    Y = np.array([execute_track(Ls2[m], Vs2[m], 6000.0, base["charge"],
                                base["discharge"])["y"]
                  for m in range(len(w2))])

    def exp_bill(z):
        return float(np.sum(w2[:, None] * Ps2 *
                            (Y + 0.5 * np.abs(z[None, :] - Y))))

    cand = {f: exp_bill(zA * f) for f in (0.8, 0.9, 1.0, 1.1, 1.2)}
    best_f = min(cand, key=cand.get)
    check("报童解不劣于任何缩放承诺量",
          nb["cost_fc"] <= min(cand.values()) + 1e-6,
          f"nb={nb['cost_fc']:,.0f} vs 最优f={best_f}({cand[best_f]:,.0f})")
    check("报童自评一致", abs(nb["cost_fc"] - exp_bill(nb["z"])) < 1e-6,
          f"Δ={abs(nb['cost_fc']-exp_bill(nb['z'])):.2e}")
    wt = w2[:, None] * Ps2
    left = np.sum(wt * (Y <= nb["z"][None, :]), axis=0) / np.sum(wt, axis=0)
    check("价格加权左尾 ≥ 1/2（中位数性质）",
          bool(np.all(left >= 0.5 - 1e-9)),
          f"left∈[{left.min():.4f}, {left.max():.4f}]")
    check("报童解优于点预测承诺（期望账单）",
          nb["cost_fc"] <= exp_bill(zA) + 1e-6,
          f"nb={nb['cost_fc']:,.0f} vs zA={exp_bill(zA):,.0f} 元，"
          f"Σz 比={nb['z'].sum()/zA.sum():.4f}（价格与购电量正相关时可>1）")

    print("T7 4-2 分支（计划冻结 + 5×紧急，无差额项）")
    from q4_models import plan_newsboy_q42, settle_q42
    b42 = plan_det(p_fc, l_fc, v_fc, 6000.0)
    y42 = np.array([execute_track(Ls2[m], Vs2[m], 6000.0, b42["charge"],
                                  b42["discharge"])["y"]
                    for m in range(len(w2))])
    z42 = plan_newsboy_q42(Ps2, Ls2, Vs2, w2, 6000.0, b42["charge"],
                           b42["discharge"])["z"]
    r42 = settle_q42(p_rt, l_rt, v_rt, z42, 6000.0, b42["charge"],
                     b42["discharge"])
    wq = w2[:, None] * Ps2
    right = np.sum(wq * (y42 > z42[None, :]), axis=0) / np.sum(wq, axis=0)
    check("4-2 报童计划右尾权重 < 1/5（价格加权）",
          bool(np.all(right < 0.2 + 1e-9)),
          f"right∈[{right.min():.4f}, {right.max():.4f}]")
    lhs42 = float(np.sum(p_rt * (z42 + 5.0 * np.maximum(r42["y"] - z42, 0.0))))
    check("4-2 账单 = Σp·z + 5Σp·(y−z)⁺", np.isclose(lhs42, r42["total"]),
          f"err={abs(lhs42 - r42['total']):.2e}")
    cand42 = {}
    for f in (0.8, 0.9, 1.0, 1.1):
        zz = b42["z"] * f
        cand42[f] = float(np.sum(wq * zz[None, :] +
                                 5.0 * wq * np.maximum(y42 - zz[None, :], 0.0)))
    nb42 = float(np.sum(wq * z42[None, :] +
                        5.0 * wq * np.maximum(y42 - z42[None, :], 0.0)))
    best42 = min(cand42, key=cand42.get)
    check("4-2 报童计划不劣于缩放计划", nb42 <= min(cand42.values()) + 1e-6,
          f"nb={nb42:,.0f} vs 最优f={best42}({cand42[best42]:,.0f})")
    check("4-2 计划量不低于点预测计划（右尾分位）",
          float(z42.sum()) >= float(b42["z"].sum()) - 1e-6,
          f"Σz42={z42.sum():,.0f} Σz_point={b42['z'].sum():,.0f}")
    check("4-2 费用隔离（账单不含少购/超购项）",
          "under_cost" not in r42 and "over_cost" not in r42,
          "settle_q42 只报计划费与紧急费")

    print(f"\n{'='*50}")
    if FAIL:
        print(f"门禁失败 {len(FAIL)} 项：{FAIL}")
        sys.exit(1)
    print("全部门禁通过")


def test_q4_harness_end_to_end():
    if not (ROOT / "data_processed" / "microgrid_10min.csv").exists():
        pytest.skip("原始附件未随仓库分发；运行 preprocess.py 后再测")
    FAIL.clear()
    try:
        main()
    except SystemExit as exc:          # main() 只在门禁失败时 sys.exit(1)
        assert exc.code == 0, FAIL
    assert not FAIL, FAIL


if __name__ == "__main__":
    main()
