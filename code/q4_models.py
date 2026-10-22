"""问题四 harness 核心模型库（v3）。

提供三条建模路线的求解原语（全部为 10 min × 144 段、口径 C 或 B）：

  plan_det     确定性日前调度（单条预测）→ 计划量 + 储能执行轨迹
  plan_sp      两阶段随机规划：一阶段 z 全情景共享，二阶段逐情景追索
  settle       全知执行（对照）：给定承诺 z，用真实源荷一次性求最优执行 → 偏乐观
  settle_causal 逐段因果执行（正式）：储能按日初计划轨迹推进，偏差由实际购电承担

结算口径
  C：p·min(z,y) + 0.5p·(z−y)⁺ + 1.5p·(y−z)⁺ + 5p·g
  B：p·min(z,y) + 0.5p·|z−y|         + 5p·g

建模历史（务必保留，避免重复踩坑）
  v1 目标漏掉 p·z  → z 对价格不敏感、储能轨迹与电价无关，结论失效
  v2 补 p·z 后不可行 → 超购的计划电量无处可去，必须补"未利用供能 U"
  v3 增加多情景 SP，供 harness 做公平对比
  v4 settle 由"全知源荷一次性 LP 追索"改为"逐段因果执行"。旧口径等于执行层
     提前知道全天真实源荷（偏乐观），会系统性低估偏差费与紧急购电风险；
     settle_causal 只在日初用预测信息定储能轨迹，之后逐段推进，
     实际购电量 y_t 由该段平衡唯一确定，差额按口径 C 结算。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

ETA, EMIN, EMAX, PMAX, E0 = 0.9, 1200.0, 10800.0, 5000.0, 6000.0
DT, T = 1.0 / 6.0, 144
BMAX = PMAX * DT


def energy_bill(rule, z, y):
    """口径 C / B 下计费的等效电量。"""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    under = np.maximum(z - y, 0.0)
    over = np.maximum(y - z, 0.0)
    if rule == "C":
        return np.minimum(z, y) + 0.5 * under + 1.5 * over
    return np.minimum(z, y) + 0.5 * (under + over)


def settlement_q42(p, plan, emergency):
    """4-2费用：继承问题二，仅计划购电费与5倍紧急购电费。"""
    p, plan, emergency = map(lambda x: np.asarray(x, float),
                             (p, plan, emergency))
    return float(np.sum(p * plan + 5.0 * p * emergency))


def settlement_q43(p, plan, adjusted, emergency):
    """4-3费用：问题三口径C；返回总费用与可独立核验的分项。"""
    p, plan, adjusted, emergency = map(
        lambda x: np.asarray(x, float), (p, plan, adjusted, emergency))
    under = np.maximum(plan - adjusted, 0.0)
    over = np.maximum(adjusted - plan, 0.0)
    within = np.minimum(plan, adjusted)
    parts = {
        "within": float(np.sum(p * within)),
        "under": float(np.sum(0.5 * p * under)),
        "over": float(np.sum(1.5 * p * over)),
        "emergency": float(np.sum(5.0 * p * emergency)),
    }
    return {**parts, "total": float(sum(parts.values()))}


def execute_q42_causal(p_rt, load_rt, pv_rt, plan, retention, e0,
                       eta=ETA, emin=EMIN, emax=EMAX, bmax=BMAX):
    """4-2逐段执行：计划冻结，当前缺额经DP保留水位放电后才紧急购电。"""
    p_rt, load_rt, pv_rt, plan, retention = map(
        lambda x: np.asarray(x, float),
        (p_rt, load_rt, pv_rt, plan, retention))
    n = len(plan)
    charge = np.zeros(n); discharge = np.zeros(n)
    emergency = np.zeros(n); surplus = np.zeros(n)
    e_start = np.zeros(n); e_end = np.zeros(n)
    e = float(e0)
    for t in range(n):
        e_start[t] = e
        residual = load_rt[t] - pv_rt[t] - plan[t]
        if residual > 0:
            discharge[t] = min(residual, bmax,
                               eta * max(e - retention[t], 0.0))
            emergency[t] = residual - discharge[t]
        else:
            charge[t] = min(-residual, bmax, (emax - e) / eta)
            surplus[t] = -residual - charge[t]
        e = e + eta * charge[t] - discharge[t] / eta
        e_end[t] = e
    return {"total": settlement_q42(p_rt, plan, emergency),
            "plan_cost": float(np.sum(p_rt * plan)),
            "emergency_cost": float(np.sum(5.0 * p_rt * emergency)),
            "emergency_kwh": float(emergency.sum()),
            "charge": charge, "discharge": discharge,
            "emergency": emergency, "surplus": surplus,
            "energy_start": e_start, "energy_end": e_end, "eT": float(e)}


def _obj_z_coef(rule, p):
    """目标中 z 的系数：口径 C 为 p（计划内照付），口径 B 由 y/u/v 等价表达。"""
    return p if rule == "C" else np.zeros_like(p)


def _terminal_value(P_scen, w, eta=ETA, valley_slots=None):
    """日末库存续存价值 v（元/kWh）：按情景价格的次日凌晨谷价窗口估计。

    问题三日模型用固定常数 V_cont=0.481548（=附件1凌晨电价均值/η）；
    动态电价下必须改为随价格情景浮动，否则日末会恶意放空电池。
    """
    if valley_slots is None:
        valley_slots = slice(0, 30)          # 0:00–5:00
    vals = []
    for m in range(len(w)):
        p = P_scen[m][valley_slots]
        vals.append(w[m] * float(np.mean(p)))
    return float(np.sum(vals)) / eta


# ---------------------------------------------------------------- 单情景模型
def _build_det(p, l, v, e0, rule, z_fixed=None, grid_charge=False, v_term=0.0):
    """单情景（确定性）模型：变量 z,y,u,v,g,c,d,U,E。

    NT = len(p)：支持**变长**时段。4-3 滚动在剩余时段 [k,144) 上重解时传入长度
    为 144-k 的 p/l/v（NT<144）；对全天调用时 NT=144，与模块常量 T 一致，
    行为与既有版本完全相同。
    """
    p = np.asarray(p, float)
    l = np.asarray(l, float)
    v = np.asarray(v, float)
    NT = len(p)
    off = {"z": 0, "y": NT, "u": 2 * NT, "v": 3 * NT, "g": 4 * NT,
           "c": 5 * NT, "d": 6 * NT, "U": 7 * NT, "E": 8 * NT}
    n = 9 * NT + (NT + 1)

    def ix(k, t=0):
        return off[k] + t

    c = np.zeros(n)
    for t in range(NT):
        c[ix("g", t)] = 5.0 * p[t]
        if rule == "C":
            # 口径 C：p·min(z,y)+0.5p·u+1.5p·v
            #         = p·y + 0.5p·u + 0.5p·v   （用 y = z+over−under）
            # 注意：y 的系数必须是 p。漏掉它会让 z=y=0 成为零费用解，
            # 模型静默退化（本 harness 首个版本即因此让目标值恒为 0）。
            c[ix("y", t)] = p[t]
            c[ix("u", t)] = 0.5 * p[t]
            c[ix("v", t)] = 0.5 * p[t]
        else:
            # 口径 B：p·min(z,y)+0.5p·|z−y| = 0.5p·z + 0.5p·u + 0.5p·v
            #         用 y = z+(y−z) 改写为 0.5p·y + 0.5p·(u+v)
            c[ix("y", t)] = 0.5 * p[t]
            c[ix("u", t)] = 0.5 * p[t]
            c[ix("v", t)] = 0.5 * p[t]
    c[ix("E", NT)] = -v_term         # 日末库存续存价值（软终端）

    A_ub = lil_matrix((2 * NT, n))
    b_ub = np.zeros(2 * NT)
    if rule == "C":
        for t in range(NT):
            A_ub[t, ix("y", t)] = -1.0
            A_ub[t, ix("z", t)] = 1.0
            A_ub[t, ix("u", t)] = -1.0
            A_ub[NT + t, ix("y", t)] = 1.0
            A_ub[NT + t, ix("z", t)] = -1.0
            A_ub[NT + t, ix("v", t)] = -1.0

    A_eq = lil_matrix((2 * NT + 1, n))
    b_eq = np.zeros(2 * NT + 1)
    for t in range(NT):
        A_eq[t, ix("y", t)] = 1.0
        A_eq[t, ix("g", t)] = 1.0
        A_eq[t, ix("d", t)] = 1.0
        A_eq[t, ix("c", t)] = -1.0
        A_eq[t, ix("U", t)] = -1.0
        b_eq[t] = l[t] - v[t]
        A_eq[NT + t, ix("E", t + 1)] = 1.0
        A_eq[NT + t, ix("E", t)] = -1.0
        A_eq[NT + t, ix("c", t)] = -ETA
        A_eq[NT + t, ix("d", t)] = 1.0 / ETA
    A_eq[2 * NT, ix("E", 0)] = 1.0
    b_eq[2 * NT] = e0

    bd = [(0, None)] * n
    for t in range(NT):
        cap = BMAX if grid_charge else min(BMAX, max(v[t] - l[t], 0.0))
        bd[ix("c", t)] = (0, cap)
        bd[ix("d", t)] = (0, BMAX)
        bd[ix("E", t)] = (EMIN, EMAX)
    bd[ix("E", NT)] = (EMIN, EMAX)
    if z_fixed is not None:
        for t in range(NT):
            bd[ix("z", t)] = (float(z_fixed[t]), float(z_fixed[t]))
    return c, A_ub.tocsr(), b_ub, A_eq.tocsr(), b_eq, bd, ix


def plan_det(p, l, v, e0, rule="C", z_fixed=None, grid_charge=False,
             v_term=0.0):
    """确定性日前调度。z_fixed=None 时 z 自由。

    同时返回储能执行轨迹 charge/discharge，供 settle_causal 在真实日内逐段执行；
    该轨迹只由传入的（预测）价格与（预测）源荷决定，不含任何未来真实信息。
    支持变长时段（NT=len(p)），4-3 滚动据此在剩余时段上重解。
    """
    NT = len(p)
    c, Aub, bub, Aeq, beq, bd, ix = _build_det(
        p, l, v, e0, rule, z_fixed=z_fixed, grid_charge=grid_charge,
        v_term=v_term)
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bd,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"plan_det infeasible: {res.message}")
    return {"z": res.x[ix("z"):ix("z") + NT],
            "charge": res.x[ix("c"):ix("c") + NT],
            "discharge": res.x[ix("d"):ix("d") + NT],
            "cost_fc": float(res.fun),
            "eT": float(res.x[ix("E", NT)])}


def settle(p_rt, l_rt, v_rt, z, e0, rule="C", grid_charge=False,
           price_signal=None, v_term=0.0):
    """给定承诺 z，按实现价追索执行，返回真实账单明细。

    price_signal: 决策时所用的价格信号（默认=实现价）。传入固定价或预测价即得
                  对应的建模路线；账单仍按实现价 p_rt 结算。
    """
    p_dec = p_rt if price_signal is None else price_signal
    NT = len(p_rt)
    c, Aub, bub, Aeq, beq, bd, ix = _build_det(
        p_dec, l_rt, v_rt, e0, rule, z_fixed=z, grid_charge=grid_charge,
        v_term=v_term)
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bd,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"settle infeasible: {res.message}")
    y = res.x[ix("y"):ix("y") + NT]
    g = res.x[ix("g"):ix("g") + NT]
    U = res.x[ix("U"):ix("U") + NT]
    cc = res.x[ix("c"):ix("c") + NT]
    dd = res.x[ix("d"):ix("d") + NT]
    bill = energy_bill(rule, z, y)
    under, over = np.maximum(z - y, 0), np.maximum(y - z, 0)
    return {
        "total": float(np.sum(p_rt * bill + 5.0 * p_rt * g)),
        "plan_cost": float(np.sum(p_rt * np.minimum(z, y))),
        "under_cost": float(np.sum(0.5 * p_rt * under)),
        "over_cost": float(np.sum(1.5 * p_rt * over if rule == "C"
                                  else 0.5 * p_rt * over)),
        "em_cost": float(np.sum(5.0 * p_rt * g)),
        "em_kwh": float(g.sum()),
        "under_kwh": float(under.sum()),
        "over_kwh": float(over.sum()),
        "unused_kwh": float(U.sum()),
        "charge_kwh": float(cc.sum()),
        "discharge_kwh": float(dd.sum()),
        "em_slots": int((g > 1e-6).sum()),
        "eT": float(res.x[ix("E", NT)]),
        "y": y,
        "g": g,
        "charge": cc,
        "discharge": dd,
        "unused": U,
    }


def execute_track(l, v, e0, c_plan, d_plan, check_tol=1e-9):
    """逐段因果映射（无价格）：给定储能计划轨迹，返回实际购电量与状态。

    这是问题四执行层的唯一物理映射，settle_causal 与 plan_newsboy 共用。
    关键性质：实际购电量 y 由（情景或真实）源荷与固定轨迹唯一确定，与承诺量 z 无关。
    """
    e = float(e0)
    y = np.zeros(T)
    U = np.zeros(T)
    cc = np.zeros(T)
    dd = np.zeros(T)
    for t in range(T):
        c = min(float(max(c_plan[t], 0.0)), (EMAX - e) / ETA, BMAX)
        d = min(float(max(d_plan[t], 0.0)), ETA * (e - EMIN), BMAX)
        if c > 0.0 and d > 0.0:                 # 计划互斥；裁剪后仍同号时保留大者
            if c >= d:
                d = 0.0
            else:
                c = 0.0
        buy = float(l[t]) - float(v[t]) + c - d
        if buy < 0.0:
            U[t] = -buy                         # 不允许售电 → 未利用供能
            buy = 0.0
        y[t], cc[t], dd[t] = buy, c, d
        e = e + ETA * c - d / ETA
        if not (EMIN - check_tol <= e <= EMAX + check_tol):
            raise RuntimeError(f"execute_track: SOC={e:.6f} 越界（t={t}）")
    return {"y": y, "unused": U, "charge": cc, "discharge": dd, "eT": e}


def settle_causal(p_rt, l_rt, v_rt, z, e0, c_plan, d_plan, rule="C"):
    """逐段因果执行并结算：储能按日初计划轨迹推进，偏差由实际购电承担。

    与 `settle` 的区别（这是本 harness 最关键的口径修正）：
      * 执行层只用日初（预测）信息得到的轨迹 c_plan/d_plan，不用未来真实源荷；
      * 逐段推进，每段只用该段已揭示的真实源荷；
      * 价格只进入结算，不进入执行决策——这正是"电价逐段揭示"的直接后果。
    """
    ex = execute_track(l_rt, v_rt, e0, c_plan, d_plan)
    y, U, cc, dd = ex["y"], ex["unused"], ex["charge"], ex["discharge"]
    g = np.zeros(T)                              # 常规购电无上限 → 紧急购电不触发
    under, over = np.maximum(z - y, 0.0), np.maximum(y - z, 0.0)
    bill = energy_bill(rule, z, y)
    return {
        "total": float(np.sum(p_rt * bill + 5.0 * p_rt * g)),
        "plan_cost": float(np.sum(p_rt * np.minimum(z, y))),
        "under_cost": float(np.sum(0.5 * p_rt * under)),
        "over_cost": float(np.sum(1.5 * p_rt * over if rule == "C"
                                  else 0.5 * p_rt * over)),
        "em_cost": float(np.sum(5.0 * p_rt * g)),
        "em_kwh": float(g.sum()),
        "under_kwh": float(under.sum()),
        "over_kwh": float(over.sum()),
        "unused_kwh": float(U.sum()),
        "charge_kwh": float(cc.sum()),
        "discharge_kwh": float(dd.sum()),
        "em_slots": int((g > 1e-6).sum()),
        "eT": float(ex["eT"]),
        "y": y,
        "g": g,
        "charge": cc,
        "discharge": dd,
        "unused": U,
        "causal": True,
    }


def plan_newsboy(p_scen, l_scen, v_scen, w, e0, c_plan, d_plan, rule="C"):
    """执行一致性承诺量：逐段价格加权中位数（报童解）。

    口径 C 账单恒等式（见 energy_bill）：
        K = Σ_t p_t·( y_t + 0.5·|z_t − y_t| )
    而 y 由固定轨迹与情景源荷唯一确定、**与 z 无关**，故目标逐段可分离，
    z_t* = {y_{m,t}} 关于权重 w_m·p_{m,t} 的加权中位数，即建模方案 §7.3 的
    E[P·1_{Y≤z*}] = ½E[P]。

    这与 plan_sp 的差别：plan_sp 的二阶段把 y 当作可随 z 自由调整的追索变量，
    与实际执行层不一致，会把承诺量系统性推高（实测 +18%）。

    口径 B 下账单 = 0.5p(y+z) 对 z 单调递增 → 无内点最优（退化），故只支持 C。
    """
    if rule != "C":
        raise ValueError("plan_newsboy 只支持口径 C：口径 B 下承诺量退化")
    M = len(w)
    Y = np.empty((M, T))
    for m in range(M):
        Y[m] = execute_track(l_scen[m], v_scen[m], e0, c_plan, d_plan)["y"]
    z = np.empty(T)
    exp_cost = 0.0
    for t in range(T):
        yt = Y[:, t]
        wt = w * p_scen[:, t]
        order = np.argsort(yt, kind="stable")
        ys, ws = yt[order], wt[order]
        cw = np.cumsum(ws)
        cw = cw / cw[-1]
        z[t] = ys[min(np.searchsorted(cw, 0.5, side="left"), M - 1)]
        exp_cost += float(np.sum(w * p_scen[:, t] *
                                (yt + 0.5 * np.abs(z[t] - yt))))
    return {"z": z, "cost_fc": exp_cost}


def plan_newsboy_q42(p_scen, l_scen, v_scen, w, e0, c_plan, d_plan,
                     ratio=5.0, quantile=None):
    """4-2 计划量：价格加权**高分位**（缺额惩罚 ratio=5 倍）。

    4-2 账单 = Σ_t p_t z_t + ratio·Σ_t p_t (y_t − z_t)⁺，而 y 由固定轨迹与情景
    源荷唯一确定、与 z 无关（见 execute_track），故目标逐段可分离，一阶条件

        Σ_m w_m p_{m,t}·1{y_{m,t} > z_t} = (1/ratio)·Σ_m w_m p_{m,t}

    即 z_t* 是 {y_{m,t}} 以 w_m p_{m,t} 为权重的 (1 − 1/ratio) 分位；ratio=5 时
    为 80% 分位。缺额惩罚越重，分位越高（应更主动多报）。与 4-3 的对称之处：
    4-3 的差额结算给出 1/2 分位（中位数），4-2 的 5 倍紧急给出 4/5 分位。
    """
    if ratio <= 1.0:
        raise ValueError("ratio 必须 > 1，否则分位无意义")
    M = len(w)
    Y = np.empty((M, T))
    for m in range(M):
        Y[m] = execute_track(l_scen[m], v_scen[m], e0, c_plan, d_plan)["y"]
    z = np.empty(T)
    exp_cost = 0.0
    q = 1.0 - 1.0 / ratio if quantile is None else float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile 必须在 (0,1) 内")
    tail = 1.0 - q
    for t in range(T):
        yt, wt = Y[:, t], w * p_scen[:, t]
        order = np.argsort(-yt, kind="stable")          # 降序，取右尾
        ys, ws = yt[order], wt[order]
        cw = np.cumsum(ws) / ws.sum()
        k = min(int(np.searchsorted(cw, tail, side="left")), M - 1)
        z[t] = ys[k]
        exp_cost += float(np.sum(
            w * p_scen[:, t] * z[t]
            + ratio * p_scen[:, t] * np.maximum(yt - z[t], 0.0)))
    return {"z": z, "cost_fc": exp_cost}


def settle_q42(p_rt, l_rt, v_rt, z, e0, c_plan, d_plan, ratio=5.0,
               price_signal=None, release_quantile=0.6):
    """4-2 结算：计划购电量冻结（按计划量照付），缺额按 ratio 倍紧急购电。

    与 4-3 的区别：**没有少购/超购差额项**（4-2 沿用问题二口径）。
    实际购电量 y 仍由 execute_track 给出，因此两支共享同一执行层物理，
    差别只在结算规则，便于 4-2/4-3 的同口径对照。
    """
    # 正常购电 z 已冻结。计划储能动作按当前 SOC 裁剪；若执行计划充电会
    # 造成供能缺口，先取消当段充电，紧急电只承担仍未覆盖的负荷。
    z = np.asarray(z, float)
    ps = np.asarray(p_rt if price_signal is None else price_signal, float)
    cc = np.zeros(T); dd = np.zeros(T); short = np.zeros(T)
    U = np.zeros(T); soc_path = np.zeros(T)
    e = float(e0)
    for t in range(T):
        c = min(max(float(c_plan[t]), 0.0), BMAX, (EMAX - e) / ETA)
        d = min(max(float(d_plan[t]), 0.0), BMAX, ETA * (e - EMIN))
        if c > 0.0 and d > 0.0:
            if c >= d: d = 0.0
            else: c = 0.0
        gap = float(l_rt[t] - v_rt[t] + c - d - z[t])
        if gap > 0.0 and c > 0.0:
            cut = min(c, gap)
            c -= cut
            gap -= cut
        # 取消计划充电后仍不足，则在功率/SOC余量内追加放电；只有电池无法
        # 继续补偿的部分才是5倍价紧急购电。旧实现漏掉此步，会出现“有电
        # 不放、直接应急”的静默高费用错误。
        # 只在当前预测价达到剩余时域较高分位时动用额外库存；低价缺口保留
        # 电量给后续更昂贵的5倍应急风险。price_signal 是日初可得预测，不前视。
        release = ps[t] >= np.quantile(ps[t:], release_quantile)
        if gap > 0.0 and release:
            extra_d = min(gap, BMAX - d, ETA * (e - EMIN) - d)
            extra_d = max(float(extra_d), 0.0)
            d += extra_d
            gap -= extra_d
        short[t] = max(gap, 0.0)
        U[t] = max(-gap, 0.0)
        cc[t], dd[t] = c, d
        e = e + ETA * c - d / ETA
        soc_path[t] = e
    return {
        "total": float(np.sum(p_rt * z + ratio * p_rt * short)),
        "plan_cost": float(np.sum(p_rt * z)),
        "em_cost": float(np.sum(ratio * p_rt * short)),
        "em_kwh": float(short.sum()),
        "plan_kwh": float(z.sum()),
        "unused_kwh": float(U.sum()),
        "charge_kwh": float(cc.sum()),
        "discharge_kwh": float(dd.sum()),
        "em_slots": int((short > 1e-6).sum()),
        "eT": float(e),
        "y": np.maximum(z + short - U, 0.0),
        "g": short,
        "charge": cc,
        "discharge": dd,
        "unused": U,
        "soc": soc_path,
        "causal": True,
        "rule": "42",
    }


# ------------------------------------------------- 价格化 CPWL-DP 执行器
def _dp_prm() -> dict:
    """CPWL-DP 所需参数（与问题一至三同一套物理常数）。"""
    return {"charge_efficiency": ETA, "discharge_efficiency": ETA,
            "max_charge_discharge_power_kw": PMAX, "interval_hours": DT,
            "operating_min_kwh": EMIN, "operating_max_kwh": EMAX}


def build_h_tilde_q43(p, n, z, eta, S):
    """4-3 单步费用 h(y) = p*[a + 0.5*|z - a|]，a = max(0, n - eta*y)。

    y 为电池内部放电量（>0 放电，<0 充电），n 为净需求，z 为已定计划量。
    口径 C 恒等式下该函数是凸分段线性的，斜率（y 递增）：
      a > z          ->  -1.5*p*eta（超购侧）
      0 < a < z      ->  -0.5*p*eta（少购侧）
      a = 0（不允许售电） -> 0
    这是与问题二 build_h_tilde（缺额 5 倍）的关键差别：用错会把电池放空。
    """
    from q1_two_layer import CPWL
    if n > 0:                       # 缺额：只允许放电（与问题二 build_h_tilde 同构）
        y_lo, y_hi = 0.0, min(S, n) / eta
    else:                           # 富余：只允许充电
        y_lo, y_hi = -eta * min(S, -n), 0.0
    if y_hi - y_lo <= 1e-12:        # 无可用动作，返回常数费用
        a0 = max(n, 0.0)
        return CPWL(y_lo, p * (a0 + 0.5 * abs(z - a0)), [1e-12], [0.0])

    def a_of(y):
        # 外部补偿量：内部放电 Δ 供 ηΔ，内部充电 |Δ| 耗 |Δ|/η（效率不对称）
        comp = eta * y if y >= 0.0 else y / eta
        return max(n - comp, 0.0)

    def hval(y):
        a = a_of(y)
        return p * (a + 0.5 * abs(z - a))

    cand = sorted({(n - z) / eta, n / eta})
    inner = [c for c in cand if y_lo + 1e-9 < c < y_hi - 1e-9]
    bps = np.array([y_lo, *inner, y_hi])
    widths = np.diff(bps)
    mids = 0.5 * (bps[:-1] + bps[1:])
    slopes = []
    for m in mids:
        a = a_of(m)
        if a <= 0.0:
            slopes.append(0.0)
        elif a > z:
            slopes.append(-1.5 * p * eta)
        else:
            slopes.append(-0.5 * p * eta)
    return CPWL(y_lo, hval(y_lo), widths, slopes)


def retention_levels_q43(Hbar, p_rt, eta=ETA, n_rt=None, z=None):
    """4-3 保留水位：边际系数按当期所处购电侧分段。

    当期净需求 n 与计划量 z 决定「现在多补 1 单位电」的边际电价：
      * n > z（超购侧）-> 1.5 p；
      * n <= z（少购侧）-> 0.5 p。
    统一用 0.5 p 会把水位抬得过高、缺额放电不足（`--exec dp` 早期比 traj
    还贵的主因）；n_rt 传 None 时退回旧的统一 0.5 p 行为。
    n_rt 必须是非前视量（情景均值预测），不得使用未来实现值。
    """
    from solve_q2_twolayer import add_linear, argmin_max
    R = np.zeros(T)
    n_arr = None if n_rt is None else np.asarray(n_rt, float)
    z_arr = None if z is None else np.asarray(z, float)
    for t in range(T):
        if n_arr is None or z_arr is None:
            coef = 0.5
        else:
            coef = 1.5 if float(n_arr[t]) > float(z_arr[t]) + 1e-9 else 0.5
        g = add_linear(Hbar[t + 1], coef * float(p_rt[t]) * eta)
        R[t] = argmin_max(g, EMIN, EMAX)
    return R


def price_dp_retention(p_scen, l_scen, v_scen, z, p_rt, v_term=0.0,
                       rule="C", pricing="emergency"):
    """价格化 CPWL-DP：未来斜率用**价格情景**，执行水位用**已观察实现价**。

    复用问题二的凸分段线性框架（`q1_two_layer` 的 CPWL 运算 + `build_h_tilde`），
    只做一处价格化：每个情景的单步缺额费用用**该情景自己的价格**构建，
    而不是共用同一条价格曲线；终端价值用 v_term（由次日凌晨谷价估计）。

    `pricing` 决定执行层用什么代价给「缺额」定价：
      * ``"emergency"``（默认）：执行层的真实风险是**紧急购电 5p**（`a` 一旦
        确认，实际偏差只能按 5 倍补），故单步费用与水位都用 5p —— 与问题二、
        问题三的 DP 价值执行器同源。
      * ``"bill"``：按申报层的口径 C 偏差费（超购 1.5p / 少购 0.5p）定价。
        价值函数过平 → 水位偏高 → 缺额放电不足，实测比 traj 还贵，仅作对照。
    """
    from q1_two_layer import CPWL, add, infconv, restrict
    from solve_q2_twolayer import build_h_tilde, retention_levels
    prm = _dp_prm()
    S = PMAX * DT
    M = len(p_scen)
    r_scen = (np.asarray(l_scen, float) - np.asarray(v_scen, float)) - z[None, :]
    acc: list = [None] * (T + 1)
    for m in range(M):
        H: list = [None] * (T + 1)
        H[T] = CPWL(EMIN, -v_term * EMIN, [EMAX - EMIN], [-v_term])
        for t in range(T - 1, -1, -1):
            if pricing == "bill" and rule == "C":
                h = build_h_tilde_q43(float(p_scen[m, t]),
                                      float(r_scen[m, t]), float(z[t]), ETA, S)
            else:
                h = build_h_tilde(5.0 * float(p_scen[m, t]),
                                  float(r_scen[m, t]), ETA, S)
            g = infconv(h, H[t + 1])
            H[t] = restrict(g, EMIN, EMAX)
            if H[t] is None:
                raise RuntimeError(f"price_dp: 情景 {m} 时段 {t} 定义域为空")
        for t in range(T + 1):
            if acc[t] is None:
                acc[t] = H[t]
            else:
                merged = add(acc[t], H[t])
                if merged is None:
                    raise RuntimeError("price_dp: 价值函数分段相加失败")
                acc[t] = merged
    Hbar = [CPWL(f.lo, f.f_lo / M, f.widths.copy(), f.slopes / M) for f in acc]
    if pricing == "bill" and rule == "C":
        # 非前视：用情景均值（≈点预测）的净需求决定各时段所处的购电侧。
        n_pred = np.mean(np.asarray(l_scen, float) - np.asarray(v_scen, float),
                         axis=0)
        return retention_levels_q43(Hbar, np.asarray(p_rt, float),
                                    n_rt=n_pred, z=np.asarray(z, float))
    return retention_levels(Hbar, np.asarray(p_rt, float), prm)


def _settle_from_path(p_rt, z, y, U, cc, dd, eT, rule):
    g = np.zeros(T)
    under, over = np.maximum(z - y, 0.0), np.maximum(y - z, 0.0)
    bill = energy_bill(rule, z, y)
    return {
        "total": float(np.sum(p_rt * bill + 5.0 * p_rt * g)),
        "plan_cost": float(np.sum(p_rt * np.minimum(z, y))),
        "under_cost": float(np.sum(0.5 * p_rt * under)),
        "over_cost": float(np.sum(1.5 * p_rt * over if rule == "C"
                                  else 0.5 * p_rt * over)),
        "em_cost": float(np.sum(5.0 * p_rt * g)),
        "em_kwh": float(g.sum()),
        "under_kwh": float(under.sum()),
        "over_kwh": float(over.sum()),
        "unused_kwh": float(U.sum()),
        "charge_kwh": float(cc.sum()),
        "discharge_kwh": float(dd.sum()),
        "em_slots": int((g > 1e-6).sum()),
        "eT": float(eT),
        "y": y, "g": g, "charge": cc, "discharge": dd, "unused": U,
    }


def settle_causal_dp(p_rt, l_rt, v_rt, z, e0, R, rule="C"):
    """按价格化 DP 保留水位逐段执行并结算（4-3）。

    与 `settle_q42_dp` 同构：**缺额时**放电（深度受价值水位 R 限制），
    **富余时**才充电；购电量由平衡确定，无紧急项（常规购电无上限，
    缺额以超购形式进入口径 C 的 1.5× 项）。
    旧实现按「e 与 R 的大小」**无条件**追水位——在不需要放电的富余时段
    也放空、在缺额时段反用网电充电，把电池推到 1200/10800，使 Q3_oracle
    路线静默变贵，已按缺额驱动重写。
    """
    R = np.asarray(R, float)
    e = float(e0)
    y = np.zeros(T)
    U = np.zeros(T)
    cc = np.zeros(T)
    dd = np.zeros(T)
    soc_path = np.zeros(T)
    for t in range(T):
        net = float(l_rt[t]) - float(v_rt[t])
        gap = net - float(z[t])
        if gap > 0.0:                    # 计划不足：放电补缺，不越过价值水位
            d = min(gap, ETA * max(e - R[t], 0.0), BMAX)
            c = 0.0
        else:                            # 计划有余：富余充电，剩余弃用
            c = min(-gap, (EMAX - e) / ETA, BMAX)
            d = 0.0
            U[t] = max(-gap - c, 0.0)
        buy = net + c - d
        if buy < 0.0:
            U[t] = max(U[t], -buy)
            buy = 0.0
        y[t], cc[t], dd[t] = buy, c, d
        e = e + ETA * c - d / ETA
        soc_path[t] = e
        if not (EMIN - 1e-9 <= e <= EMAX + 1e-9):
            raise RuntimeError(
                f"settle_causal_dp: SOC={e:.6f} 越界（t={t}）")
    out = _settle_from_path(p_rt, z, y, U, cc, dd, e, rule)
    out["soc"] = soc_path
    out["execution"] = "dp"
    return out


def settle_q42_dp(p_rt, l_rt, v_rt, z, e0, R, ratio=5.0):
    """4-2 执行（价格化 DP 水位）：计划量冻结照付，缺额先放电后紧急购电。

    与 4-3 共用同一套保留水位 R_t（缺额代价同为 5p），差别只在结算规则：
    4-2 按计划量照付、缺额按 ratio 倍计紧急费，没有少购/超购差额项。
    计划有余时多余电量充入储能（不产生费用）。
    """
    R = np.asarray(R, float)
    e = float(e0)
    g = np.zeros(T)
    cc = np.zeros(T)
    dd = np.zeros(T)
    U = np.zeros(T)
    soc_path = np.zeros(T)
    for t in range(T):
        net = float(l_rt[t]) - float(v_rt[t])
        gap = net - float(z[t])
        if gap > 0.0:                       # 计划不足：先放电到水位，再紧急
            d = min(gap, ETA * max(e - R[t], 0.0), BMAX)
            g[t] = gap - d
            c = 0.0
        else:                               # 计划有余：多余电量充电
            c = min(-gap, (EMAX - e) / ETA, BMAX)
            U[t] = max(-gap - c, 0.0)
            d = 0.0
        cc[t], dd[t] = c, d
        e = e + ETA * c - d / ETA
        soc_path[t] = e
        if not (EMIN - 1e-9 <= e <= EMAX + 1e-9):
            raise RuntimeError(f"settle_q42_dp: SOC={e:.6f} 越界（t={t}）")
    return {
        "total": float(np.sum(p_rt * z + ratio * p_rt * g)),
        "plan_cost": float(np.sum(p_rt * z)),
        "em_cost": float(np.sum(ratio * p_rt * g)),
        "em_kwh": float(g.sum()),
        "plan_kwh": float(z.sum()),
        "unused_kwh": float(U.sum()),
        "charge_kwh": float(cc.sum()),
        "discharge_kwh": float(dd.sum()),
        "em_slots": int((g > 1e-6).sum()),
        "eT": float(e),
        "g": g, "charge": cc, "discharge": dd, "unused": U, "soc": soc_path,
        "execution": "dp", "rule": "42",
    }


# ------------------------------------------------------------- 多情景 SP 模型
def _build_sp(p_scen, l_scen, v_scen, w, e0, rule, grid_charge=False,
              v_term=0.0):
    """两阶段 SP：z 全情景共享，其余逐情景追索。

    p_scen/l_scen/v_scen: (M, T)；w: (M,) 情景权重（和为 1）。
    """
    M = len(w)
    per = 8 * T + (T + 1)          # y,u,v,g,c,d,U,E
    n = T + M * per
    off = {"y": 0, "u": T, "v": 2 * T, "g": 3 * T, "c": 4 * T,
           "d": 5 * T, "U": 6 * T, "E": 7 * T}

    def ix(m, k, t=0):
        return T + m * per + off[k] + t

    c = np.zeros(n)
    for m in range(M):
        p = p_scen[m]
        for t in range(T):
            c[ix(m, "g", t)] = w[m] * 5.0 * p[t]
            if rule == "C":
                # 同 _build_det：y 的系数必须为 p，否则零费用退化
                c[ix(m, "y", t)] = w[m] * p[t]
                c[ix(m, "u", t)] = w[m] * 0.5 * p[t]
                c[ix(m, "v", t)] = w[m] * 0.5 * p[t]
            else:
                c[ix(m, "y", t)] = w[m] * 0.5 * p[t]
                c[ix(m, "u", t)] = w[m] * 0.5 * p[t]
                c[ix(m, "v", t)] = w[m] * 0.5 * p[t]
        c[ix(m, "E", T)] = -w[m] * v_term      # 情景日末库存续存价值

    n_ub = (2 * T if rule == "C" else 0) * M
    A_ub = lil_matrix((n_ub, n))
    b_ub = np.zeros(n_ub)
    if rule == "C":
        for m in range(M):
            for t in range(T):
                r1 = m * 2 * T + t
                A_ub[r1, ix(m, "y", t)] = -1.0
                A_ub[r1, t] = 1.0
                A_ub[r1, ix(m, "u", t)] = -1.0
                r2 = m * 2 * T + T + t
                A_ub[r2, ix(m, "y", t)] = 1.0
                A_ub[r2, t] = -1.0
                A_ub[r2, ix(m, "v", t)] = -1.0

    A_eq = lil_matrix((M * (2 * T + 1), n))
    b_eq = np.zeros(M * (2 * T + 1))
    for m in range(M):
        base = m * (2 * T + 1)
        l, v = l_scen[m], v_scen[m]
        for t in range(T):
            A_eq[base + t, ix(m, "y", t)] = 1.0
            A_eq[base + t, ix(m, "g", t)] = 1.0
            A_eq[base + t, ix(m, "d", t)] = 1.0
            A_eq[base + t, ix(m, "c", t)] = -1.0
            A_eq[base + t, ix(m, "U", t)] = -1.0
            b_eq[base + t] = l[t] - v[t]
            A_eq[base + T + t, ix(m, "E", t + 1)] = 1.0
            A_eq[base + T + t, ix(m, "E", t)] = -1.0
            A_eq[base + T + t, ix(m, "c", t)] = -ETA
            A_eq[base + T + t, ix(m, "d", t)] = 1.0 / ETA
        A_eq[base + 2 * T, ix(m, "E", 0)] = 1.0
        b_eq[base + 2 * T] = e0

    bd = [(0, None)] * n
    for m in range(M):
        for t in range(T):
            cap = BMAX if grid_charge else min(
                BMAX, max(v_scen[m][t] - l_scen[m][t], 0.0))
            bd[ix(m, "c", t)] = (0, cap)
            bd[ix(m, "d", t)] = (0, BMAX)
            bd[ix(m, "E", t)] = (EMIN, EMAX)
        bd[ix(m, "E", T)] = (EMIN, EMAX)
    return c, A_ub.tocsr(), b_ub, A_eq.tocsr(), b_eq, bd, ix


def plan_sp(p_scen, l_scen, v_scen, w, e0, rule="C", grid_charge=False,
            v_term=0.0):
    """两阶段随机规划：返回一阶段承诺量 z 与期望费用。"""
    c, Aub, bub, Aeq, beq, bd, ix = _build_sp(
        p_scen, l_scen, v_scen, w, e0, rule, grid_charge=grid_charge,
        v_term=v_term)
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bd,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"plan_sp infeasible: {res.message}")
    return {"z": res.x[:T], "cost_fc": float(res.fun)}


# ------------------------------------------------------------------ 离线下界
def offline_annual(p, l, v, e0=None, grid_charge=False):
    """全年全知离线下界：一次求解 365×144 段 LP，跨日 SOC 连续、无终端约束。

    p/l/v: (D, T)。以 (z=y, g=0) 可行，故这是任意可行因果策略费用的下界。
    返回 (总费用, 每日购电量, 每日末 SOC)。
    """
    D = p.shape[0]
    n_soc = D * T + 1                 # E[0..D*T]，跨日连续的单一序列
    ND = 4 * T                        # 每天：y, c, d, U
    n = D * ND + n_soc

    def iy(d, t):
        return d * ND + t

    def ic(d, t):
        return d * ND + T + t

    def idd(d, t):
        return d * ND + 2 * T + t

    def iU(d, t):
        return d * ND + 3 * T + t

    def iE(d, t):
        # E 是跨日连续的单一序列：第 d 天第 t 个间隔结束时的储电量
        return D * ND + d * T + t

    c = np.zeros(n)
    for d in range(D):
        for t in range(T):
            c[iy(d, t)] = p[d, t]      # 购电费 p·y（无调整费、无紧急费：全知等价）
    A_eq = lil_matrix((D * 2 * T + 1, n))
    b_eq = np.zeros(D * 2 * T + 1)
    for d in range(D):
        for t in range(T):
            # 平衡：y + d = (l − v) + c + U，U 为未利用供能
            r = d * 2 * T + t
            A_eq[r, iy(d, t)] = 1.0
            A_eq[r, idd(d, t)] = 1.0
            A_eq[r, ic(d, t)] = -1.0
            A_eq[r, iU(d, t)] = -1.0
            b_eq[r] = l[d, t] - v[d, t]
            r2 = d * 2 * T + T + t
            A_eq[r2, iE(d, t + 1)] = 1.0
            A_eq[r2, iE(d, t)] = -1.0
            A_eq[r2, ic(d, t)] = -ETA
            A_eq[r2, idd(d, t)] = 1.0 / ETA
    A_eq[D * 2 * T, iE(0, 0)] = 1.0
    b_eq[D * 2 * T] = E0 if e0 is None else e0

    bd = [(0, None)] * n
    for d in range(D):
        for t in range(T):
            bd[ic(d, t)] = (0, BMAX if grid_charge else min(
                BMAX, max(v[d, t] - l[d, t], 0.0)))
            bd[idd(d, t)] = (0, BMAX)
            bd[iE(d, t)] = (EMIN, EMAX)
    bd[iE(D, 0)] = (EMIN, EMAX)

    res = linprog(c, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bd, method="highs")
    if not res.success:
        raise RuntimeError(f"offline annual infeasible: {res.message}")
    buy = np.array([[res.x[iy(d, t)] for t in range(T)] for d in range(D)])
    soc = np.array([[res.x[iE(d, t)] for t in range(T + 1)]
                    for d in range(D)])
    return {"cost": float(np.sum(p * buy)), "buy": buy, "soc": soc}
