"""问题四建模路线对比 harness。

对比筛后的建模路线（公平口径：同一实现价与源荷结算、各自独立跨日 SOC 轨迹、
同一结算口径、严格非前视）：

  Q0  固定价基准   附件1固定分时电价作信号    确定性日前调度 + 因果执行
  Q1  预测调度量承诺 因果价格预测作信号        确定性日前调度 + 因果执行（承诺=预测最优调度量）
  Q2  报童承诺       因果价格预测作信号        执行一致性报童（逐段价格加权中位数）
  Q3_oracle_det 全知价·点预测承诺  当日真实价格作信号  确定性日前调度 + 因果执行
  Q3_oracle_nb  全知价·同一承诺规则  当日真实价格作信号  报童承诺（价格信息上界）
  Q4  离线下界      全年全知（一次 LP，跨日连续）

口径要点
  * 决策只用预测信息；结算一律用实现价与真实源荷；
  * 执行层默认逐段因果（settle_causal）：储能按日初预测轨迹推进，
    不用未来真实源荷；`--foresight` 可切回旧的全知执行作对照；
  * 承诺量默认用执行一致性报童（plan_newsboy）：在因果执行下 y 与 z 无关，
    故 z 逐段取价格加权中位数；`--commit sp` 可切回两阶段 SP 承诺作对照。
    （SP 的二阶段把 y 当作可随 z 调的自由变量，会系统性抬高承诺量，
    实测因果执行下 +18% 承诺量 → 偏差费上升，故不作开发口径。）
  * Q2→Q3_oracle_nb 是同一承诺规则下的纯价格信息价值；
    Q3_oracle_nb→Q3_oracle_det 是全知价格下承诺规则的价值。

用法
  python scripts/harness_q4.py --days 6              # 冒烟
  python scripts/harness_q4.py --start 2025-03-01 --days 60
  python scripts/harness_q4.py                        # 全年
  python scripts/harness_q4.py --resume               # 断点续跑

产物
  results/harness_q4_daily.csv      逐日分项
  results/harness_q4_summary.json   汇总 + 边际贡献
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
from q4_models import (BMAX, DT, E0, EMIN, EMAX, ETA, T, offline_annual,  # noqa: E402
                       plan_det, plan_newsboy, plan_newsboy_q42, plan_sp,
                       price_dp_retention, settle, settle_causal,
                       settle_causal_dp, settle_q42, settle_q42_dp,
                       settlement_q43)
from forecast_q4_price import (day_ahead_price, intraday_update,
                               walk_forward_hybrid_prices)  # noqa: E402
from solve_q3 import _official_current_day  # noqa: E402  （附件3 整点预报→10min 插值）

DATA = ROOT / "data_processed/microgrid_10min.csv"
OFFICIAL_CSV = ROOT / "data_processed/pv_forecast_hourly.csv"
RES = ROOT / "results"
DAILY_CSV = RES / "harness_q4_daily.csv"
SUMMARY_JSON = RES / "harness_q4_summary.json"
SEGMENTS_CSV = RES / "q4_segments.csv"

# 4-3 滚动决策时刻（段号，0-based，对应 0:00/6:00/12:00/18:00），每段 36 个 10min 段
ROLL_ISSUES = (0, 36, 72, 108)
SEG_LEN = 36

LIVE = ("Q0_fixed", "Q1_point", "Q2_nb", "Q3_oracle_det", "Q3_oracle_nb")
LIVE42 = ("Q42_fixed", "Q42_point", "Q42_nb")
PATHS = LIVE + LIVE42 + ("Q4_offline",)
SHORT = {"Q0_fixed": "Q0", "Q1_point": "Q1", "Q2_nb": "Q2",
         "Q3_oracle_det": "Q3d", "Q3_oracle_nb": "Q3n",
         "Q42_fixed": "42F", "Q42_point": "42P", "Q42_nb": "42N"}
BACKTEST_START = "2025-02-01"
BACKTEST_END = "2025-12-31"


# ------------------------------------------------------------------ 价格预测层
def causal_price_forecast(P: np.ndarray, i: int, shape_win: int, lvl_win: int):
    """因果价格预测——**唯一来源**是 scripts/forecast_q4_price.py。

    本函数只是薄封装：把 harness 的命令行参数映射到模块参数。
    历史上 harness 内另写过一份（均值口径）日水平×形状预测，与模块并存会让
    报告数字与模块说明对不上，故统一到模块。
    """
    return day_ahead_price(P, i, shape_window=shape_win,
                           level_window=lvl_win)


def build_scenarios(P, L, V, i, M, win, p_fc, l_fc, v_fc, tod_price,
                    center=True, price_override=None, l_base=None,
                    v_base=None):
    """三维配对残差情景：价格/负荷/光伏残差取自同一历史日，权重等权。

    center=True 时对残差池做零均值化。否则残差池的均值会把情景中心整体推离
    点预测（实测价格情景中心偏低 0.0366 元/kWh），使 SP 的承诺量被系统性抬高。

    price_override 给定后，价格情景全部取该值（用于 Q3_oracle_nb：价格已知，
    但源荷不确定性依旧保留）。此时与 Q2 的唯一差别就是价格信息。
    """
    pool = np.arange(max(0, i - win), i)[-M:]
    ep = P[pool] - tod_price[None, :]            # 价格残差（相对时段均值）
    # 源荷残差必须相对**同一预测基准**：用 Ridge 时以逐日 Ridge 预测为基准，
    # 否则残差池中心（历史均值）与 l_fc 不一致，情景中心会整体偏移，
    # 导致承诺量分位系统性跑偏（实测全年由 −2.16% 反转为 +6.1%）。
    lb = (L[pool].mean(axis=0) if l_base is None
          else np.asarray(l_base, float)[pool])
    vb = (V[pool].mean(axis=0) if v_base is None
          else np.asarray(v_base, float)[pool])
    el = L[pool] - lb
    ev = V[pool] - vb
    if center:
        ep = ep - ep.mean(axis=0, keepdims=True)
        el = el - el.mean(axis=0, keepdims=True)
        ev = ev - ev.mean(axis=0, keepdims=True)
    if price_override is not None:
        p_list = np.tile(np.asarray(price_override, float), (len(pool), 1))
    else:
        p_list = np.maximum(p_fc[None, :] + ep, 1e-3)
    l_list = np.maximum(l_fc[None, :] + el, 0.0)
    v_list = np.maximum(v_fc[None, :] + ev, 0.0)
    w = np.full(len(pool), 1.0 / len(pool))
    return p_list, l_list, v_list, w


def _walk(l, v, e0, c, d):
    """逐段因果执行（长度灵活版）：与 q4_models.execute_track 同一物理映射。

    execute_track 硬编码全天 T=144；本函数按输入长度循环，用于滚动的分段推进
    （每段只推进 36 个 10min 步，SOC 跨段连续）。返回末端 SOC 与逐段明细。
    """
    l = np.asarray(l, float)
    v = np.asarray(v, float)
    c = np.asarray(c, float)
    d = np.asarray(d, float)
    n = len(l)
    e = float(e0)
    y = np.zeros(n)
    U = np.zeros(n)
    cc = np.zeros(n)
    dd = np.zeros(n)
    soc = np.zeros(n)
    for t in range(n):
        ct = min(float(max(c[t], 0.0)), (EMAX - e) / ETA, BMAX)
        dt = min(float(max(d[t], 0.0)), ETA * (e - EMIN), BMAX)
        if ct > 0.0 and dt > 0.0:               # 计划互斥；裁剪后仍同号时保留大者
            if ct >= dt:
                dt = 0.0
            else:
                ct = 0.0
        buy = float(l[t]) - float(v[t]) + ct - dt
        if buy < 0.0:
            U[t] = -buy                         # 不允许售电 → 未利用供能
            buy = 0.0
        y[t], cc[t], dd[t] = buy, ct, dt
        e = e + ETA * ct - dt / ETA
        soc[t] = e
    return e, soc, y, U, cc, dd


# ---------------------------------------------------------------------- 主流程
def run(args) -> None:
    global DAILY_CSV, SUMMARY_JSON, SEGMENTS_CSV
    if getattr(args, "out_tag", None):
        DAILY_CSV = RES / f"_tmp_{args.out_tag}_daily.csv"
        SUMMARY_JSON = RES / f"_tmp_{args.out_tag}_summary.json"
        SEGMENTS_CSV = RES / f"_tmp_{args.out_tag}_segments.csv"
    m = pd.read_csv(DATA, parse_dates=["date"])
    official = pd.read_csv(OFFICIAL_CSV, parse_dates=["issue_at", "target_end"])
    piv = {k: m.pivot(index="date", columns="slot", values=k)
           for k in ("price_dynamic", "price_fixed", "load_kwh", "pv_kwh")}
    dates = piv["price_dynamic"].index
    P = piv["price_dynamic"].values
    FIX = np.tile(piv["price_fixed"].values[0], (len(dates), 1))
    L, V = piv["load_kwh"].values, piv["pv_kwh"].values
    tod_price = P.mean(axis=0)

    lo = dates.get_loc(pd.Timestamp(args.start))
    hi = (dates.get_loc(pd.Timestamp(args.end)) if args.end
          else len(dates) - 1)
    if args.days:
        hi = min(hi, lo + args.days - 1)

    hybrid_prices = None
    if args.price_model == "hybrid":
        hybrid_prices = walk_forward_hybrid_prices(
            P, dates, FIX[0], start_i=lo, retrain_days=args.price_retrain,
            ridge_alpha=args.price_alpha, shape_window=args.shape_win,
            calibration_window=args.price_cal_win)
        if not np.isfinite(hybrid_prices[lo:hi + 1]).all():
            raise RuntimeError("混合价格预测在评价窗口包含非有限值")
        print("[price] hybrid：Ridge水平 + 动态形状融合 + 保序校准", flush=True)

    # 断点续跑
    done = {}
    if args.resume and DAILY_CSV.exists():
        prev = pd.read_csv(DAILY_CSV, parse_dates=["date"])
        for _, r in prev.iterrows():
            done[pd.Timestamp(r["date"])] = r.to_dict()
        print(f"[resume] 已载入 {len(done)} 天", flush=True)

    soc = {k: E0 for k in LIVE + LIVE42}
    gc = not args.no_grid_charge

    # 4-2与4-3源荷接口解耦：问题二升级时只重跑4-2；问题三冻结后再替换4-3。
    def _source_arrays(kind):
        if kind == "mean":
            return None, None
        if kind == "fusion":
            path = ROOT / "data_processed/q2_forecasts_fusion.csv"
            cols = ("load_forecast_kwh", "pv_forecast_kwh")
        elif kind == "q3fusion":
            # 问题三最终接口：分解⊕Ridge 融合负荷 + Ridge 光伏基线。
            # 日内光伏在 4-3 滚动链中仍由附件3最新预报校正替换。
            path = ROOT / "data_processed/q3_forecasts_fused.csv"
            cols = ("lgb_load_kwh", "lgb_pv_kwh")
        else:
            path = ROOT / "data_processed/q2_forecasts_reg.csv"
            cols = ("reg_load_kwh", "reg_pv_kwh")
        reg = pd.read_csv(path, parse_dates=["date"])
        sl = reg.pivot(index="date", columns="slot", values=cols[0]).reindex(dates).values
        sp = reg.pivot(index="date", columns="slot", values=cols[1]).reindex(dates).values
        if np.isnan(sl[lo:hi + 1]).any() or np.isnan(sp[lo:hi + 1]).any():
            raise RuntimeError(f"窗口内{kind}预测缺日：{path.name}")
        # 1 月为预热期（无 Ridge 预测），用列均值占位：残差池回看 30 天会碰到它，
        # 留 NaN 会让 2 月上旬的情景与承诺量变成 NaN。回测窗口自 2/1 起，不影响正式结果。
        def _fill(a):
            cm = np.nanmean(a, axis=0)
            m = np.isnan(a)
            b = np.array(a, dtype=float, copy=True)
            b[m] = np.take(cm, np.nonzero(m)[1])
            return b
        return _fill(sl), _fill(sp)

    src_load, src_pv = _source_arrays(args.source_q43)
    src42_load, src42_pv = _source_arrays(args.source_q42)
    print(f"[source] 4-2={args.source_q42}；4-3={args.source_q43}", flush=True)

    # 问题三式融合光伏（--pv-fused）：偏差校正 + 含交叉项最小二乘动态定权，
    # 与 scripts/ablation_q3_forecast.py:build_library 的正式口径一致（窗口 28 天）。
    # 未给定 --pv-fused 时以下变量为 None，行为与既有正式结果完全一致。
    pv_fuse_anchor = pv_fuse_act = pv_fuse_row = None
    if getattr(args, "pv_fused", False):
        pv_fuse_act = {pd.Timestamp(d): g.sort_values("slot").pv_kwh.to_numpy(float)
                       for d, g in m.groupby("date")}
        pv_fuse_row = {pd.Timestamp(d): j for j, d in enumerate(dates)}
        pv_fuse_anchor = {}
        for d in dates:
            for kk in (0, 6, 12, 18):
                try:
                    pv_fuse_anchor[(pd.Timestamp(d), kk)] = np.asarray(
                        _official_current_day(official, m, d, kk), float)
                except Exception:
                    pass
        print(f"[pv-fused] 预计算附件3锚点 {len(pv_fuse_anchor)} 个(日,时刻)", flush=True)

    def execute(p_rt, l_rt, v_rt, z, e0, track=None, price_signal=None,
                v_term=0.0, scen=None):
        """4-3 执行入口：价格化 CPWL-DP（默认）／日初轨迹／全知（对照）。"""
        if args.foresight:
            return settle(p_rt, l_rt, v_rt, z, e0, rule=args.rule,
                          grid_charge=gc, price_signal=price_signal,
                          v_term=v_term)
        if args.exec == "dp" and scen is not None:
            R = price_dp_retention(scen[0], scen[1], scen[2], z, p_rt, v_term)
            return settle_causal_dp(p_rt, l_rt, v_rt, z, e0, R, rule=args.rule)
        return settle_causal(p_rt, l_rt, v_rt, z, e0, track["charge"],
                             track["discharge"], rule=args.rule)

    def execute42(p_rt, l_rt, v_rt, z, e0, track, scen, v_term):
        """4-2 执行入口：价格化 CPWL-DP（默认）／日初轨迹。"""
        if args.exec == "dp":
            R = price_dp_retention(scen[0], scen[1], scen[2], z, p_rt, v_term)
            return settle_q42_dp(p_rt, l_rt, v_rt, z, e0, R)
        p_sig = np.average(scen[0], axis=0, weights=scen[3])
        return settle_q42(p_rt, l_rt, v_rt, z, e0, track["charge"],
                          track["discharge"], price_signal=p_sig,
                          release_quantile=args.release_quantile)

    # ------------------------------------------------ 4-3 四时刻滚动执行链
    def _fused_pv(day, i, issue_hour):
        """问题三式融合光伏：历史偏差校正 + 含交叉项最小二乘滚动权重（窗口 28）。"""
        k = issue_hour * 6
        raw = pv_fuse_anchor.get((pd.Timestamp(day), issue_hour))
        if raw is None or len(raw) != T - k:
            return None
        prior = [d for d in dates if d < day][-28:]
        prior = [d for d in prior
                 if (pd.Timestamp(d), issue_hour) in pv_fuse_anchor]
        if not prior:
            return None
        off = np.vstack([pv_fuse_anchor[(pd.Timestamp(d), issue_hour)]
                         - pv_fuse_act[d][k:] for d in prior])
        bias = np.median(off, axis=0)
        reg = np.vstack([np.asarray(src_pv[pv_fuse_row[d]], float)[k:]
                         - pv_fuse_act[d][k:] for d in prior])
        eA, eR = off - bias, reg
        den = float(np.sum((eA - eR) ** 2))
        w = (0.5 if den <= 0
             else float(np.clip(np.sum(eR * (eR - eA)) / den, 0.0, 1.0)))
        return np.maximum(0.0, w * np.maximum(0.0, raw - bias)
                          + (1.0 - w) * np.asarray(src_pv[i], float)[k:])

    def _pv_rem(day, i, issue_hour, v_fc0):
        """附件3 在 issue_hour 发布的当日**剩余段**光伏预报（kWh/10min）。

        对齐 harness 索引 [k, 144)（k = issue_hour*6）：返回数组第 j 个元素即
        原始 144 段中第 k+j 段。插值口径沿用 solve_q3._official_current_day
        （同发布时刻内，在 issue 时刻值与 24 个小时末锚点之间线性插值）。
        缺失时回退到当日非前视 Ridge 光伏预测对应段。
        """
        k = issue_hour * 6
        if getattr(args, "pv_fused", False) and src_pv is not None:
            fused = _fused_pv(day, i, issue_hour)
            if fused is not None:
                return fused
        try:
            out = np.asarray(_official_current_day(official, m, day, issue_hour),
                             float)
            if len(out) != T - k:
                raise ValueError(f"长度 {len(out)} ≠ {T - k}")
            if (getattr(args, "pv_blend", False) and issue_hour == 0
                    and src_pv is not None):
                # 0:00 融合：0.389×附件3 + 0.611×周期历史(Ridge)，与问题三的
                # 0 时组合预报同构，抑制单次发布的跳动（调增的主要来源）。
                out = (0.388815 * out
                       + 0.611185 * np.asarray(src_pv[i], float)[k:])
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"  [roll] {day.date()} issue={issue_hour}h 附件3 缺失"
                  f"（{exc}），回退 Ridge", flush=True)
            src = src_pv[i] if src_pv is not None else np.asarray(v_fc0, float)
            return np.asarray(src, float)[k:]

    def roll_q43(day, i, e0, p_fc0, l_fc0, v_fc0, p_rt, l_rt, v_rt,
                 commitment="nb", update_price=True):
        """4-3 主路线（Q2_nb）四时刻滚动重解 + 因果执行 + 口径 C 结算。

        时刻 k∈{0,36,72,108}（0:00/6:00/12:00/18:00）重解**剩余时段 [k,144)**
        的光伏/价格预测与储能轨迹，只用当时可得信息：

          * 光伏：附件3 issue=k 的整点预报经 _pv_rem 插值到 10min（issue<=k）；
          * 价格：day_ahead_price 全日基准 + intraday_update(仅用 p_rt[:k])；
          * 负荷：非前视 Ridge 预测 l_fc0；
          * 储能量：上一段执行完毕后的实际 SOC（跨段连续）。

        已执行时段 [0,k) 冻结：其储能动作来自更早时刻的重解。
        x0（承诺量）固定为 0:00 求解值；口径 C 的差额只相对 x0 结算一次；
        a_t 为各段所属（最后一次）重解确认的购电量，由拼接轨迹因果执行给出。
        """
        # --- k=0：承诺量 x0（附件3 issue=0 光伏 + 价格预测情景 → 报童）---
        v0 = _pv_rem(day, i, 0, v_fc0)
        Ps0, Ls0, Vs0, w0 = build_scenarios(
            P, L, V, i, args.scen, args.win, p_fc0, l_fc0, v0, tod_price,
            l_base=src_load, v_base=src_pv)
        b0 = plan_det(p_fc0, l_fc0, v0, e0, grid_charge=gc, v_term=v_term)
        if commitment == "nb":
            x0 = plan_newsboy(Ps0, Ls0, Vs0, w0, e0, b0["charge"],
                              b0["discharge"])["z"]
        elif commitment == "nb5":
            # 严格解读（窗口内不可追加）下的正确承诺量：窗口内缺口的边际代价是
            # 5p（紧急购电），故承诺量应取价格加权 4/5 分位，与 4-2 报童解同构。
            # 确定性 LP 隐含「计划能精确匹配需求」，会低估峰段订量。
            x0 = plan_newsboy_q42(Ps0, Ls0, Vs0, w0, e0, b0["charge"],
                                  b0["discharge"], quantile=0.8)["z"]
        elif commitment == "sp":
            # 两阶段情景 SP：为不确定性留余地，使后续重解以退订（0.5×）
            # 而非追加（1.5×）为主。
            x0 = np.asarray(plan_sp(Ps0, Ls0, Vs0, w0, e0, rule=args.rule,
                                    grid_charge=gc, v_term=v_term)["z"], float)
        else:
            x0 = np.asarray(b0["z"], float).copy()
        # 价格化 CPWL-DP 保留水位（--exec dp）：执行层据此决定「缺额时放多少」，
        # 与问题二/问题三的 DP 价值执行器同源；traj 口径保持原行为。
        # --greedy-discharge 时水位恒为物理下限，无需再算 R（省一次全天递推）。
        dj = (args.exec == "dp" and not args.greedy_discharge)
        Rdp = (price_dp_retention(Ps0, Ls0, Vs0, x0, np.asarray(p_rt, float),
                                  v_term=v_term, rule=args.rule)
               if dj else None)
        a_all = np.zeros(T)
        c_all = np.zeros(T)
        d_all = np.zeros(T)
        g_all = np.zeros(T)
        u_all = np.zeros(T)
        soc_all = np.zeros(T)
        # 0:00后第一段尚无调整，确认量就是原计划。
        a_all[:SEG_LEN] = x0[:SEG_LEN]
        c_all[0:SEG_LEN] = b0["charge"][0:SEG_LEN]
        d_all[0:SEG_LEN] = b0["discharge"][0:SEG_LEN]

        def execute_confirmed(s, stop, e, a, cplan, dplan):
            """执行已确认购电量；源荷偏差只能变成紧急购电或富余，不能改写a。"""
            for t in range(s, stop):
                c = min(max(float(cplan[t - s]), 0.0), BMAX, (EMAX - e) / ETA)
                d = min(max(float(dplan[t - s]), 0.0), BMAX, ETA * (e - EMIN))
                if c > 0 and d > 0:
                    if c >= d: d = 0.0
                    else: c = 0.0
                gap = float(l_rt[t] - v_rt[t] + c - d - a[t])
                # 紧急电只保供、不用于储能：若短缺与计划充电并存，先因果取消
                # 当段充电，再对仍未覆盖的负荷启动紧急购电。
                if gap > 0.0 and c > 0.0:
                    cut = min(c, gap)
                    c -= cut
                    gap -= cut
                if gap > 0.0 and (Rdp is not None or args.greedy_discharge):
                    # 放电补缺的深度上限：DP 口径受保留水位 R 限制（问题二/三
                    # 同源），--greedy-discharge 时为物理下限 EMIN。
                    floor = (EMIN if (Rdp is None or args.greedy_discharge)
                             else float(Rdp[t]))
                    # 可额外放电量 = 库存余量 - 已安排的计划放电量 d（否则会
                    # 重复计数、把 SOC 压到下限以下）。
                    extra = min(gap, BMAX - d,
                                max(ETA * (e - floor) - d, 0.0))
                    d += extra
                    gap -= extra
                if args.intra_adj and gap > 0.0:
                    # 窗口内追加：把执行前最后确认值上调，缺口转为超购（1.5×）
                    # 而非 5× 紧急（反事实对照开关，默认关闭，非本文口径）。
                    a_all[t] += gap
                    gap = 0.0
                g_all[t] = max(gap, 0.0)
                u_all[t] = max(-gap, 0.0)
                c_all[t], d_all[t] = c, d
                e = e + ETA * c - d / ETA
                soc_all[t] = e
            return e

        e = execute_confirmed(0, SEG_LEN, e0, a_all,
                              b0["charge"][:SEG_LEN], b0["discharge"][:SEG_LEN])
        # --- k=36/72/108：用更新信息重解剩余段，执行 [k,k+36) ---
        for k in ROLL_ISSUES[1:]:
            vk = _pv_rem(day, i, k // 6, v_fc0)
            p_pred = (intraday_update(p_fc0, p_rt, k) if update_price
                      else np.asarray(p_fc0, float))
            bk = plan_det(p_pred[k:], l_fc0[k:], vk, e, grid_charge=gc,
                          v_term=v_term)
            a_all[k:k + SEG_LEN] = bk["z"][:SEG_LEN]
            e = execute_confirmed(k, k + SEG_LEN, e, a_all,
                                  bk["charge"][:SEG_LEN], bk["discharge"][:SEG_LEN])
        parts = settlement_q43(p_rt, x0, a_all, g_all)
        under = np.maximum(x0 - a_all, 0.0)
        over = np.maximum(a_all - x0, 0.0)
        r = {"total": parts["total"], "plan_cost": parts["within"],
             "under_cost": parts["under"], "over_cost": parts["over"],
             "em_cost": parts["emergency"], "em_kwh": float(g_all.sum()),
             "under_kwh": float(under.sum()), "over_kwh": float(over.sum()),
             "unused_kwh": float(u_all.sum()), "charge_kwh": float(c_all.sum()),
             "discharge_kwh": float(d_all.sum()), "em_slots": int((g_all>1e-6).sum()),
             "eT": float(e), "y": a_all, "g": g_all, "charge": c_all,
             "discharge": d_all, "unused": u_all, "soc": soc_all,
             "adjusted": a_all, "execution": "rolling_confirmed"}
        return x0, c_all, d_all, r

    rows = [done[d] for d in sorted(done)] if done else []
    seg_rows: list = []
    seg_diag = {"soc_min": np.inf, "soc_max": -np.inf, "bal": 0.0,
                "socrec": 0.0}
    t0 = time.perf_counter()

    for i in range(lo, hi + 1):
        day = dates[i]
        if day in done:
            for k in LIVE + LIVE42:
                soc[k] = float(done[day][f"{k}_eT"])
            continue
        idx = np.arange(max(0, i - args.win), i)
        p_fc = (hybrid_prices[i] if hybrid_prices is not None else
                causal_price_forecast(P, i, args.shape_win, args.level_win))
        if src_load is not None:
            l_fc, v_fc = src_load[i], src_pv[i]
        else:
            l_fc = L[idx].mean(axis=0)
            v_fc = V[idx].mean(axis=0)
        if src42_load is not None:
            l42_fc, v42_fc = src42_load[i], src42_pv[i]
        else:
            l42_fc, v42_fc = L[idx].mean(axis=0), V[idx].mean(axis=0)
        p_rt, l_rt, v_rt = P[i], L[i], V[i]

        # 日末库存续存价值：用因果预测的凌晨谷价窗口估计（问题三用固定 0.481548）
        v_term = (0.0 if args.no_vterm else
                  args.vterm_mult * float(np.mean(p_fc[:30])) / ETA)

        # 三维配对残差情景（4-3 与 4-2 共用）；Q3 支把价格换成当日真实价
        P_s, L_s, V_s, w = build_scenarios(
            P, L, V, i, args.scen, args.win, p_fc, l_fc, v_fc, tod_price,
            l_base=src_load, v_base=src_pv)
        P_s42, L_s42, V_s42, w42 = build_scenarios(
            P, L, V, i, args.scen, args.win, p_fc, l42_fc, v42_fc, tod_price,
            l_base=src42_load, v_base=src42_pv)
        P_o, L_o, V_o, wo = build_scenarios(
            P, L, V, i, args.scen, args.win, p_fc, l_fc, v_fc, tod_price,
            price_override=p_rt, l_base=src_load, v_base=src_pv)
        sc_fix = (np.tile(FIX[i], (len(w), 1)), L_s, V_s, w)
        sc_fc = (P_s, L_s, V_s, w)
        sc42_fix = (np.tile(FIX[i], (len(w42), 1)), L_s42, V_s42, w42)
        sc42_fc = (P_s42, L_s42, V_s42, w42)
        sc_rt = (P_o, L_o, V_o, wo)

        rec = {"date": day}
        costs = {}

        # Q0/Q1/Q2 必须使用完全相同的四时刻滚动结构；否则“价格信息/承诺量
        # 贡献”会混入是否滚动的结构差异，不能解释为单因素效应。
        if args.roll and not args.foresight:
            z0, c0, d0, r0 = roll_q43(
                day, i, soc["Q0_fixed"], FIX[i], l_fc, v_fc,
                p_rt, l_rt, v_rt, commitment=args.q43_commit,
                update_price=False)
            p0 = {"z": z0, "charge": c0, "discharge": d0}
            z1, c1, d1, r1 = roll_q43(
                day, i, soc["Q1_point"], p_fc, l_fc, v_fc,
                p_rt, l_rt, v_rt, commitment="point", update_price=True)
            p1 = {"z": z1, "charge": c1, "discharge": d1}
        else:
            p0 = plan_det(FIX[i], l_fc, v_fc, soc["Q0_fixed"],
                          grid_charge=gc, v_term=v_term)
            r0 = execute(p_rt, l_rt, v_rt, p0["z"], soc["Q0_fixed"], p0,
                         price_signal=FIX[i], v_term=v_term, scen=sc_fix)
            p1 = plan_det(p_fc, l_fc, v_fc, soc["Q1_point"],
                          grid_charge=gc, v_term=v_term)
            r1 = execute(p_rt, l_rt, v_rt, p1["z"], soc["Q1_point"], p1,
                         price_signal=p_fc, v_term=v_term, scen=sc_fc)

        # ---- Q2_nb 消融路线：因果预测价 + 加权中位数；与正式 Q0 同构滚动 ----
        q2_e0 = soc["Q2_nb"]
        if args.roll and not args.foresight:
            z2, _c_roll, _d_roll, r2 = roll_q43(
                day, i, q2_e0, p_fc, l_fc, v_fc, p_rt, l_rt, v_rt,
                commitment="nb", update_price=True)
            b2 = {"charge": _c_roll, "discharge": _d_roll}
        else:
            b2 = plan_det(p_fc, l_fc, v_fc, q2_e0, grid_charge=gc,
                          v_term=v_term)
            if args.commit == "nb":
                z2 = plan_newsboy(P_s, L_s, V_s, w, q2_e0, b2["charge"],
                                  b2["discharge"])["z"]
            else:
                z2 = plan_sp(P_s, L_s, V_s, w, q2_e0, grid_charge=gc,
                             v_term=v_term)["z"]
                b2 = plan_det(p_fc, l_fc, v_fc, q2_e0, z_fixed=z2,
                              grid_charge=gc, v_term=v_term)
            r2 = execute(p_rt, l_rt, v_rt, z2, q2_e0, b2,
                         price_signal=p_fc, v_term=v_term, scen=sc_fc)

        # ---- Q3_oracle_det：当日真实价格 + 点预测承诺 ----
        p3d = plan_det(p_rt, l_fc, v_fc, soc["Q3_oracle_det"],
                       grid_charge=gc, v_term=v_term)
        r3d = execute(p_rt, l_rt, v_rt, p3d["z"], soc["Q3_oracle_det"], p3d,
                      price_signal=p_rt, v_term=v_term, scen=sc_rt)

        # ---- Q3_oracle_nb：当日真实价格 + 同一承诺规则（价格信息上界）----
        b3n = plan_det(p_rt, l_fc, v_fc, soc["Q3_oracle_nb"], grid_charge=gc,
                       v_term=v_term)
        if args.commit == "nb":
            z3n = plan_newsboy(P_o, L_o, V_o, wo, soc["Q3_oracle_nb"],
                               b3n["charge"], b3n["discharge"])["z"]
        else:
            z3n = plan_sp(P_o, L_o, V_o, wo, soc["Q3_oracle_nb"],
                          grid_charge=gc, v_term=v_term)["z"]
            b3n = plan_det(p_rt, l_fc, v_fc, soc["Q3_oracle_nb"], z_fixed=z3n,
                           grid_charge=gc, v_term=v_term)
        r3n = execute(p_rt, l_rt, v_rt, z3n, soc["Q3_oracle_nb"], b3n,
                      price_signal=p_rt, v_term=v_term, scen=sc_rt)

        # ============ 4-2 分支：计划量冻结 + 5×紧急购电（无差额项）============
        p42f = plan_det(FIX[i], l42_fc, v42_fc, soc["Q42_fixed"],
                        grid_charge=gc, v_term=v_term)
        r42f = execute42(p_rt, l_rt, v_rt, p42f["z"], soc["Q42_fixed"],
                         p42f, sc42_fix, v_term)
        b42 = plan_det(p_fc, l42_fc, v42_fc, soc["Q42_point"],
                       grid_charge=gc, v_term=v_term)
        r42p = execute42(p_rt, l_rt, v_rt, b42["z"], soc["Q42_point"],
                         b42, sc42_fc, v_term)
        b42n = plan_det(p_fc, l42_fc, v42_fc, soc["Q42_nb"],
                        grid_charge=gc, v_term=v_term)
        if args.commit == "sp":
            # 4-2 计划量改用两阶段情景 SP（对照：问题二的计划量规则）。
            z42n = np.asarray(plan_sp(P_s42, L_s42, V_s42, w42, soc["Q42_nb"],
                                      rule=args.rule, grid_charge=gc,
                                      v_term=v_term)["z"], float)
        else:
            z42n = plan_newsboy_q42(P_s42, L_s42, V_s42, w42, soc["Q42_nb"],
                                    b42n["charge"], b42n["discharge"],
                                    quantile=args.q42_quantile)["z"]
        q42_e0 = soc["Q42_nb"]
        r42n = execute42(p_rt, l_rt, v_rt, z42n, soc["Q42_nb"],
                         b42n, sc42_fc, v_term)

        for tag, r, z in (("Q0_fixed", r0, p0["z"]), ("Q1_point", r1, p1["z"]),
                          ("Q2_nb", r2, z2),
                          ("Q3_oracle_det", r3d, p3d["z"]),
                          ("Q3_oracle_nb", r3n, z3n),
                          ("Q42_fixed", r42f, p42f["z"]),
                          ("Q42_point", r42p, b42["z"]),
                          ("Q42_nb", r42n, z42n)):
            costs[tag] = r["total"]
            rec[f"{tag}_total"] = r["total"]
            rec[f"{tag}_plan"] = r["plan_cost"]
            rec[f"{tag}_under"] = r.get("under_cost", 0.0)
            rec[f"{tag}_over"] = r.get("over_cost", 0.0)
            rec[f"{tag}_em_cost"] = r["em_cost"]
            rec[f"{tag}_em_kwh"] = r["em_kwh"]
            rec[f"{tag}_under_kwh"] = r.get("under_kwh", 0.0)
            rec[f"{tag}_over_kwh"] = r.get("over_kwh", 0.0)
            rec[f"{tag}_unused_kwh"] = r["unused_kwh"]
            rec[f"{tag}_charge_kwh"] = r["charge_kwh"]
            rec[f"{tag}_discharge_kwh"] = r["discharge_kwh"]
            rec[f"{tag}_em_slots"] = r["em_slots"]
            rec[f"{tag}_z_kwh"] = float(np.sum(z))
            rec[f"{tag}_eT"] = r["eT"]
            soc[tag] = r["eT"]

        # ---- 正式落盘：4-3 取样本外最优的 Q0 稳健价格信号；4-2 取 Q42_nb ----
        if args.export_segments:
            official43, z43 = r0, p0["z"]
            # 本日循环开头的 Q0 SOC 已被 r0 覆盖前保存在其逐段轨迹中；
            # 用首段递推反推，避免另设可被更新顺序污染的状态变量。
            soc43 = np.asarray(official43.get("soc"), float)
            y43 = np.asarray(official43.get("adjusted", official43["y"]), float)
            g43 = np.asarray(official43.get("g", np.zeros(T)), float)
            U43 = np.asarray(official43["unused"], float)
            cc43 = np.asarray(official43["charge"], float)
            dd43 = np.asarray(official43["discharge"], float)
            q43_e0 = float(soc43[0] - ETA * cc43[0] + dd43[0] / ETA)
            soc42 = np.asarray(r42n["soc"], float)
            y42 = np.asarray(z42n, float)
            U42 = np.asarray(r42n["unused"], float)
            cc42 = np.asarray(r42n["charge"], float)
            dd42 = np.asarray(r42n["discharge"], float)
            g42 = np.asarray(r42n["g"], float)
            # 平衡：确认购电+紧急+光伏+放电 = 负荷+充电+富余。
            bal = y43 + g43 + v_rt + dd43 - l_rt - cc43 - U43
            socrec = np.diff(np.r_[q43_e0, soc43]) - (ETA * cc43 - dd43 / ETA)
            seg_diag["bal"] = max(seg_diag["bal"], float(np.abs(bal).max()))
            seg_diag["socrec"] = max(seg_diag["socrec"],
                                     float(np.abs(socrec).max()))
            seg_diag["soc_min"] = min(seg_diag["soc_min"], float(soc43.min()))
            seg_diag["soc_max"] = max(seg_diag["soc_max"], float(soc43.max()))
            for t in range(T):
                seg_rows.append({
                    "date": day, "slot": t + 1, "variant": "4-3",
                    "price": float(p_rt[t]), "load_kwh": float(l_rt[t]),
                    "pv_kwh": float(v_rt[t]), "plan_kwh": float(z43[t]),
                    "adjusted_kwh": float(y43[t]),
                    "charge_kwh": float(cc43[t]),
                    "discharge_kwh": float(dd43[t]),
                    "emergency_kwh": float(g43[t]), "unused_kwh": float(U43[t]),
                    "soc_end_kwh": float(soc43[t])})
                seg_rows.append({
                    "date": day, "slot": t + 1, "variant": "4-2",
                    "price": float(p_rt[t]), "load_kwh": float(l_rt[t]),
                    "pv_kwh": float(v_rt[t]), "plan_kwh": float(z42n[t]),
                    "adjusted_kwh": float(z42n[t]),
                    "charge_kwh": float(cc42[t]),
                    "discharge_kwh": float(dd42[t]),
                    "emergency_kwh": float(g42[t]),
                    "unused_kwh": float(U42[t]),
                    "soc_end_kwh": float(soc42[t])})

        # ---- Q4 的离线下界在窗口跑完后一次性计算（见 compute_offline）----
        rec["Q4_offline_total"] = np.nan

        rows.append(rec)
        if len(rows) % args.log_every == 0:
            el = time.perf_counter() - t0
            d = pd.DataFrame(rows)
            print(f"  [{len(rows):3d}] {day.date()} " + "  ".join(
                f"{SHORT[k]}={d[f'{k}_total'].sum()/1e4:.1f}万"
                for k in LIVE + LIVE42) + f"   用时 {el:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(DAILY_CSV, index=False,
                                      encoding="utf-8-sig")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig")

    # 离线下界：整个窗口一次 LP，跨日 SOC 连续，无终端约束（真下界）
    o0, o1 = dates.get_loc(pd.Timestamp(df.date.min())), \
        dates.get_loc(pd.Timestamp(df.date.max()))
    offline_cost = np.nan
    if args.no_offline:
        cached = RES / "q4_offline_bound.json"
        if cached.exists():
            try:
                offline_cost = float(json.loads(cached.read_text(
                    encoding="utf-8"))["cost"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    else:
        print("\n计算离线下界（整窗口一次 LP）...", flush=True)
        try:
            off = offline_annual(P[o0:o1 + 1], L[o0:o1 + 1], V[o0:o1 + 1],
                                 grid_charge=not args.no_grid_charge)
            offline_cost = off["cost"]
            print(f"  离线下界 = {offline_cost:,.0f} 元", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  离线 LP 失败（跳过）：{exc}", flush=True)
    summarize(df, args, offline_cost)

    if args.export_segments:
        seg_df = pd.DataFrame(seg_rows, columns=[
            "date", "slot", "variant", "price", "load_kwh", "pv_kwh",
            "plan_kwh", "adjusted_kwh", "charge_kwh", "discharge_kwh",
            "emergency_kwh", "unused_kwh", "soc_end_kwh"])
        seg_df.to_csv(SEGMENTS_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[segments] 已写出 {SEGMENTS_CSV.name}（{len(seg_df)} 行，"
              f"= {len(df)} 天 × 144 段 × 2 模型）")
        if seg_rows:
            print(f"[segments] 4-3 SOC ∈ [{seg_diag['soc_min']:.3f}, "
                  f"{seg_diag['soc_max']:.3f}] kWh；"
                  f"能量平衡残差 max={seg_diag['bal']:.2e}，"
                  f"SOC 递推残差 max={seg_diag['socrec']:.2e} kWh")


def summarize(df: pd.DataFrame, args, offline_cost: float = float("nan")) -> None:
    n = len(df)
    tot = {k: float(df[f"{k}_total"].sum()) for k in LIVE}
    tot42 = {k: float(df[f"{k}_total"].sum()) for k in LIVE42}
    base_q1 = tot["Q1_point"]

    def cvar(x, q=0.95):
        v = np.sort(np.asarray(x, float))
        k = max(1, int(np.ceil(q * len(v))))
        return float(v[-k:].mean())

    summary = {
        "days": n,
        "window": [str(df.date.min().date()), str(df.date.max().date())],
        "scenarios": args.scen, "residual_window": args.win,
        "shape_window": args.shape_win, "level_window": args.level_win,
        "price_model": args.price_model,
        "source_q42": args.source_q42,
        "source_q43": args.source_q43,
        "q42_quantile": args.q42_quantile,
        "vterm_mult": args.vterm_mult,
        "release_quantile": args.release_quantile,
        "rule": args.rule, "grid_charge": not args.no_grid_charge,
        "execution": "foresight" if args.foresight else "causal",
        "commitment": args.commit,
        "total_yuan": {**tot, "Q4_offline": offline_cost},
        "total_yuan_42": tot42,
        "emergency_kwh_42": {k: float(df[f"{k}_em_kwh"].sum())
                              for k in LIVE42},
        "cvar95_daily_42": {k: cvar(df[f"{k}_total"]) for k in LIVE42},
        "worst_day_42": {k: float(df[f"{k}_total"].max()) for k in LIVE42},
        "terminal_soc_42": {k: float(df[f"{k}_eT"].iloc[-1]) for k in LIVE42},
        "effects_42_yuan": {
            "price_signal_Q42fixed_to_Q42point":
                tot42["Q42_point"] - tot42["Q42_fixed"],
            "commitment_Q42point_to_Q42nb":
                tot42["Q42_nb"] - tot42["Q42_point"],
        },
        "wan_yuan": {k: v / 1e4
                     for k, v in {**tot, "Q4_offline": offline_cost}.items()},
        "emergency_kwh": {k: float(df[f"{k}_em_kwh"].sum())
                          for k in LIVE},
        "under_kwh": {k: float(df[f"{k}_under_kwh"].sum())
                      for k in LIVE},
        "over_kwh": {k: float(df[f"{k}_over_kwh"].sum())
                     for k in LIVE},
        "unused_kwh": {k: float(df[f"{k}_unused_kwh"].sum())
                       for k in LIVE},
        "cvar95_daily": {k: cvar(df[f"{k}_total"]) for k in LIVE},
        "worst_day": {k: float(df[f"{k}_total"].max()) for k in LIVE},
        "terminal_soc": {k: float(df[f"{k}_eT"].iloc[-1]) for k in LIVE},
        "effects_yuan": {
            "price_signal_Q0_to_Q1": tot["Q1_point"] - tot["Q0_fixed"],
            "commitment_Q1_to_Q2": tot["Q2_nb"] - tot["Q1_point"],
            "oracle_price_Q2_to_Q3n": tot["Q3_oracle_nb"] - tot["Q2_nb"],
            "commitment_under_oracle_Q3n_to_Q3det":
                tot["Q3_oracle_det"] - tot["Q3_oracle_nb"],
            "regret_Q2_vs_offline": tot["Q2_nb"] - offline_cost,
            "regret_Q1_vs_offline": tot["Q1_point"] - offline_cost,
            "regret_Q0_vs_offline": tot["Q0_fixed"] - offline_cost,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\n{'='*72}\n问题四建模路线对比（{n} 天，{summary['window'][0]}"
          f"~{summary['window'][1]}，M={args.scen}，口径{args.rule}，"
          f"执行={summary['execution']}）\n{'='*72}")
    hdr = f"{'路线':<16}{'总费用/万元':>14}{'相对Q1':>14}{'紧急/万kWh':>13}"
    print(hdr + f"{'CVaR95/元':>13}{'最坏日/元':>12}{'末态SOC':>11}")
    for k in LIVE:
        diff = tot[k] - base_q1
        print(f"{k:<14}{tot[k]/1e4:>14.2f}{diff:>+14,.0f}"
              f"{df[f'{k}_em_kwh'].sum()/1e4:>13.2f}"
              f"{summary['cvar95_daily'][k]:>13,.0f}"
              f"{summary['worst_day'][k]:>12,.0f}"
              f"{summary['terminal_soc'][k]:>11,.0f}")
    if np.isfinite(offline_cost):
        print(f"{'Q4_offline':<14}{offline_cost/1e4:>14.2f}"
              f"{offline_cost-base_q1:>+14,.0f}")
    print("\n边际贡献（元）：")
    for k, v in summary["effects_yuan"].items():
        print(f"  {k:<28}{v:>+14,.0f}")

    print("\n4-2 分支（计划冻结 + 5×紧急购电，无差额项）：")
    base42 = tot42["Q42_point"]
    for k in LIVE42:
        print(f"  {k:<28}{tot42[k]:>14,.0f} 元  相对Q42_point "
              f"{tot42[k]-base42:>+12,.0f}")
    for k, v in summary["effects_42_yuan"].items():
        print(f"    {k:<38}{v:>+12,.0f}")
    print(f"\n已写出 {DAILY_CSV.name} 与 {SUMMARY_JSON.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=BACKTEST_START)
    ap.add_argument("--end", default=BACKTEST_END)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--scen", type=int, default=10)
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--shape-win", type=int, default=35)
    ap.add_argument("--level-win", type=int, default=14)
    ap.add_argument("--price-model", choices=["legacy", "hybrid"],
                    default="hybrid",
                    help="价格预测：hybrid=Ridge水平+形状融合+保序校准（默认）")
    ap.add_argument("--price-retrain", type=int, default=7)
    ap.add_argument("--price-alpha", type=float, default=10.0)
    ap.add_argument("--price-cal-win", type=int, default=28)
    ap.add_argument("--q42-quantile", type=float, default=0.8,
                    help="4-2有限情景分位；0.8为理论值，可用预热窗校准")
    ap.add_argument("--vterm-mult", type=float, default=1.0,
                    help="日末库存续存价值倍率")
    ap.add_argument("--release-quantile", type=float, default=0.6,
                    help="4-2预测价达到剩余时域该分位后，才额外放电缓冲预测误差")
    ap.add_argument("--rule", default="C", choices=["B", "C"])
    ap.add_argument("--no-grid-charge", action="store_true",
                    help="禁止用网电给储能充电（题面允许低价时段从外网充电，默认开启）")
    ap.add_argument("--no-vterm", action="store_true",
                    help="关闭日末库存续存价值（会恶意放空电池，仅作对照）")
    ap.add_argument("--no-offline", action="store_true",
                    help="跳过离线下界 LP（大窗口较慢）")
    ap.add_argument("--foresight", action="store_true",
                    help="用旧的全知执行（一次性 LP 追索，偏乐观），仅作对照")
    ap.add_argument("--exec", default="traj", choices=["dp", "traj"],
                    help="执行器：traj=日初轨迹裁剪（正式口径）；"
                         "dp=价格化 CPWL-DP（可用但比 traj 贵约 9%，待精化水位规则）")
    ap.add_argument("--greedy-discharge", action="store_true",
                    help="4-3 缺额时储能充分放电补缺（放到SOC下限），以降低紧急购电")
    ap.add_argument("--source-q42", default="fusion",
                    choices=["fusion", "ridge", "mean"],
                    help="4-2源荷预测；默认使用问题二最终融合模型")
    ap.add_argument("--source-q43", default="q3fusion",
                    choices=["q3fusion", "fusion", "ridge", "mean"],
                    help="4-3源荷接口；默认使用问题三最终融合预测")
    ap.add_argument("--commit", default="nb", choices=["nb", "sp"],
                    help="承诺量求解方式：nb=执行一致性报童（默认）；sp=两阶段SP（对照）")
    ap.add_argument("--intra-adj", action="store_true",
                    help="4-3 窗口内允许追加确认量（缺口走 1.5× 超购而非 5× 紧急）")
    ap.add_argument("--pv-blend", action="store_true",
                    help="0:00 光伏用「附件3 + 周期历史」融合（权重 0.389）")
    ap.add_argument("--pv-fused", action="store_true",
                    help="4-3 光伏接问题三正式通道：偏差校正 + 含交叉项最小二乘滚动融合（窗口 28 天）")
    ap.add_argument("--q43-commit", default="point",
                    choices=["point", "sp", "nb", "nb5"],
                    help="4-3 正式路线（Q0_fixed）的 0:00 计划量规则："
                         "point=确定性 LP（默认）；sp=两阶段情景 SP；nb=报童分位")
    ap.add_argument("--roll", dest="roll", action="store_true", default=True,
                    help="4-3 主路线 Q2_nb 启用 0/6/12/18 四时刻滚动重解（默认开启）")
    ap.add_argument("--no-roll", dest="roll", action="store_false",
                    help="Q2_nb 回到 0:00 单次决策（与既有签核结果一致）")
    ap.add_argument("--export-segments", action="store_true",
                    help="逐段结果落盘 results/q4_segments.csv（默认不写）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out-tag", default=None,
                    help="把产物写到 results/_tmp_<tag>_*.csv/json，避免覆盖正式结果")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
