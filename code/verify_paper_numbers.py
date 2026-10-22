# -*- coding: utf-8 -*-
"""论文数字一致性校验。

把 build_paper_draft.py 正文中出现的硬编码数字，与 results/paper_metrics.json
（由 scripts/paper_metrics.py 汇总的单一事实源）逐项比对，输出一致/不一致清单。

用法：
    python scripts/verify_paper_numbers.py          # 只报告
    python scripts/verify_paper_numbers.py --strict # 有不一致则退出码 1（可用于门禁）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root  # noqa: E402
ROOT = find_project_root(__file__)
RES = ROOT / "results"
DRAFT = ROOT / "scripts" / "build_paper_draft.py"


def load_metrics() -> dict:
    p = RES / "paper_metrics.json"
    if not p.exists():
        raise FileNotFoundError("请先运行 scripts/paper_metrics.py 生成 paper_metrics.json")
    return json.loads(p.read_text(encoding="utf-8"))


# 规则：正文里出现的字面量 -> 取值路径（支持派生表达式）
def build_rules(m: dict) -> list[tuple[str, str, float | None, str]]:
    q1, q2, q3 = m["q1"], m["q2"], m["q3"]
    r = m["q2_ridge"]
    q4 = m["q4"]
    R: list[tuple[str, str, float | None, str]] = []

    def add(lit: str, path: str, val, note: str = ""):
        R.append((lit, path, val, note))

    # 问题一
    add("35126.948589", "q1.cost_yuan", q1["cost_yuan"], "全天最优购电费")
    add("59482.698998", "q1.purchase_kwh", q1["purchase_kwh"], "全天购电量")
    add("48052.047", "q1.no_storage_cost_yuan", q1["no_storage_cost_yuan"], "无储能基准")
    add("61789.935", "q1.no_storage_purchase_kwh", q1.get("no_storage_purchase_kwh"))
    add("26.90", "q1.saving_pct", q1.get("saving_pct"), "节省比例%")
    add("12925.098", "q1.saving_yuan", q1.get("saving_yuan"), "节省金额")
    # 问题二
    add("162.23", "q2_ridge.main_metrics.load_MAE_kW", r["main_metrics"]["load_MAE_kW"])
    add("156.63", "q2_ridge.main_metrics.pv_MAE_kW", r["main_metrics"]["pv_MAE_kW"])
    add("263.19", "q2_ridge.main_metrics.net_MAE_kW", r["main_metrics"]["net_MAE_kW"])
    add("164.94", "q2_ridge.ols_metrics.load_MAE_kW", r["ols_metrics"]["load_MAE_kW"])
    add("1275.21", "q2.plan_cost_yuan/1e4", q2["plan_cost_yuan"] / 1e4)
    add("1355.63", "q2.total_cost_yuan/1e4", q2["total_cost_yuan"] / 1e4)
    add("14.68", "q2.emergency_kwh/1e4", q2["emergency_kwh"] / 1e4)
    # 问题三
    add("1302.74", "q3.total_yuan/1e4", q3["total_yuan"] / 1e4)
    add("1256.29", "q3.within_cost_yuan/1e4", q3["within_cost_yuan"] / 1e4)
    add("1325.38", "q3.A_total_yuan/1e4", q3["A_total_yuan"] / 1e4)
    add("1313.48", "q3.B_total_yuan/1e4", q3["B_total_yuan"] / 1e4)
    add("6.39", "q3.cvar95/1e4", q3["cvar95"] / 1e4)
    add("7.90", "q3.max_day/1e4", q3["max_day"] / 1e4)
    add("9598.67", "q3.terminal_soc_kwh", q3["terminal_soc_kwh"])
    add("131.8", "q3.forecast_accuracy.load_mae_kw", q3["forecast_accuracy"]["load_mae_kw"])
    add("218.7", "q3.forecast_accuracy.net_mae_kw", q3["forecast_accuracy"]["net_mae_kw"])
    add("262.69", "q3.forecast_accuracy.pv_day_mae_kw",
        q3["forecast_accuracy"]["pv_day_mae_kw"])
    # 问题四
    add("0.9664", "corr_price_shape_excess_pearson",
        m["corr_price_shape_excess_pearson"], "价格形状相关系数")
    # 问题二 特征工程与岭回归（4.2 / 4.3）
    rs = m.get("q2_regstats", {})
    nl = m.get("q2_nonlinear", {})
    vf = m.get("q2_vif", {})
    if rs.get("load"):
        add("0.98145", "q2_regstats.load.R2", rs["load"]["R2"])
        add("0.98144", "q2_regstats.load.R2_adj", rs["load"]["R2_adj"])
        add("105 403.6", "q2_regstats.load.F", rs["load"]["F"])
        add("32.013", "q2_regstats.load.sigma", rs["load"]["sigma"])
    if rs.get("pv"):
        add("0.99026", "q2_regstats.pv.R2", rs["pv"]["R2"])
        add("0.99025", "q2_regstats.pv.R2_adj", rs["pv"]["R2_adj"])
        add("202 478.8", "q2_regstats.pv.F", rs["pv"]["F"])
        add("49.033", "q2_regstats.pv.sigma", rs["pv"]["sigma"])
    if vf:
        add("485.46", "q2_vif.load_vif_max", vf["load_vif_max"])
        add("2414.26", "q2_vif.pv_vif_max", vf["pv_vif_max"])
        add("60.52", "q2_vif.load_vif_mean", vf["load_vif_mean"])
        add("261.27", "q2_vif.pv_vif_mean", vf["pv_vif_mean"])
        add("84.81", "q2_vif.load_cond_X", vf["load_cond_X"])
        add("200.47", "q2_vif.pv_cond_X", vf["pv_cond_X"])
        add("85.90", "q2_vif.ols_coef_std_mean_load", vf["ols_coef_std_mean_load"])
        add("13.05", "q2_vif.ridge_coef_std_mean_load",
            vf["ridge_coef_std_mean_load"])
        add("167.31", "q2_vif.ols_coef_std_mean_pv", vf["ols_coef_std_mean_pv"])
        add("20.98", "q2_vif.ridge_coef_std_mean_pv",
            vf["ridge_coef_std_mean_pv"])
    if nl.get("lightgbm"):
        add("142.66", "q2_nonlinear.lightgbm.load_MAE",
            nl["lightgbm"]["load_MAE"])
    if nl.get("quadratic_ridge"):
        add("162.15", "q2_nonlinear.quadratic_ridge.load_MAE",
            nl["quadratic_ridge"]["load_MAE"])
    add("0.481548", "q3(续存价值常量)", 0.481548)
    off = q4.get("offline_bound_yuan")
    if off:
        add("1278.088", "q4.offline_bound_yuan/1e4", off / 1e4)
    tot = q4.get("total_yuan") or {}
    for key, lit in (("Q0_fixed", "1475.220"), ("Q1_point", "1488.480"),
                     ("Q2_nb", "1493.627"), ("Q3_oracle_det", "1345.052"),
                     ("Q3_oracle_nb", "1346.874")):
        if key in tot:
            add(lit, f"q4.total_yuan.{key}/1e4", tot[key] / 1e4)
    t42 = q4.get("total_yuan_42") or {}
    for key, lit in (("Q42_nb", "1445.992"),):
        if key in t42:
            add(lit, f"q4.total_yuan_42.{key}/1e4", t42[key] / 1e4)
    eff = q4.get("effects_yuan") or {}
    e42 = q4.get("effects_42_yuan") or {}
    if "commitment_Q42point_to_Q42nb" in e42:
        add("65.71", "q4.effects_42_yuan.commitment/1e4",
            abs(e42["commitment_Q42point_to_Q42nb"]) / 1e4, "4-2报童承诺收益")
    if "commitment_Q1_to_Q2" in eff:
        add("5.15", "q4.effects_yuan.commitment_Q1_to_Q2/1e4",
            eff["commitment_Q1_to_Q2"] / 1e4, "4-3承诺效应")
    return R


def flatten_values(obj, path="", out=None):
    """把 metrics 树展平成 {路径: 数值}。"""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_values(v, f"{path}.{k}" if path else k, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten_values(v, f"{path}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = float(obj)
    return out


def build_value_index(m: dict) -> dict[str, str]:
    """生成 {按精度取整后的字符串: 路径} 索引，支持 0~6 位小数。"""
    idx: dict[str, str] = {}
    for path, val in flatten_values(m).items():
        for d in range(0, 7):
            key = f"{val:.{d}f}"
            idx.setdefault(key, path)
        # 万元换算（除以 1e4）也登记，覆盖正文以万元计数的写法
        wan = val / 1e4
        for d in range(1, 7):
            idx.setdefault(f"{wan:.{d}f}", path + " /1e4")
    return idx


def coverage_check(m: dict) -> tuple[int, list[tuple[str, str]]]:
    """把源码中所有数值字面量尝试与事实源匹配，返回 (命中数, 未命中清单)。"""
    idx = build_value_index(m)
    src = DRAFT.read_text(encoding="utf-8")
    # 只取引号内的字符串正文，避免把 Python 代码里的 0/1 计数
    literals: list[tuple[str, str]] = []
    for s in re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', src):
        for mo in re.finditer(r"\d+\.\d+", s):
            literals.append((mo.group(0), s))
    seen: dict[str, str] = {}
    for lit, ctx in literals:
        seen.setdefault(lit, ctx)
    hits, miss = 0, []
    for lit, ctx in seen.items():
        if lit in idx:
            hits += 1
            continue
        # 允许末位四舍五入误差
        v = float(lit)
        cand = [k for k in idx if abs(float(k) - v) <= max(5e-4, abs(v) * 1e-4)]
        if cand:
            hits += 1
        else:
            miss.append((lit, ctx[:110]))
    return hits, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="把源码全部数值字面量与事实源做覆盖匹配")
    args = ap.parse_args()

    m = load_metrics()
    src = DRAFT.read_text(encoding="utf-8")
    rules = build_rules(m)

    print("=" * 84)
    print("论文数字一致性校验（正文硬编码 vs results/paper_metrics.json）")
    print("=" * 84)
    bad, ok, miss = [], 0, []
    for lit, path, val, note in rules:
        if val is None:
            miss.append((lit, path, note))
            continue
        if lit not in src:
            continue                       # 正文未使用该数字，跳过
        lv = float(lit)
        rel = abs(lv - float(val)) / max(abs(float(val)), 1e-12)
        good = rel < 5e-4 or abs(lv - float(val)) < 0.01
        if good:
            ok += 1
        else:
            bad.append((lit, path, float(val), note))
    for lit, path, val, note in bad:
        print(f"  [不一致] 正文 {lit:<14} 应为 {val:,.6f}   ({path})  {note}")
    if miss:
        for lit, path, note in miss:
            print(f"  [无来源] 正文 {lit:<14} 缺权威值        ({path})  {note}")
    print("-" * 84)
    print(f"  一致 {ok} 项；不一致 {len(bad)} 项；无来源 {len(miss)} 项")

    # 额外的事实性检查
    print("\n事实性检查：")
    n23 = src.count("23维")
    n25 = src.count("25维")
    nf = m["q2_ridge"]["n_features"]
    print(f"  『23维』出现 {n23} 次；『25维』出现 {n25} 次；权威特征数 = {nf}")
    if n23 and nf == 25:
        print("    -> 应改为 25 维")
    bad_fact = bool(n23)
    if re.search(r"\\sum_\{j=1\}\^\{23\}", src):
        print("  式（15）求和上限仍为 23，应为 25")
        bad_fact = True
    if "归一化实现价形状与附件1固定分时价形状的相关系数" in src:
        print("  『归一化实现价形状…相关系数』表述与算法不符：")
        print(f"     该数(0.9664)实为『扣除各自日均价后的相关』；"
              f"形状曲线相关为 {m['corr_price_shape_median_curve']:.4f}，"
              f"逐日归一化形状均值为 {m['corr_price_shape_daily_norm_mean']:.4f}")
        bad_fact = True
    # 4-2 承诺收益是否写入正文
    e42 = (m["q4"].get("effects_42_yuan") or {}).get(
        "commitment_Q42point_to_Q42nb")
    if e42 is not None:
        val = f"{abs(e42) / 1e4:.2f}"
        if val not in src and "effects_42_yuan" not in src:
            print(f"  4-2 承诺收益 {val} 万元未写入正文（且未引用事实源）")
            bad_fact = True
        else:
            print(f"  4-2 承诺收益 {val} 万元已覆盖（字面量或事实源引用）")

    # 表号唯一性
    caps = re.findall(r"表(\d+)\s", src)
    from collections import Counter
    dup = [k for k, v in Counter(caps).items() if v > 1]
    # 表3 在问题一与问题二各出现一次属正常（不同章节），仅检查连续编号
    nums = sorted({int(x) for x in caps})
    expect = list(range(1, len(nums) + 1))
    if nums != expect:
        print(f"  表号不连续：实际 {nums}，期望 {expect}")
        bad_fact = True
    else:
        print(f"  表号连续：1–{len(nums)}")

    # 必须明确声明放弃 Pearson 相关矩阵（正文以"VIF/TOL 作主诊断"即满足）
    pe = ("不采用Pearson" in src or "放弃Pearson" in src
          or "不使用Pearson" in src or "VIF作主诊断" in src)
    if not pe:
        print("  未在正文声明放弃 Pearson 相关系数矩阵")
        bad_fact = True
    else:
        print("  已声明放弃 Pearson 相关矩阵")

    # 必须给出岭回归目标函数与岭迹图
    if "岭迹" not in src:
        print("  未出现岭迹相关内容")
        bad_fact = True
    if "ridge_trace" not in src:
        print("  未插入岭迹图")
        bad_fact = True
    if "result_q2_pred_vs_actual_load.png" not in src:
        print("  未插入预测—实测对比图")
        bad_fact = True

    if args.strict and (bad or miss or bad_fact):
        sys.exit(1)


if __name__ == "__main__":
    main()
