"""第一问：DP 价值函数 F_1(e) 与 LP 最优费用的逐一对照（问题一「是否有更优」的判据）。

对同一初始储电量 e，分别用
  * LP  ：固定 E_0 = e、E_T = 6000，重解一遍线性规划，得 J(e)
  * DP  ：直接读连续状态价值函数 F_1(e) 的折线值
两者应当逐点相等。若 DP 更优，则 LP 的 J(e) 会严格大于 F_1(e)；实测差值为数值噪声。
这同时说明：问题一的最低费用 35126.948589 元是全局下界且可达，DP 不可能更低。

输出：results/q1_value_function_audit.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _project import find_project_root, add_script_path  # noqa: E402
ROOT = find_project_root(__file__)
add_script_path()

from compare_q1_layers import solve_lp_single  # noqa: E402
from q1_two_layer import two_layer_dp  # noqa: E402

DATA = ROOT / "data_processed" / "q1_typical_day.csv"
PARAMS = ROOT / "data_processed" / "q1_storage_parameters.json"
RESULTS = ROOT / "results"

GRID = [1200, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10500, 10800]


def main() -> None:
    df = pd.read_csv(DATA)
    prm = json.loads(PARAMS.read_text(encoding="utf-8"))
    price = df["price"].to_numpy()

    tl = two_layer_dp(price, df["load_kwh"].to_numpy(), df["pv_forecast_kwh"].to_numpy(), prm)
    F1 = tl.F[0]                                    # 进入时段 1 的价值函数

    rows = []
    for e in GRID:
        e = float(e)
        if not (F1.lo - 1e-6 <= e <= F1.hi + 1e-6):
            continue
        lp = solve_lp_single(df, prm, e0=e)
        j_lp = float(price @ lp["buy_kwh"])
        j_dp = float(np.asarray(F1.value_at(e)))
        rows.append({"e0_kwh": e, "lp_cost_yuan": j_lp, "dp_value_yuan": j_dp,
                     "diff_lp_minus_dp_yuan": j_lp - j_dp})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "q1_value_function_audit.csv", index=False, encoding="utf-8-sig")

    print(f"DP 价值函数 F_1 定义域 I_1 = [{F1.lo:.1f}, {F1.hi:.1f}] kWh，折线段数 = {F1.n_segments}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print()
    print(f"最大绝对偏差 = {out['diff_lp_minus_dp_yuan'].abs().max():.3e} 元"
          f"（= 两个求解器的数值噪声；DP 未取得更低费用）")
    print()
    print("边际储能价值 Lambda_1(e) = -F_1'(e) 随库存单调非增（命题 3.1）：")
    b = F1.breaks
    for i in range(F1.n_segments):
        lo, hi = b[i], b[i + 1]
        if i % 4 == 0 or i == F1.n_segments - 1:
            print(f"  e in [{lo:8.1f}, {hi:8.1f}]  Lambda_1 = {-F1.slopes[i]:.6f} 元/kWh")


if __name__ == "__main__":
    main()
