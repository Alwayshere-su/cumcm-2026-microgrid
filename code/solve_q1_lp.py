"""问题一主模型：LP（连续松弛，无 0-1 变量）。

本脚本是问题一的「定稿口径」求解器：
  * 主模型 = LP 松弛，先最小化费用，再在最优费用容差内最小化电池吞吐；
  * 不引入充放电互斥 0-1 变量，互斥性由命题保证（非负电价 + η<1 + 允许未利用供能
    + 日末闭环 ⇒ 同时充放电严格劣），代码里用断言兜底；
  * 含 0-1 的 MILP 由同一标准入口求解一次，仅作数值验证（两者最优值相同）。

生成：result1.xlsx、results/q1_table1_purchase.csv、results/q1_table2_storage.csv、
      results/q1_dispatch_10min.csv、results/q1_summary.json、results/q1_result1_mapping_audit.csv
"""

from __future__ import annotations

import json

from solve_q1 import solve_q1, write_outputs


def main() -> None:
    result, summary = solve_q1()
    write_outputs(result, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nresult1.xlsx 已用 LP 解重新生成。")


if __name__ == "__main__":
    main()
