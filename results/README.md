# 结果目录

## `final/`

对外发布的最终结果，均为参赛队程序产出：

### 赛题结果表（5）

| 文件 | 内容 |
| --- | --- |
| `result1.xlsx` | 问题一：典型日计划购电量、充放电量 |
| `result2.xlsx` | 问题二：全年计划购电量、充放电量 |
| `result3.xlsx` | 问题三：计划购电量、调整购电量、充放电量、紧急购电量 |
| `result4-2.xlsx` | 问题四 4-2：波动电价下的计划与执行结果 |
| `result4-3.xlsx` | 问题四 4-3：波动电价下的滚动调整结果 |

### 关键指标 JSON（14）

包括 `paper_metrics.json`（论文数值总表）、`q1_summary.json`、`q2_summary.json`、`q2_method3_summary.json`、`q2_accuracy_comparison.json`、`q3_summary.json`、`q3_final_summary.json`、`q3_voi_summary.json`、`q1_layer_comparison.json`、`q2_four_days_final.json`、`q4_42_fusion_summary.json`、`q4_offline_bound.json` 及两份写入审计。

## 本地运行产物

`results/` 根目录下的逐段调度 CSV、情景面板、实验备份等体积较大（本机约 300 MB），不纳入版本控制。运行 `code/` 中的求解入口可重新生成。
