# 微网与外部电网电力调控策略

[![CI](https://github.com/Alwayshere-su/cumcm-2026-microgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/Alwayshere-su/cumcm-2026-microgrid/actions/workflows/ci.yml)

2026 年高教社杯全国大学生数学建模竞赛 **C 题** 的赛后公开整理。项目研究并网型微网在光伏、负荷与电价不确定条件下的购电与储能协同调度：从确定性线性规划出发，逐层引入源荷情景、滚动预报与实时波动电价，形成一套可回测、可复现的日前—日内调度框架。

> 本仓库为参赛队赛后自行整理，**不代表竞赛官方**，也未收录任何赛题原始附件、官方参考答案或参考论文。

## 问题与模型

| 问题 | 信息条件 | 主要方法 | 决策粒度 |
| --- | --- | --- | --- |
| 问题一 | 电价、负荷、光伏均确定 | 线性规划（HiGHS）；MILP 互证；对偶影子价格；连续状态动态规划扩展 | 典型日 144 个 10 分钟时段 |
| 问题二 | 电价按日重复，源荷逐日变化 | 分解 + Ridge 融合预测；30 条历史残差情景；情景平均近似（SAA）；动态规划库存价值；紧急购电 5 倍价 | 2025 年逐日 0:00 计划 |
| 问题三 | 每日 0:00 / 6:00 / 12:00 / 18:00 更新光伏预报 | 条件库存价值；滚动时域（MPC）；少购违约金与超购附加费；A/B/C 三种结算口径对比 | 每日 4 次滚动调整 |
| 问题四 | 外部电价实时波动 | 因果价格预测；重建 4-2 / 4-3 模型；执行层因果约束；离线完全信息下界与遗憾值 | 逐段执行 |

四问共用三项骨架：逐时段能量守恒、储能 SOC 递推、不可前视的因果执行。模型细节见 [`docs/model/`](docs/model/)。

## 主要结果

回测区间为 **2025-02-01 至 2025-12-31（334 天）**，全部结果由 `results/final/` 中的机器可读指标复核。

| 指标 | 结果 |
| --- | --- |
| 问题一典型日购电费用 | **35,126.95 元** |
| 问题一相对无储能方案节省 | **12,925.10 元（26.90%）** |
| 问题一储能吞吐量 | 37,540.61 kWh |
| 问题二全年总费用 | **13,556,295.47 元**（计划 12,752,110.73 + 紧急 804,184.74） |
| 问题二紧急购电量 | 146,794.55 kWh（119 天触发） |
| 问题三全年总费用（口径 C） | **13,027,438.25 元** |
| 问题三 A / B / C 口径 | 13,253,798.33 / 13,134,811.88 / 13,027,438.25 元 |
| 问题三 CVaR95（日费用） | 63,895.07 元 |
| 问题四离线完全信息下界 | 12,780,883.29 元 |

问题四的 4-2 / 4-3 多口径结果、价格信号价值与承诺价值分解见 [`results/final/q4_42_fusion_summary.json`](results/final/q4_42_fusion_summary.json)、[`results/final/paper_metrics.json`](results/final/paper_metrics.json)。

## 仓库结构

```text
.
├── code/                 45 个 Python 源程序（含 _project.py、_style/plot_style.py）
├── tests/                9 个验收测试（物理守恒、结算恒等式、因果性、写入器契约）
├── paper/                论文、AI 工具使用报告、附录 A/B
├── docs/                 四问最终模型、复现说明、数据说明、论文数字校验、岭回归证据
├── figures/paper/        论文插图 30 张（按论文图号命名）
├── results/final/        5 个赛题结果表 + 14 个关键指标 JSON
├── DATA.md               数据来源与目录约定
├── ATTRIBUTION.md        赛题、第三方资源与合规说明
└── requirements.txt
```

### 代码入口

| 文件 | 作用 |
| --- | --- |
| `code/preprocess.py` | 附件 1–5 → 统一 10 分钟面板、质量契约、基础图 |
| `code/solve_q1_lp.py` | 问题一正式线性规划入口 |
| `code/solve_q2_method3.py` | 问题二融合预测 + 情景 SAA + DP 库存价值 |
| `code/solve_q3_final.py` | 问题三四时刻滚动优化与口径 C 结算 |
| `code/forecast_q4_price.py` | 问题四因果价格预测模块 |
| `code/harness_q4.py` | 问题四 4-2 / 4-3 策略回测与对照 |
| `code/write_result4.py` | 问题四结果表写入与审计 |
| `code/verify_paper_numbers.py` | 论文数值与结果文件的一致性核对 |

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. 把官方附件按 DATA.md 的目录约定放到 附件/
python code/preprocess.py

# 2. 四问正式求解（完整回测耗时较长）
python code/solve_q1_lp.py
python code/solve_q2_method3.py --save
python code/solve_q3_final.py --save
python code/harness_q4.py
python code/write_result4.py

# 3. 验收测试
pytest tests -q
```

本机在包含附件与派生数据的环境下执行结果为 **51 passed, 4 skipped**；4 项跳过均因未放入 `附件/附件5/` 的官方结果模板。**全新 clone 不含赛题附件**，数据相关测试会自动跳过（**29 passed, 26 skipped**），合成数据与结算恒等式测试仍会运行。

## AI 工具使用

参赛队按竞赛要求提交了独立的 AI 使用报告：[`paper/AI使用报告.pdf`](paper/AI使用报告.pdf)（Word 版与逐环节记录见 `paper/AI使用报告.docx`、`docs/ai_usage_log.md`）。使用程度为**中度辅助**：AI 参与代码调试与实现、绘图实现、自动化检查及部分文字初稿；读题拆解、模型选择、数学口径与参数、结果解释与最终定稿由参赛队完成。所有 AI 生成内容均经人工审查、物理守恒检查与样本外回测。

## 数据与合规

- **不收录**赛题附件 1–5、官方参考答案、参考论文与格式模板；复现所需数据请从竞赛官方渠道获取，见 [`DATA.md`](DATA.md)。
- 仓库中的论文、图表、代码与结果文件均为参赛队自主产出。
- 第三方 Python 依赖遵循各自许可证，详见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。

## 引用

```bibtex
@misc{cumcm2026microgrid,
  title  = {微网与外部电网电力调控策略},
  author = {Alwayshere-su},
  year   = {2026},
  note   = {2026 年全国大学生数学建模竞赛 C 题},
  url    = {https://github.com/Alwayshere-su/cumcm-2026-microgrid}
}
```

## 许可证

- 代码：[MIT](LICENSE)
- 论文、文档与图表：[CC BY 4.0](LICENSE-docs)
