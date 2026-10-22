# 微网与外部电网电力调控策略

> Microgrid dispatch optimization — 2026 年高教社杯全国大学生数学建模竞赛 C 题

[![CI](https://github.com/Alwayshere-su/cumcm-2026-microgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/Alwayshere-su/cumcm-2026-microgrid/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

## 简介

一个微网与外部电网之间的购电与储能调度项目：在电价、负荷与光伏出力可预测或部分未知的条件下，决定每个 10 分钟时段的购电量与储能充放电功率，使全年用电成本最小。

项目包含四个递进的问题：确定性线性规划、考虑源荷不确定性的情景随机优化、日内四次滚动调整，以及外部电价实时波动下的因果执行。四问共用同一套逐时段能量守恒与 SOC 递推骨架，差别在信息条件与决策粒度。

## 方法

| 问题 | 信息条件 | 主要方法 | 决策粒度 |
| --- | --- | --- | --- |
| 问题一 | 电价、负荷、光伏均确定 | 线性规划（HiGHS）；MILP 互证；对偶影子价格；连续状态动态规划扩展 | 典型日 144 个 10 分钟时段 |
| 问题二 | 电价按日重复，源荷逐日变化 | 分解 + Ridge 融合预测；30 条历史残差情景；情景平均近似（SAA）；动态规划库存价值；紧急购电 5 倍价 | 2025 年逐日 0:00 计划 |
| 问题三 | 每日 0:00 / 6:00 / 12:00 / 18:00 更新光伏预报 | 条件库存价值；滚动时域（MPC）；少购违约金与超购附加费；A/B/C 三种结算口径对比 | 每日 4 次滚动调整 |
| 问题四 | 外部电价实时波动 | 因果价格预测；重建 4-2 / 4-3 模型；执行层因果约束；离线完全信息下界与遗憾值 | 逐段执行 |

完整推导、符号与约束见 [`docs/model/`](docs/model/)。

## 结果

回测区间为 **2025-02-01 至 2025-12-31（334 天）**，全部数值由 `results/final/` 中的机器可读指标复核。

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

问题四的 4-2 / 4-3 多口径结果、价格信号价值与承诺价值分解见 [`results/final/q4_42_fusion_summary.json`](results/final/q4_42_fusion_summary.json) 与 [`results/final/metrics.json`](results/final/metrics.json)。

## 项目结构

```text
.
├── code/                 求解、预测、回测、绘图与核验程序
│   └── _style/           统一绘图样式
├── tests/                验收测试
├── docs/model/           四问模型推导、特征选择与灵敏度分析
├── results/final/        5 张结果表 + 14 个关键指标 JSON
├── ATTRIBUTION.md        题目来源、第三方依赖与使用声明
├── requirements.txt
└── .github/workflows/    CI：依赖安装、语法检查、测试
```

代码入口与模块说明见 [`code/README.md`](code/README.md)。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 准备数据

赛题的原始附件不在本仓库中，请从竞赛官方渠道获取，并按以下结构放在项目根目录：

```text
附件/
├── 附件1.xlsx           # 典型日：144 时段电价、负荷、光伏预测
├── 附件2.xlsx           # 全年实测：小区负载、光伏发电实际功率
├── 附件3.xlsx           # 每日 0:00 / 6:00 / 12:00 / 18:00 光伏预报
├── 附件4.xlsx           # 全年动态电价
└── 附件5/
    ├── result1.xlsx     # 问题一结果模板
    ├── result2.xlsx     # 问题二结果模板
    ├── result3.xlsx     # 问题三结果模板
    ├── result4-2.xlsx   # 问题四 4-2 结果模板
    └── result4-3.xlsx   # 问题四 4-3 结果模板
```

### 运行

```bash
python code/preprocess.py          # 附件 1–5 → 统一 10 分钟面板
python code/solve_q1_lp.py         # 问题一
python code/solve_q2_method3.py --save
python code/solve_q3_final.py --save
python code/harness_q4.py
python code/write_result4.py
```

中间数据写入 `data_processed/`，结果写入 `results/`（均不纳入版本控制）。

### 测试

```bash
pytest tests -q
```

- 含附件与派生数据：**51 passed, 4 skipped**（跳过项缺 `附件/附件5/` 的官方结果模板）。
- 全新 clone 不含附件：数据相关测试自动跳过，**29 passed, 26 skipped**；合成数据与结算恒等式测试仍会运行。

## 声明

- 赛题、附件与官方模板的版权归 2026 年全国大学生数学建模竞赛组委会所有；本仓库未收录赛题附件、官方参考答案、参考论文或格式模板，也不代表竞赛官方。
- `code/`、`docs/` 与 `results/final/` 为作者独立完成与运行产出。
- 第三方依赖遵循各自许可证，见 [`ATTRIBUTION.md`](ATTRIBUTION.md)。

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
- 文档：[CC BY 4.0](LICENSE-docs)
