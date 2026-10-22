# 问题二 · 岭回归相关性证据包

本文件夹汇总了"问题二预测层使用岭回归"的全部支撑证据，核心是**证明回归特征之间
存在显著多重共线性**（即你要的相关性证明），从而说明采用岭回归（Ridge, α=1.0）
而非普通最小二乘（OLS）是必要的、有依据的。

- 数据区间：`data_processed/microgrid_10min.csv`（2025 全年）
- 特征：问题二预测层所用的 **25 维**特征
  （13 个滞后/滚动/日历 + 12 个日内/年内多阶谐波）
- 口径：与 `code/forecast_q2_reg.py` 的 `FEATURES` 完全一致
- 概念澄清：岭回归需要的"相关性"是**自变量彼此之间的相关性（多重共线性）**，
  不是自变量与目标变量的相关性。

---

## 一、关键结论速览

### 1. 多重共线性诊断（负荷 / 光伏）

| 指标 | 负荷 load | 光伏 pv | 判定 |
|---|---:|---:|---|
| `|r|≥0.7` 的强相关特征对数 | 28 | 57 | 成片强相关 |
| 最强相关对（r） | 滚动均值7 ~ 滚动均值14：**+0.997** | 同左：**+0.999** | 严重重叠 |
| VIF 最大值 | **485.5** | **2414.3** | 远超阈值 10 |
| VIF 均值 | 60.5 | 261.3 | — |
| VIF>10 的特征数 | 10 / 25 | 10 / 25 | 严重共线 |
| 条件数 κ(X) | 84.8 | 200.5 | 设计矩阵病态 |
| 条件数 κ(X′X) | 7 192 | 40 187 | — |

结论：滞后项、滚动统计项与谐波项之间存在结构性重叠，OLS 系数方差被严重放大。

### 2. 系数稳定性（问题二"每 7 天重训"共 ≈48 个训练窗口）

| 指标 | 负荷 load | 光伏 pv |
|---|---:|---:|
| OLS 系数跨窗口标准差（均值） | 85.90 | 167.31 |
| Ridge(α=1) 系数跨窗口标准差（均值） | **13.05** | **20.98** |
| 平均波动降至 | **1/6.6** | **1/8.0** |
| 最不稳特征 `doy_sin1` 的 std | OLS 580.2 → Ridge 22.5 | OLS 1094.6 → Ridge 23.6 |

结论：共线性使 OLS 系数在不同训练窗口间剧烈跳动；岭回归以极小偏差把它压稳。
**这是"为什么不用 OLS"最直接的量化证据。**

### 3. 岭回归的实际效果（辅助证据）

| 指标 | OLS | Ridge(α=1) |
|---|---:|---:|
| 负荷预测 MAE (kW) | 164.94 | **162.23** |
| 光伏预测 MAE (kW) | 157.05 | **156.63** |

调度总成本（元）：`reg_ridge` 15 319 339 vs `lgb` 15 372 861 vs `naive` 21 396 118
（见 `data/q2_reg_vs_lgb_cost.json`）。

---

## 二、目录结构

```
岭回归相关性证据/
├─ README.md                        本说明
├─ data/                            数据文件（CSV / JSON）
│   ├─ q2_reg_feature_corr_load.csv      特征相关矩阵（负荷）
│   ├─ q2_reg_feature_corr_pv.csv        特征相关矩阵（光伏）
│   ├─ q2_reg_high_corr_pairs.csv        |r|≥0.7 高相关特征对（含中英列名）
│   ├─ q2_reg_vif_load.csv               VIF（负荷）
│   ├─ q2_reg_vif_pv.csv                 VIF（光伏）
│   ├─ q2_reg_condition_number.csv       条件数 κ(X)、κ(X′X)（负荷/光伏）
│   ├─ q2_reg_ridge_trace_load.csv       岭迹：系数随 α（60 个取值）
│   ├─ q2_reg_ridge_trace_pv.csv         岭迹（光伏）
│   ├─ q2_reg_ols_vs_ridge_coef_load.csv OLS vs Ridge 系数（标准化尺度，负荷）
│   ├─ q2_reg_ols_vs_ridge_coef_pv.csv   同上（光伏）
│   ├─ q2_reg_coef_stability.csv         系数跨窗口稳定性（≈48 窗口）
│   ├─ q2_reg_collinearity_summary.json  以上全部指标汇总（便于复查）
│   ├─ q2_reg_ols_comparison.json        OLS 预测精度对比
│   ├─ q2_reg_ridge_comparison.json      Ridge 预测精度对比
│   ├─ q2_reg_ols_vs_ridge_4dates.csv    四个代表日 OLS vs Ridge 逐点预测
│   └─ q2_reg_vs_lgb_cost.json           naive / lgb / reg_ridge 调度成本对比
├─ figures/                         图（PNG，150 dpi）
│   ├─ q2_reg_feature_correlation.png    特征相关热力图（|r|≥0.7 标注）
│   ├─ q2_reg_vif.png                    VIF 柱状图（含 5/10 阈值线）
│   ├─ q2_reg_ridge_trace_load.png       岭迹图·负荷（标注 α=1）
│   ├─ q2_reg_ridge_trace_pv.png         岭迹图·光伏（标注 α=1）
│   └─ q2_reg_coef_stability.png         OLS vs Ridge 系数稳定性对比
└─ scripts/
    └─ q2_reg_collinearity.py        生成 data/ 与 figures/ 的复现脚本
```

---

## 三、文件来源说明

- `q2_reg_collinearity.py` 产出：相关矩阵、高相关对、VIF、条件数、岭迹、
  OLS/Ridge 系数、系数稳定性这七类文件，以及 `figures/` 全部 4 张图。
- `forecast_q2_reg.py` 产出：`q2_reg_ridge_comparison.json`、`q2_reg_ols_comparison.json`。
- `compare_reg_ols_ridge_dates.py` 产出：`q2_reg_ols_vs_ridge_4dates.csv`。
- `benchmark_q2_reg.py` 产出：`q2_reg_vs_lgb_cost.json`。

---

## 四、复现方式

```powershell
# 主脚本（生成共线性诊断全套数据与图）
python scripts/q2_reg_collinearity.py

# 精度 / 成本对比（可选）
python scripts/forecast_q2_reg.py ridge
python scripts/forecast_q2_reg.py ols
python scripts/compare_reg_ols_ridge_dates.py
python scripts/benchmark_q2_reg.py
```

> 注意：`q2_reg_collinearity.py` 的输出路径固定为仓库的 `results/` 与 `figures/`。
> 重跑会在那两个目录重新生成同名文件；如需回填本文件夹，请把产物移到
> `data/` 与 `figures/`（或修改脚本中的 `RESULTS` / `FIGDIR` 常量指向本文件夹）。

---

## 五、论文可直接引用的措辞

> 预测层所用特征同时包含多阶滞后（lag1/2/7/14）、多窗口滚动统计
> （mean/std 3/7/14）与日内/年内谐波项。经诊断，这些特征之间呈显著多重共线性：
> 负荷特征中 |r|≥0.7 的对数达 28 对（光伏 57 对），滚动均值7与滚动均值14 的
> 相关系数高达 0.997；方差膨胀因子 VIF 最大为 485.5（负荷）/2414.3（光伏），
> 远超 10 的共线判定阈值，标准化设计矩阵条件数为 84.8/200.5。多重共线性会使
> OLS 估计方差剧增、系数不稳定——在 48 个重训窗口上，OLS 系数跨窗口标准差均值
> 为 85.9（负荷）/167.3（光伏），而岭回归（α=1）仅为 13.1/21.0。故采用岭回归，
> 以极小的偏差换取系数稳定性的显著提升。

---

## 六、两点提醒

1. **特征维度**：本证据基于 25 维特征。若论文正文写"23 维"，请与代码
   （`forecast_q2_reg.FEATURES`，长度 25）统一，避免不一致。
2. **α=1 时 Ridge 与 OLS 结果接近**（系数相关 0.99997）：若被追问"为何不用 OLS"，
   用"系数稳定性"小节回答——差别不在精度而在估计的稳健性，这正是岭回归的目的。
   若需进一步强化，可对 α 做交叉验证选参并给出精度对比。
