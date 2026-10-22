# 源程序清单

所有脚本均可在项目根目录或任意工作目录下以 `python code/<脚本>.py` 运行，不含本机绝对路径。

| 模块 | 主要文件 |
| --- | --- |
| 数据预处理 | `preprocess.py`、`audit_q3_pv_forecast.py` |
| 问题一 | `solve_q1.py`、`solve_q1_lp.py`、`q1_two_layer.py`、`compare_q1_layers.py`、`audit_q1_value_function.py`、`sensitivity_q1.py` |
| 问题二 | `forecast_q2.py`、`forecast_q2_reg.py`、`build_q2_blend_panel.py`、`solve_q2.py`、`solve_q2_twolayer.py`、`solve_q2_method3.py`、`solve_q2_saa.py`、`q2_reg_collinearity.py`、`analyze_q2_ridge.py`、`analyze_q2_features.py`、`q2_ridge_regstats.py` |
| 问题三 | `build_q3_fused_load.py`、`solve_q3.py`、`solve_q3_final.py`、`ablation_q3_forecast.py`、`ablation_q3_2x3.py`、`probe_q3_rolling_sp.py`、`compare_q3_forecasts.py`、`run_q3_voi_fusion.py`、`report_q3_final.py` |
| 问题四 | `forecast_q4_price.py`、`q4_models.py`、`harness_q4.py`、`q4_offline_bound.py`、`write_result4.py`、`report_q4_dates.py` |
| 可视化 | `plot_q1.py`、`plot_q2.py`、`plot_q3.py`、`plot_q4.py` |
| 结果核验 | `report_metrics.py`、`verify_result2.py` |

## 公共模块

| 文件 | 作用 |
| --- | --- |
| `_project.py` | 路径引导：`find_project_root()` 从脚本所在目录逐级上溯，定位同时含 `code/` 与 `results/` 的工程根；`add_source_path()` / `add_style_path()` 注册模块与样式目录 |
| `_style/plot_style.py` | 统一绘图样式（`apply_publication_style` / `export_figure`） |

包内模块（如 `solve_q2`、`q4_models`）优先从本目录导入，不依赖外部同名副本。
