# 1997年历史台风回算：阶段1分析

本目录是对相邻 `typhoon_events_1997_seed42_compact/` 的**只读二次分析**。原始预测、模型权重、ERA5 和 GESLA 均未修改；这里的CSV和图全部由已有逐小时预测重新计算。1997年案例已被查看过，属于追加历史回算，不是新的盲测。未来气象强迫为已知的 ERA5 再分析，不是业务实时预报。

## 文件

- `figures/*_process.png/.pdf`：每场三次固定起报的72小时曲线，黑点是该起报窗口的实测正向峰，彩色叉号是各模型在同一窗口内的预测峰。北海最早起报窗口不包含整场主峰，已在面板中标明。
- `figures/*_lead_error.png/.pdf`：前三面板为**一次起报在每个提前量的单个绝对误差**；右下面板是三个重叠起报在同一提前量上的描述性RMSE，`n=3`，不是独立事件置信区间。未做平滑。
- `figures/*_window_comparison.png/.pdf`：不同起报的72小时累计RMSE和窗口内峰高绝对误差；每个点对应一个起报。
- `figures/lianyungang_*_loss_ablation.png/.pdf`：WINNIE损失函数补充曲线，和主图分开，避免拥挤。
- `tables/window_metrics_recomputed.csv`：所有模型、三次起报、24/48/72小时的累计RMSE/MAE/Bias和同窗口峰值指标。与原 `event_metrics.csv` 数值核对。
- `tables/exact_lead_metrics_recomputed.csv`：精确24/48/72小时在三次起报上的误差汇总，与原 `exact_lead_metrics.csv` 数值核对。
- `tables/lead_error_by_origin.csv`、`lead_rmse_three_origins.csv`：原始逐小时误差及跨起报描述性曲线的数据，单位cm。
- `tables/lead_band_metrics.csv`：按1–12、13–24、25–48、49–72小时固定时段汇总每次起报的误差。
- `tables/peak_coverage.csv`：整场120小时主峰是否进入各72小时起报窗口。
- `tables/hourly_variability_diagnostics.csv`：1小时变化幅度和lag-1自相关的辅助诊断。常数Persistence的自相关为未定义（空值）；不能仅凭该表认定残余潮汐。
- `stage1_report.md`：两场案例的科学解读和限制。
- `loss_selection.md`：已归档1996滚动验证证据、下一轮固定的损失函数主方案及仍需补充的跨事件稳定性证据。

图件为通用期刊风格草稿，提供400 dpi PNG及矢量PDF；正式投稿时仍需核对目标期刊尺寸和字体要求。所有三次起报窗口彼此重叠，不能视为三个独立台风。图中“峰值误差”总在**同一次起报、同一预设窗口**内计算，不等同于整场主峰误差。

## 复现

从 `projects/xiamen_short_term/short_term_forecast/` 运行（无需原始数据或GPU）：

```powershell
python -m src.xiamen_forecast.analyze_typhoon_stage1 --source "..\..\..\reports\experiment_results\typhoon_events_1997_seed42_compact" --output-dir "..\..\..\reports\experiment_results\typhoon_events_1997_stage1_analysis_new"
```

命令默认拒绝覆盖已有输出。若在Mac上复现，同样可从项目目录运行 `python -m src.xiamen_forecast.analyze_typhoon_stage1 --output-dir /tmp/typhoon_stage1_check`。运行时会读取原始精简CSV，逐项核对原指标后再保存结果。

参考的真实最佳路径和模型checkpoint来源见原包的 `source_and_protocol.json` 与各事件 `experiment_config.json`；本目录不复制权重和原始最佳路径总文件。用户提到的“Zhou等（2026）Fig.10/12”未提供可核对的具体文献版本，本轮仅采用清晰展示过程曲线与提前量误差的通用思路，不声称复刻其图或采用其数值。
