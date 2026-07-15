# Journal Figures for Short-Term Storm Surge Forecasts

这些脚本用于把已经完成的短时风暴潮实验结果整理成期刊级图片。它们不会重新训练模型，也不会重新构建 ERA/GESLA 样本，只读取已有的 `metrics.json`、预测 CSV 和滚动预报 CSV。

## 运行位置

建议在保存实验结果的远程实验室 Windows 电脑上运行。当前 Mac 本地没有真实结果，因此不要在本地扫描 `outputs/` 判断实验是否存在。

统一入口示例：

```bash
python -m src.short_term_forecast.journal_figures.make_all_journal_figures ^
  --results-root "E:\path\to\outputs" ^
  --output-dir "E:\path\to\journal_figures"
```

单独绘制某次滚动预报：

```bash
python -m src.short_term_forecast.journal_figures.plot_rolling_forecast ^
  --forecast-csv "E:\path\to\rolling_forecast.csv" ^
  --output-dir "E:\path\to\journal_figures"
```

## 输出格式

当前版本默认只保存：

```text
PNG, 400 dpi
```

保存时使用 `bbox_inches="tight"`。所有 storm surge 图统一显示为 `cm`。

这个设置适合当前实验分析和组会汇报阶段，避免反复生成 PDF/SVG 导致目录过乱。后续论文投稿阶段如需要矢量格式，可再在 `style.py` 中重新开启 PDF/SVG 输出。

## 目录识别规则

统一入口会递归扫描 `--results-root`，寻找以下文件：

```text
metrics.json
val_predictions.csv
rolling_forecast.csv
predictions.csv
loss_history.csv
```

实验信息会优先从 `metrics.json` 读取；如果缺失，则从目录名中推断：

```text
t8 / t12 / t24      -> input_steps
n12 / n72           -> forecast_steps
rolling             -> rolling forecast experiment
cnn / cnn_lstm 等   -> model_name
```

目录名不写死。类似下面这些名称都只是识别示例：

```text
ERA5_1985_1985_hourly_t12_h1
ERA5_1985_1985_hourly_t24_h1
ERA5_1985_1985_t8_h1
ERA5_1985_1997_t8_h1
ERA5_hourly_rolling_1985110100_n12
ERA5_hourly_rolling_1985110100_n72
ERA5_rolling_19851201_n7
ERA5_rolling_19971201_n7
```

## 列名兼容

预测 CSV 不要求列名完全一致。脚本支持常见别名：

```text
observed / observation / y_true / target / actual
predicted / prediction / y_pred / forecast / pred
datetime / time / date / timestamp
lead_time / forecast_step / step / lead_step
persistence / persistence_prediction / persistence_forecast / baseline
```

缺少必要字段时会打印 warning 并跳过当前图，不会中断整个批处理。

## 自动单位转换

脚本会判断 storm surge 数值是否像以 `m` 保存。如果 95% 绝对值明显小于等于 5，会按米处理并乘以 100 转成厘米。已经是厘米的结果不会再转换。

## 测试实验过滤

输入窗口比较图会自动过滤样本量不足的测试实验，避免异常调试结果进入论文图件。默认阈值在：

```text
src/short_term_forecast/journal_figures/config.py
```

```python
MIN_VALIDATION_SAMPLES = 500
```

对于窗口比较实验，如果 `metrics.json` 中的 `n_val`、`val_size`、`validation_samples` 或 `n_validation` 小于该阈值，脚本会跳过该实验并输出类似日志：

```text
Skipped experiment:
ERA5_1985_1985_t8_h1
Reason:
validation sample number too small (n_val=72)
```

## 图的科学含义

`figure_window_metrics`：比较有效输入窗口实验对 Pearson r、RMSE、MAE、RRMSE 的影响。样本量不足的调试实验会被自动过滤。

`figure_window_timeseries`：在共同验证时间段内比较 observed 和不同输入窗口的预测序列。

`figure_window_scatter`：用 observed vs predicted 密度散点检查整体拟合、离散程度和系统偏差。

`figure_rolling_forecast_*`：展示一次滚动预报的时间序列、误差、绝对误差随 lead time 的变化，以及累计 RMSE/MAE。

`figure_peak_analysis_*`：关注 90%、95%、99% 高分位 storm surge，检查峰值 RMSE、MAE、bias、低估比例和相关性。

`figure_residual_diagnostics_*`：检查残差时间结构、分布形态、Q-Q 图和 1-72 小时自相关。

`figure_event_comparison`：对比多个滚动事件，每个事件一行，左侧 observed vs predicted，右侧 error。

## 输出清单

统一入口会生成：

```text
figure_manifest.csv
```

字段包括：

```text
figure_name
source_experiments
output_png
status
warning
```

即使某些结果缺失，也会继续生成其他可用图片。

## 调整字体、颜色和分辨率

统一样式在：

```text
src/short_term_forecast/journal_figures/style.py
```

可调整：

```python
DPI = 400
MIN_VALIDATION_SAMPLES = 500
MODEL_COLORS = {...}
setup_journal_style(font_size=8.0)
```

字体优先使用 Arial；如果系统没有 Arial，会自动回退到 DejaVu Sans。颜色固定为：

```text
observed: black
persistence: orange
cnn: blue
cnn_lstm: green
cnn_gru: purple
tcn: red
transformer: brown
```

脚本只使用 matplotlib，不使用 seaborn，也不会使用 rainbow/jet 色带。
