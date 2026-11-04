# 滚动模型台风事件后处理

本目录是只读归档预测的派生分析，原始事件目录和旧图保持原样。
只比较 MSE Rollout-6 与 WHuber Rollout-6；每场三次起报均单独评价72小时，不拼接，也不视为三场独立事件。

- `all_events_rollout_comparison.csv`：前24/48/72小时累计窗口RMSE、72小时窗口峰高/峰时误差和P95/P99统计。
- 各事件的 `extreme_error_summary.csv`：逐起报、逐模型的全部/P95/P99样本误差与低估统计。
- 各事件的四张PNG：过程、逐小时有符号误差、累计RMSE、极端误差。旧图仍在原归档目录。
- `rollout_only_summary.md` 与各事件 `研究报告.md`：中文描述性分析。

阈值优先使用训练期固定阈值。厦门、连云港、北海使用训练期全部有效目标的正向尾部阈值；坎门缺少归档训练阈值，使用已固定的1996验证集1小时正向阈值，并发出警告。若其他站缺少固定阈值，才使用事件正值分位数，仅作描述性比较。阈值来源在CSV中逐行记录。

从 `projects/xiamen_short_term/short_term_forecast` 运行：

```powershell
python -m src.xiamen_forecast.replot_rollout_typhoon_events
```

程序拒绝覆盖已存在的目标目录；重绘时指定新的 `--output-dir`。全部结果是已知未来ERA5强迫下的追加历史回算，不是实时预报或新的盲测。
