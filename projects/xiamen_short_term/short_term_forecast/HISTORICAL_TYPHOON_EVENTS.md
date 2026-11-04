# 测试期历史台风事件回算

本流程使用已经锁定的模型，在各站既定测试期内筛选真实台风并开展事件级72小时回算。它用于补充全年平均指标，重点检查峰值高度、峰值时间、低估和长提前量误差。

当前不执行训练期独立台风留出重训。台风事件结果不能用于返回调整模型、损失参数或早停策略。

## 1. 实验性质

- 观测来自各站处理后的非潮增水序列。
- 台风路径来自IBTrACS西北太平洋最佳路径。
- 未来气象使用ERA5再分析真值，因此属于历史回算，不是实时业务预报。
- 候选事件必须经过人工核验；路径距离或P99增水本身不能证明台风归因。
- 若测试期没有合格事件，应如实报告，不用普通增水过程代替台风。

## 2. 候选筛选与人工核验

对每个站点：

1. 按站点与台风路径距离初筛候选，500 km只作为参考范围。
2. 检查最佳路径、局地风向、风速、气压和实测增水过程。
3. 检查起报前24小时历史、未来72小时ERA5和观测是否完整。
4. 在 `candidate_events.csv` 中保留事件ID和`t0`，填写 `review_note`。
5. 只有证据充分且数据完整的事件才设为 `review_decision=yes`。

存在台风窗口重叠、复合影响或周期振荡时必须在说明中保留限制，不能把整个增水过程简单归因给单一台风。

## 3. 固定起报规则

默认使用相对最近路径时刻`t0`的三次固定起报：

- `t0−72h`
- `t0−48h`
- `t0−24h`

每次独立预测未来72小时。模型只使用起报前真实增水，起报后递归回填模型自身预测。三次起报不能拼接成一条连续曲线，也不能视为三场独立台风。

## 4. 实验室命令

从项目目录运行。将 `<station>`、`<test_year>` 和结果目录替换为当前站点配置。

```powershell
$events = "..\..\..\reports\experiment_results\<station>_typhoon_events_<test_year>_seed42"

python -m src.xiamen_forecast.run_typhoon_events scan `
  --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" `
  --stations <station> `
  --year <test_year> `
  --seed 42 `
  --results-dir $events
```

人工完成候选审核后运行：

```powershell
python -m src.xiamen_forecast.run_typhoon_events run `
  --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" `
  --stations <station> `
  --year <test_year> `
  --seed 42 `
  --results-dir $events `
  --device cuda
```

程序会复查最佳路径文件哈希、数据完整性、模型身份、输入长度和scaler。任何检查失败都应停止对应事件，不能自动补训、替换权重或覆盖旧结果。

## 5. 模型比较

基础参考：

- Persistence
- Ridge

神经网络主模型：

- CNN-GRU
- CNN-GRU Rollout-6

极值损失模型只有在该站已使用验证期完成选择、checkpoint身份检查通过时才加入。没有对应权重时不为单次台风临时补训。

## 6. 输出与指标

每场事件保存：

```text
forecasts_72h.csv
event_metrics.csv
exact_lead_metrics.csv
experiment_config.json
figure1_best_track.png
figure2_fixed_origin_forecasts.png
figure3_event_errors.png
研究报告.md
```

指标口径：

- `event_metrics.csv`：单次起报后前24/48/72小时窗口内的RMSE、MAE、Bias、峰高和峰时误差。
- `exact_lead_metrics.csv`：各固定起报在恰好提前24/48/72小时处的描述性误差。
- 峰值必须在同一起报、同一预先定义窗口中比较。

报告应同时展示成功和失败案例，说明峰值低估、错误局部峰、峰时偏移及长时效误差累积。

## 7. 已有参考结果

仓库已经保存原有四场1997年追加历史回算、MSE/WHuber对照，以及坎门WINNIE事件结果：

```text
reports/experiment_results/historical_typhoon_four_case_review/
reports/experiment_results/typhoon_events_1997_mse_whuber_comparison_seed42/
reports/experiment_results/kanmen_1996_1997_seed42_compact/typhoon_1997/
```

这些案例是在1997全年结果已经查看后的补充分析，不是新的盲测。它们用于验证分析方法和展示模型差异，不用于重新选择训练方案。

## 8. 上传范围

只提交事件清单、配置、指标CSV、关键PNG和Markdown报告。不上传IBTrACS原始整文件、GESLA、ERA5、NPY、模型权重或大型中间结果。提交前显式指定结果目录，不使用 `git add .`。
