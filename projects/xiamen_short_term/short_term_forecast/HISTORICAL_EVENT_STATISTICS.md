# 中国沿海三站台风候选与极端增水统计

本程序只读已处理的 `time.npy`、`surge.npy`、`atmosphere.npy` 和实验室已有的 IBTrACS 西北太平洋最佳路径 CSV。不会训练模型，也不会修改或复制 ERA5、GESLA、NPY、checkpoint。Mac 上没有实验室数据时，只能运行合成测试，不能声称得到真实统计。

## 实验室 PowerShell 操作

先确保已把包含本程序的 `codex/china-multistation` 分支同步到实验室，并在 PowerShell 中执行：

```powershell
conda activate jjq
Set-Location -LiteralPath "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast"
Test-Path -LiteralPath "outputs\best_tracks\ibtracs.WP.list.v04r01.csv"
Test-Path -LiteralPath "outputs\processed\xiamen\aligned_dataset\time.npy"
Test-Path -LiteralPath "outputs\processed\lianyungang\aligned_dataset\time.npy"
Test-Path -LiteralPath "outputs\processed\beihai\aligned_dataset\time.npy"
python -m src.xiamen_forecast.build_station_event_statistics --stations xiamen lianyungang beihai --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv"
```

前面四个检查都应返回 `True`。程序还会检查各站的 `surge.npy`、`atmosphere.npy`、站点元数据和配置年份端点；缺文件会列出路径后停止。运行中断后可直接重跑；只有完整成功后才会出现正式结果目录。已生成正式结果时默认拒绝覆盖，复查请指定新的 `--output-dir`。

结果位于仓库根目录的 `reports/experiment_results/china_three_station_event_distribution_v1/`。检查：

```powershell
Get-ChildItem -LiteralPath "..\..\..\reports\experiment_results\china_three_station_event_distribution_v1"
Import-Csv "..\..\..\reports\experiment_results\china_three_station_event_distribution_v1\station_typhoon_counts_by_split.csv" | Format-Table -AutoSize
Get-Content -LiteralPath "..\..\..\reports\experiment_results\china_three_station_event_distribution_v1\statistics_summary.md" -Encoding UTF8
```

预期有五个 CSV：候选数、候选清单、逐年极端分布、24小时合并间隔下的独立 P99 峰簇、候选窗口与峰簇交叉表；另有三张 PNG、训练阈值 JSON 和两份 Markdown。请检查样本数、缺测率、跨阶段标记及阈值后，只提交该小型结果目录，不上传原始数据和模型权重。

P90/P95/P99 仅用训练年份中满足原一步训练有效性规则的目标增水拟合（包含正负值，正向统计时要求增水大于零）；验证和测试沿用这些固定阈值。逐年超阈值比例以有效观测小时为分母。500 km 是路径初筛，不等于台风影响的因果确认；已有1997年四场只是人工初步纳入。若当前 IBTrACS 文件的 SHA256 与归档案例不同，不继承这四场的初步状态。跨阶段候选单列，不当作同阶段完整事件。最快上涨统计使用另一套训练期 1 小时正向增量阈值，不能与最大增水混为一谈。

四场1997年已归档回算的跨案例汇总可在有代码和精简 CSV 的电脑运行，不需要实验室大数据：

```powershell
python -m src.xiamen_forecast.summarize_typhoon_four_cases
```

输出到 `reports/experiment_results/historical_typhoon_four_case_review/`。当前仓库已附有这份小型汇总；重复运行时程序拒绝覆盖。
