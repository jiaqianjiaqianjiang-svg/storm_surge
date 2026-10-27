# 三站极值损失统一实验：实验室执行说明

本流程只用于厦门、连云港、北海的1996验证集方案比较。固定模型为CNN-GRU，固定矩阵为MSE/MSE、WMSE/MSE、WMSE/WMSE、AWMSE/AWMSE、WHuber/WHuber。斜线前是一步训练损失，斜线后是Rollout-6微调损失。

程序默认不把1997载入训练或模型选择。训练阈值只使用对应站点训练期目标；1996仅用于验证和早停。1997只能在验证方案书面固定后单独评价。

## 已有证据与边界

- Git精简包证明连云港已经产生四种一步损失和五种Rollout-6的1996结果。
- Git精简包证明厦门、北海已有MSE一步CNN-GRU及MSE/MSE Rollout-6的1996结果。
- Git不保存模型权重、处理数组和完整预测文件。Mac上看不到这些文件，不代表实验室没有；是否可复用必须以实验室状态审计为准。
- 完整旧结果如果缺少后来新增的形式化参数字段，会标记为`legacy_metadata`，不会自动重训。人工核对后可显式使用`--accept-legacy-metadata`。
- 配置冲突会标记为`configuration_mismatch`并停止，程序不会覆盖。已有目录只是不完整时，也必须显式给出`--restart-incomplete`才会重新开始该任务。

## 1. 安全同步仓库

在实验室PowerShell中逐段执行：

```powershell
conda activate jjq
$repo = 'H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge'
Set-Location -LiteralPath $repo
git branch --show-current
git status --short
if ((git branch --show-current).Trim() -ne 'codex/china-multistation') { throw '请先切换到 codex/china-multistation' }
git fetch origin codex/china-multistation
if ($LASTEXITCODE -ne 0) { throw '获取远程更新失败' }
$remotePaths = @(git diff --name-only HEAD..origin/codex/china-multistation)
$localPaths = @(@(git diff --name-only) + @(git diff --cached --name-only) | Sort-Object -Unique)
$clashes = @($remotePaths | Where-Object { $localPaths -contains $_ })
if ($clashes.Count -gt 0) { $clashes; throw '远程更新与本机已跟踪修改冲突，请停止' }
git merge --ff-only origin/codex/china-multistation
if ($LASTEXITCODE -ne 0) { throw '安全同步失败，不要清理工作区' }
git log -1 --oneline
git status --short
```

`models/`、`outputs/`通常被Git忽略，安全同步不会删除它们。不要使用`git reset --hard`、`git clean`或`git add .`。

## 2. 编译与小型测试

```powershell
$project = Join-Path $repo 'projects\xiamen_short_term\short_term_forecast'
Set-Location -LiteralPath $project
python -m compileall -q src
if ($LASTEXITCODE -ne 0) { throw '编译检查失败' }
python -m pytest tests/test_three_station_extreme_loss_suite.py tests/test_extreme_losses.py tests/test_xiamen_v2.py tests/test_multistation.py tests/test_new_station_workflow.py -q
if ($LASTEXITCODE -ne 0) { throw '测试失败，暂不训练' }
```

## 3. 审计实验室已有文件

先只生成状态表，不训练：

```powershell
$statusCsv = Join-Path $repo 'reports\experiment_results\three_station_extreme_loss_suite_plan\laboratory_extreme_loss_status.csv'
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage status --stations xiamen lianyungang beihai --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --status-csv $statusCsv
Import-Csv -LiteralPath $statusCsv | Format-Table station,scheme,one_step_status_on_this_computer,rollout_status_on_this_computer,one_step_1996_compact_evidence,rollout_1996_compact_evidence -AutoSize
```

状态含义：

- `compatible`：权重、预测、配置和必要元数据齐全且与正式参数一致，可以跳过训练。
- `legacy_metadata`：文件齐全，但旧元数据没有记录全部正式参数；先看对应JSON和旧日志，再决定是否显式复用。
- `absent`：本机没有该实验目录，需要训练。
- `incomplete`：目录存在但必要文件缺失；先看日志，不能静默覆盖。
- `configuration_mismatch`：已记录参数与本轮正式矩阵冲突，必须停止并另行归档处理。

检查旧元数据的示例：

```powershell
Get-Content -LiteralPath 'models\lianyungang\formal_seed42\cnn_gru_extreme_wmse\training_config.json'
Get-Content -LiteralPath 'models\lianyungang\formal_seed42\cnn_gru_extreme_wmse_rollout6_wmse\training_metadata.json'
Get-Content -LiteralPath 'models\lianyungang\formal_seed42\cnn_gru_extreme_wmse_rollout6_wmse\extreme_loss_config.json'
```

## 4. Dry-run查看完整任务

```powershell
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage all --stations xiamen lianyungang beihai --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --num-workers 0 --dry-run
```

Dry-run会打印每个实际训练命令。每条底层命令均显式包含站点、一步/rollout损失、训练/验证年份、数据目录、输出目录以及`--base-checkpoint`。它不会运行训练。若旧结果为`legacy_metadata`，会显示“blocked”，而不是假定可复用。

## 5. 按站补齐缺失训练

只有状态表和旧日志确认以后再执行。对于确认来自原正式流程但元数据字段较旧的完整权重，加入`--accept-legacy-metadata`。以下命令会复用兼容结果，只补实际缺失项：

```powershell
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage train --stations lianyungang --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --num-workers 0 --accept-legacy-metadata
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage train --stations xiamen --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --num-workers 0 --accept-legacy-metadata
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage train --stations beihai --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --num-workers 0 --accept-legacy-metadata
```

推荐先执行连云港：正常情况下应全部跳过，借此验证目录映射正确。然后依次运行厦门、北海，便于中断和检查。任何任务失败都会停止其后依赖项；已完成站点不受影响。日志保存在：

`outputs/experiments/three_station_extreme_loss_suite_seed42/logs/`

状态保存在：

`outputs/experiments/three_station_extreme_loss_suite_seed42/suite_state.json`

底层正式参数固定为：一步50 epoch、patience 8、batch 32、学习率0.001；Rollout-6为20 epoch、patience 5、teacher forcing 5 epoch、batch 256、embedding batch 1024、学习率0.0001；alpha 2、beta 1、Huber delta 5 cm、seed 42。

如果某目录显示`incomplete`，先查看该任务日志。只有确认要从头重新开始该目录时，才在原命令末尾加`--restart-incomplete`。该选项不允许绕过配置冲突。

## 6. 生成三站1996统一验证

三个站的五种权重均准备好以后，逐站生成共同时间戳的一步比较和共同起报的72小时滚动比较：

```powershell
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage evaluate --stations lianyungang --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --accept-legacy-metadata
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage evaluate --stations xiamen --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --accept-legacy-metadata
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage evaluate --stations beihai --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42 --device cuda --accept-legacy-metadata
python -m src.xiamen_forecast.run_three_station_extreme_loss_suite --stage summarize --stations xiamen lianyungang beihai --schemes mse/mse wmse/mse wmse/wmse awmse/awmse whuber/whuber --seed 42
```

每站新建且不覆盖旧结果：

- `outputs/experiments/<station>/extreme_loss_suite_one_step_1996_seed42/`
- `outputs/experiments/<station>/extreme_loss_suite_rolling_1996_seed42/`

跨站精简结果：

`reports/experiment_results/three_station_extreme_loss_validation_1996_seed42/`

其中包含任务要求的两个CSV、两张PNG、中文报告和实验元数据。滚动CSV包含整体RMSE/MAE/Bias/Pearson r、P90/P95/P99误差与低估比例、独立正向峰值的峰高误差和峰时误差。

## 7. 固定方案后再评价1997

先审阅1996跨站报告，逐站书面固定一种重点方案和统一选型规则。不要因为1997结果改变alpha、beta、阈值、早停或模型结构。固定后可用下面的显式模板评价某站选定权重；将示例路径替换成该站实际锁定方案：

```powershell
$station = 'xiamen'
$selectedCheckpoint = 'models\xiamen\formal_seed42\cnn_gru_extreme_wmse_rollout6_wmse\best_model.pth'
$testOutput = 'outputs\experiments\xiamen\fixed_extreme_loss_test_1997_seed42'
python -m src.xiamen_forecast.rolling_diagnostics --station $station --train-start-year 1970 --evaluation-year 1997 --split test --models cnn_gru --input-steps 24 --max-lead 72 --seed 42 --device cuda --dataset-path 'outputs\processed\xiamen\aligned_dataset' --checkpoint-root 'models\xiamen\formal_seed42' --ridge-path 'models\xiamen\baselines\ridge_pipeline.joblib' --extra-checkpoint "selected_extreme=$selectedCheckpoint" --output-dir $testOutput
```

连云港、北海必须改为各自的站点、训练起始年、数据、模型和输出目录。1997四场历史台风结果只能作为固定方案的追加案例分析复用，不能包装成新的盲测。

## 8. Git上传范围

真实训练完成后，先检查跨站精简包，再只暂存小型报告目录：

```powershell
Set-Location -LiteralPath $repo
git status --short
git add -- 'reports/experiment_results/three_station_extreme_loss_validation_1996_seed42'
git diff --cached --stat
git commit -m 'Archive three-station extreme-loss validation results'
git push origin codex/china-multistation
```

不要上传`models/`、`outputs/`、NPY、NetCDF、GESLA原始文件或训练权重。

## 暂缓事项

- 独立台风留出目前只有人工核验候选和排除窗口计划，不等于完成了UTide、标准化、Ridge和神经网络全链路重训；本轮不启动。
- 汕尾已有单独的源数据审计和核心流程，本轮不重复开发，也不对汕尾搜索全部极值损失。
