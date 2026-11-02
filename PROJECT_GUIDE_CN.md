# 中国沿海短期风暴增水预测项目说明

- 更新日期：2026-11-02
- 当前工作分支：`codex/china-multistation`
- GitHub：`jiaqianjiaqianjiang-svg/storm_surge`

## 1. 项目目标

本项目研究中国沿海验潮站的小时级短期风暴增水预测。核心问题是：利用过去 24 小时的非潮增水和 ERA5 大气场，预测未来 1—72 小时增水过程，并提高长时效递归稳定性和强增水峰值预测能力。

当前研究重点包括：

1. 建立可复用于不同中国沿海站点的数据准备、训练和评价流程；
2. 比较 Persistence、Ridge、CNN-GRU 与 CNN-GRU Rollout-6；
3. 检查 WMSE、AWMSE、WHuber 等极值损失是否改善强增水预测；
4. 使用真实历史台风案例评价峰高、峰时和长提前量误差；
5. 在不破坏时间隔离的前提下扩展汕尾及更多中国沿海站点。

当前使用未来 ERA5 再分析场作为气象强迫，因此研究结果属于**历史滚动回算**。只有未来替换为实际发布时可得的气象预报，并保存发布时刻和当时可得观测后，才能称为实时业务预报。

## 2. 当前站点与数据划分

| 站点 | 坐标 | 训练期 | 验证期 | 测试期 | 状态 |
|---|---|---|---|---|---|
| 厦门 | 24.450°N, 118.067°E | 1970—1995 | 1996 | 1997 | 全流程完成 |
| 连云港 | 34.750°N, 119.417°E | 1975—1995 | 1996 | 1997 | 全流程完成 |
| 北海 | 21.483°N, 109.083°E | 1975—1995 | 1996 | 1997 | 全流程完成 |
| 汕尾 | 由站点配置和真实文件审计确定 | 待共同覆盖审计 | 计划 1996 | 计划 1997 | 代码入口完成，真实实验待运行 |

时间隔离规则：

- UTide 潮汐参数和标准化参数只使用训练期拟合；
- 1996 用于模型选择、早停和损失方案验证；
- 1997 只用于锁定方案后的全年评价和追加历史案例分析；
- 不使用 1997 重新调整损失参数、模型结构或早停策略；
- 新的独立台风留出实验必须把完整事件、24 小时历史输入和最长 72 小时预测缓冲全部排除在训练之外。

## 3. 数据来源与标签

### 3.1 验潮与非潮增水

验潮记录来自 GESLA。程序读取时间、潮位和质量标记，执行质量控制后，使用训练期观测拟合 UTide 调和参数，再计算：

```text
非潮增水 = 观测潮位 - 天文潮位
```

无效质量记录和缺测小时不参与训练与评价。程序不会为了保持样本数而随意填补极端值或跨长缺口插值。

### 3.2 大气输入

大气资料来自 ERA5，变量顺序固定为：

```text
U10, V10, MSL
```

每个样本使用站点附近 40×40 网格、过去 24 小时气象场和过去 24 小时增水历史，目标是下一小时增水。递归预测时，第一步之后只能回填模型自己的预测增水；未来真实增水不得作为输入。

### 3.3 处理后数据

各站数据相互隔离：

```text
projects/xiamen_short_term/short_term_forecast/outputs/processed/<station>/aligned_dataset/
├── atmosphere.npy
├── surge.npy
├── time.npy
└── dataset_metadata.json
```

大型数组、原始 NetCDF 和模型权重均由 `.gitignore` 排除。

## 4. 仓库结构

```text
storm_surge_clean/
├── README.md
├── PROJECT_GUIDE_CN.md
├── projects/
│   ├── xiamen_short_term/
│   │   └── short_term_forecast/       中国沿海短期预测正式代码
│   ├── xiamen_reconstruction/         早期日最大增水论文复现
│   └── caribbean_forecast/            暂停推进的加勒比历史项目
├── reports/
│   ├── experiment_results/            小型指标、图和研究报告
│   └── project_documents/             Word、PPT等项目材料
├── references/original_paper_code/    原论文代码，只读参考
├── tools/                              跨站比较和实时资料工具
└── archive/                            归档目录
```

虽然正式代码目录仍名为 `xiamen_short_term`，但其中 `src/xiamen_forecast/` 已经是厦门、连云港、北海、汕尾共用的多站点模块。不要因目录名称继续复制第二套连云港、北海或汕尾项目。

## 5. 核心代码入口

位置：`projects/xiamen_short_term/short_term_forecast/`

| 文件 | 功能 |
|---|---|
| `src/xiamen_forecast/station_config.py` | 集中管理站点、坐标、年份、数据和输出路径 |
| `prepare_xiamen.py` | GESLA、UTide、ERA5 对齐及数组生成 |
| `audit_station_sources.py` | 新站源文件覆盖、单位、坐标和变量审计 |
| `audit_prepared_dataset.py` | 处理后数据、样本量和泄漏风险审计 |
| `train_xiamen.py` | 一步基线和神经网络训练 |
| `train_rollout_temporal.py` | CNN-GRU 六步递归微调 |
| `rolling_diagnostics.py` | 1—72 小时递归预测和统一指标 |
| `extreme_losses.py` | MSE、WMSE、AWMSE、WHuber 损失实现 |
| `run_three_station_extreme_loss_suite.py` | 三站极值损失实验调度与恢复 |
| `summarize_three_station_extreme_losses.py` | 三站 1996 统一验证汇总 |
| `typhoon_events.py` | IBTrACS 筛选、事件窗口和峰值指标 |
| `run_typhoon_events.py` | 三次固定起报的 72 小时台风历史回算 |
| `prepare_typhoon_mse_whuber_comparison.py` | 合并四场已审核事件，准备 MSE/WHuber 对照 |
| `run_station_workflow.py` | 新站点标准端到端入口 |
| `journal_figures/` | 学术图件 |

项目内的重要说明：

- `README.md`：项目内环境、预处理、训练与评价入口；
- `THREE_STATION_EXTREME_LOSS_SUITE.md`：极值损失统一实验；
- `HISTORICAL_TYPHOON_EVENTS.md`：台风事件筛选和回算；
- `SHANWEI_WORKFLOW.md`：汕尾源数据审计与标准流程。

## 6. 模型体系

### 6.1 基线与核心模型

| 模型 | 作用 |
|---|---|
| Persistence | 使用最后一个历史增水值，检验强自相关基线 |
| Ridge | 使用历史增水和大气统计特征的线性基线 |
| CNN-GRU | 每小时 ERA5 场经 CNN 编码，再由 GRU 结合增水历史建模 |
| CNN-GRU Rollout-6 | 从一步 CNN-GRU 继续微调，同时优化连续 6 步递归误差 |

CNN、CNN-LSTM、TCN、Transformer、Surge-MLP、ERA5-only CNN 已在前三站完成正式比较或消融。当前后续实验不再扩大模型数量，重点是跨站稳定性、极值损失和真实事件表现。

### 6.2 极值损失矩阵

统一比较：

| 方案 | 一步训练 | Rollout-6微调 |
|---|---|---|
| MSE/MSE | MSE | MSE |
| WMSE/MSE | WMSE | MSE |
| WMSE/WMSE | WMSE | WMSE |
| AWMSE/AWMSE | AWMSE | AWMSE |
| WHuber/WHuber | WHuber | WHuber |

极值阈值只由各站训练期目标计算。训练损失使用的 P90/P99 阈值和评价时重新计算的 P99 样本阈值必须分别记录，不能混用。

## 7. 已完成结果

### 7.1 三站核心预测

1997 全年统一评价：

| 站点 | 一步最佳模型 | 一步RMSE | 一步MAE | Pearson r | CNN-GRU 72h | Rollout-6 72h |
|---|---|---:|---:|---:|---:|---:|
| 厦门 | CNN-LSTM | 3.47 cm | 2.67 cm | 0.986 | 12.14 cm | **10.00 cm** |
| 连云港 | CNN-LSTM | 5.49 cm | 3.99 cm | 0.979 | 19.96 cm | **15.71 cm** |
| 北海 | CNN | 3.37 cm | 2.47 cm | 0.983 | 17.16 cm | **14.49 cm** |

Rollout-6 对三站 72 小时误差的降幅分别约为 17.6%、21.3% 和 15.5%，说明多步递归训练的改善具有跨站一致性，但各站绝对误差尺度明显不同。

### 7.2 三站极值损失验证

1996 统一验证结论：

- 厦门 72 h 整体 RMSE 最低为 MSE/MSE（9.85 cm），P99 RMSE 最低为 WMSE/WMSE（20.96 cm）；
- 连云港整体 RMSE 最低为 WMSE/MSE（15.86 cm），P99 RMSE 最低为 AWMSE/AWMSE（18.77 cm）；
- 北海整体 RMSE 和 P99 RMSE 均以 WHuber/WHuber 最低，分别为 14.66 cm 和 28.25 cm；
- 极值损失的收益存在站点差异，不能仅按一个台风或一个 P99 指标确定跨站统一最优方案。

完整结果：`reports/experiment_results/three_station_extreme_loss_validation_1996_seed42/`。

### 7.3 历史台风案例

已经审核并回算四场 1997 事件：

| 站点 | 台风 | 说明 |
|---|---|---|
| 北海 | ZITA | 与 CASS 窗口重叠，归因属于初步判断 |
| 连云港 | WINNIE | 长提前量和峰值预测存在明显失败案例 |
| 连云港 | VICTOR | 序列存在约 12 h 周期振荡，不能直接认定为残余潮汐 |
| 厦门 | AMBER | 与 CASS 构成复合影响过程，按带限制方式纳入 |

每场使用 `t0−72h`、`t0−48h`、`t0−24h` 三次固定起报，每次单独预测未来 72 小时。三个重叠起报不是三场独立事件，不能用于虚增样本量或直接计算独立案例置信区间。

当前已完成旧模型的四场汇总，并已接入三站统一 MSE/MSE 与 WHuber/WHuber 权重检查和推理流程。新的真实对照需要在实验室电脑读取被 Git 忽略的 checkpoint 后运行。

## 8. 结果目录

`reports/experiment_results/` 中与中国沿海主线直接相关的目录：

```text
xiamen_short_term_1996_seed42/
xiamen_short_term_1997_seed42/
lianyungang_short_term_1996_seed42/
lianyungang_short_term_1997_seed42/
beihai_short_term_1996_seed42/
beihai_short_term_1997_seed42/
china_three_station_comparison_1997_seed42/
three_station_extreme_loss_validation_1996_seed42/
china_three_station_event_distribution_v1/
historical_typhoon_four_case_review/
typhoon_events_1997_seed42_compact/
typhoon_events_1997_amber_victor_seed42/
typhoon_events_1997_mse_whuber_comparison_seed42/
independent_typhoon_holdout_plan_v1/
```

这些目录只保存 CSV、JSON、Markdown 和关键 PNG。完整模型和大型结果保留在实验室电脑项目目录内。

## 9. 电脑与职责

### MacBook Air

```text
/Users/jjq/Documents/storm_surge/storm_surge_clean
```

用途：代码修改、自动化测试、文档、Git、精简结果审核。Mac 不保存实验室 F 盘的中国沿海原始数据，不在本机假装执行真实训练。

### 实验室 Windows 电脑

```text
仓库：H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge
项目：H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast
Conda：jjq
GPU：NVIDIA GeForce RTX 4090 D
```

用途：中国沿海原始数据处理、模型训练、滚动评价和台风事件回算。原始数据位于 `F:\GESLA` 和 `F:\ERA5-NEW`，全部只读。

### 个人 Windows 台式机

后续用于实验室数据备份和额外计算。大型资料应通过硬盘或局域网复制并校验，不通过 GitHub 传输。

个人 Windows 笔记本上保留的加勒比实验目前暂停，不属于近期中国沿海实验调度。

## 10. 实验室工作规范

每次运行前：

```powershell
conda activate jjq
$repo = "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
Set-Location -LiteralPath $repo
git status --short
git fetch origin codex/china-multistation
git merge --ff-only origin/codex/china-multistation
Set-Location -LiteralPath (Join-Path $repo "projects\xiamen_short_term\short_term_forecast")
python -m compileall -q src
python -m pytest tests -q
```

原则：

1. 先检查分支和本地修改，再同步；
2. 不使用 `git add .`；
3. 不上传 `outputs/`、`models/`、NPY、NetCDF 和 GESLA 原始文件；
4. 每个站点、年份和实验方案使用独立目录，不覆盖旧结果；
5. 正式训练结束后使用导出脚本生成 Git-safe 小型结果包；
6. 1996 结果用于选型，方案固定后才读取 1997；
7. 任何缺失 checkpoint、站点身份或 scaler 冲突都应停止，不能自动替换。

## 11. 当前任务状态

### 已完成

- 厦门、连云港、北海预处理、质量审计和时间划分；
- 三站基线、核心深度模型和 1996/1997 评价；
- 三站 CNN-GRU Rollout-6 和 1—72 小时滚动诊断；
- 三站五种极值损失组合的 1996 统一验证；
- 三站台风候选和极端增水样本分布统计；
- 四场 1997 历史台风的审核、固定起报回算和汇总分析；
- 三站 MSE/MSE 与 WHuber/WHuber 台风对照代码和审核清单；
- 汕尾统一配置、源数据审计入口和标准工作流；
- 合成数据测试、路径隔离和数据泄漏检查。

### 等待实验室运行

- 四场台风 MSE/MSE 与 WHuber/WHuber 统一对照；
- 汕尾真实 GESLA/ERA5 共同覆盖审计；
- 审计通过后的汕尾核心模型、1996验证、1997测试和台风事件流程。

### 后续研究

- 扩充更多中国沿海站点，而不是继续增加大量新模型；
- 建立 3—5 场真正未参与模型开发的台风留出实验；
- 固定跨站评价规则后分析站点差异与极值预测机制；
- 最后再评估将未来 ERA5 替换为实际发布 GFS 的实时预报可行性。

## 12. 非当前主线

`projects/caribbean_forecast/` 中的 Prickly Bay 实验保留用于历史追溯，但当前暂停推进，不参与近期实验排期和中国沿海模型选择。`projects/xiamen_reconstruction/` 是早期 ERA-20C 日最大增水论文复现，也不属于当前小时级短时预测主线。
