# 中国沿海小时级短时风暴增水预测

本项目使用验潮站非潮增水和ERA5气象场，预测中国沿海不同站点未来1—72小时的增水过程。目标是建立一套可持续扩展到更多站点的数据审计、模型训练、滚动评价和历史台风测试流程，而不是只服务某三个站点或某一个站点。

代码包仍名为 `src/xiamen_forecast/`，这是为了兼容早期命令；其中核心流程已经改造成多站点共用。加勒比项目不属于当前研究主线。

## 1. 当前状态

- 已有参考站点完成数据处理、1996验证、1997全年测试和1—72小时滚动评价。
- 已完成CNN-GRU、Rollout-6及多种极值损失的跨站验证。
- 已完成五个1997年历史台风站点事件案例的固定起报回算，以及MSE/WHuber对照。
- 坎门已完成数据审计、1996验证、1997全年测试和WINNIE事件回算。
- 后续站点继续使用统一流程，实际训练、验证和测试年份由各站数据共同覆盖范围决定。

当前主模型分工：

- `CNN-GRU`：一步预测核心模型。
- `CNN-GRU Rollout-6`：降低长时效递归误差的主要方案。
- `WHuber/WHuber`：部分强增水事件的补充方案，其收益具有事件和站点差异。

## 2. 最重要结果

已完成参考站点的1997全年滚动评价：

| 参考站点 | 一步最佳RMSE | CNN-GRU 72h | Rollout-6 72h | 降幅 |
|---|---:|---:|---:|---:|
| 厦门 | 3.47 cm | 12.14 cm | **10.00 cm** | 17.6% |
| 连云港 | 5.49 cm | 19.96 cm | **15.71 cm** | 21.3% |
| 北海 | 3.37 cm | 17.16 cm | **14.49 cm** | 15.5% |
| 坎门 | 5.03 cm | 11.60 cm | **9.94 cm** | 14.3% |

已有历史台风站点事件案例的72小时窗口平均RMSE：

| 事件 | CNN-GRU | MSE Rollout-6 | WHuber Rollout-6 |
|---|---:|---:|---:|
| ZITA（北海） | 15.33 cm | **14.85 cm** | 16.39 cm |
| VICTOR（连云港） | 19.82 cm | 16.07 cm | **14.84 cm** |
| WINNIE（连云港） | 26.20 cm | **25.07 cm** | 27.30 cm |
| AMBER（厦门） | 13.09 cm | **12.29 cm** | 12.91 cm |
| WINNIE（坎门） | 41.64 cm | **20.93 cm** | 23.15 cm |

主要结论：Rollout-6对长时效误差的改善较稳定；极值损失的收益存在站点和事件差异。坎门1997全年WHuber Rollout-6的72小时RMSE略低于MSE Rollout-6，但1996验证期MSE方案更优，不能依据测试期重新选型。MSE Rollout-6继续作为当前主方案，WHuber保留为强增水补充。

## 3. 当前研究边界

1. 每个站点独立计算UTide、标准化参数和模型权重。
2. 模型选择只使用验证期，方案锁定后才评价测试期。
3. 当前台风测试只从各站既定测试期划出真实事件做事件级评价。
4. 暂不开展训练期独立台风留出重训，旧的候选计划不属于当前执行路线。
5. 递归预测从第二步起只回填模型自身预测，不使用未来真实增水。
6. 未来气象使用ERA5再分析真值，因此结果属于历史回算，不是实时业务预报。

## 4. 统一输入与评价

标准输入包括：

- 过去24小时非潮增水；
- 过去及未来有效时次的U10、V10、MSL气象场；
- 站点附近40×40 ERA5网格。

主要指标包括：

- RMSE、MAE、Bias、Pearson r、RRMSE；
- 1、3、6、12、24、48、72小时滚动误差；
- P90、P95、P99强增水误差和低估比例；
- 历史台风过程的峰高误差、峰时误差及不同起报稳定性。

训练损失阈值只由训练期计算。评价期P99分组阈值与训练损失阈值必须分开记录。

## 5. 模型体系

通用核心模型：

- Persistence
- Ridge
- CNN-GRU
- CNN-GRU Rollout-6

已经完成过的结构与消融还包括Fusion CNN、CNN-LSTM、TCN、Transformer、Surge-MLP、ERA5-only CNN，以及WMSE、AWMSE、WHuber等损失。新增站点默认先运行核心模型，不再为每个站重复进行全部模型搜索。

## 6. 目录结构

```text
short_term_forecast/
├── src/xiamen_forecast/       # 数据、训练、滚动预测和台风事件
├── src/short_term_forecast/   # 期刊绘图与公用指标
├── tests/                     # 自动化测试
├── outputs/                   # 本机处理数据和完整实验输出
├── models/                    # 本机模型权重
└── README.md                  # 项目入口

仓库根目录/
└── reports/experiment_results/ # 可提交的小型指标、图件和报告
```

`outputs/`、`models/`、原始GESLA、NetCDF和大型NPY不上传GitHub。

## 7. 实验室常用操作

### 同步与测试

```powershell
conda activate jjq
Set-Location -LiteralPath "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git checkout codex/china-multistation
git pull --ff-only origin codex/china-multistation
Set-Location -LiteralPath "projects\xiamen_short_term\short_term_forecast"
python -m compileall -q src
python -m pytest tests -q
```

### 已注册站点标准流程

将 `<station>` 替换为经过源数据审计并已注册的站点ID：

```powershell
python -m src.xiamen_forecast.run_station_workflow prepare --station <station>
python -m src.xiamen_forecast.run_station_workflow validation --station <station> --seed 42 --device cuda --language zh
python -m src.xiamen_forecast.run_station_workflow lock --station <station> --seed 42
python -m src.xiamen_forecast.run_station_workflow test --station <station> --seed 42 --device cuda --language zh
```

完整的新站接入、测试期台风事件和输出规范见 [CHINA_COASTAL_WORKFLOW.md](CHINA_COASTAL_WORKFLOW.md)。

## 8. 历史台风结果解释

- 候选事件必须结合IBTrACS路径、风场、气压和实测增水人工核验。
- 同一台风的多个固定起报属于同一个事件，不能当成多个独立台风样本。
- 每次起报单独预测72小时，不把多次起报曲线拼接成虚假的连续预报。
- `event_metrics.csv`记录起报后固定窗口内的累计指标。
- `exact_lead_metrics.csv`记录恰好提前24/48/72小时处的误差。
- 已有1997案例属于全年结果已查看后的追加分析，不是新的盲测。

## 9. 专题文档

项目根目录只保留三份仍在使用的专题说明：

| 文档 | 内容 |
|---|---|
| [CHINA_COASTAL_WORKFLOW.md](CHINA_COASTAL_WORKFLOW.md) | 中国沿海新增站点统一实验流程 |
| [HISTORICAL_TYPHOON_EVENTS.md](HISTORICAL_TYPHOON_EVENTS.md) | 测试期历史台风筛选与固定起报回算 |
| [JOURNAL_FIGURES_README.md](JOURNAL_FIGURES_README.md) | 论文和汇报图生成 |

已完成的阶段性统计、极值损失审计和具体站点操作记录保存在对应的 `reports/experiment_results/` 结果包中，不再在项目根目录重复维护说明文件。

## 10. 数据与Git安全

1. 原始数据目录只读，不在其中创建、修改或删除文件。
2. 每站处理数据、模型和结果使用独立目录，不覆盖其他站点。
3. 不上传原始数据、NPY、模型权重或大型逐时预测。
4. 只提交经过检查的小型结果包、代码和文档。
5. 不使用 `git add .`，显式指定提交路径。
6. 不根据测试期和台风事件结果返回调整训练参数。

## 11. 下一步

1. 将坎门结果纳入跨站全年与台风事件汇总。
2. 继续接入中国沿海其他具备GESLA与ERA5共同覆盖的站点。
3. 在各站测试期内筛选真实台风，统一比较全年表现和事件峰值表现。
4. 积累足够跨站、跨事件证据后，再评估实时气象预报接入。
