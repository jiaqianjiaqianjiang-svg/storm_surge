# 中国沿海小时级短时风暴增水预测

本仓库当前主线是中国沿海多站点小时级短时风暴增水预测。项目使用GESLA验潮观测和ERA5大气再分析资料，建立可扩展的数据审计、模型训练、1—72小时滚动评价和测试期历史台风回算流程。

- 工作分支：`codex/china-multistation`
- 正式代码：`projects/xiamen_short_term/short_term_forecast/`
- 项目详细说明：[PROJECT_GUIDE_CN.md](PROJECT_GUIDE_CN.md)
- 实际操作入口：[项目README](projects/xiamen_short_term/short_term_forecast/README.md)

## 研究目标

1. 将统一流程扩展到更多中国沿海验潮站，而不是为每站复制独立项目。
2. 比较Persistence、Ridge、CNN-GRU和CNN-GRU Rollout-6的短时预测能力。
3. 评价普通时段、强增水和真实台风过程中的1—72小时误差。
4. 分析Rollout-6和极值损失对长时效稳定性及峰值预测的影响。

当前未来气象输入使用ERA5再分析真值，因此属于历史回算，不是实时业务预报。

## 当前进展

- 已有多个参考站点完成预处理、验证、全年测试和台风事件分析。
- 已完成CNN-GRU、Rollout-6及MSE、WMSE、AWMSE、WHuber极值损失验证。
- 已完成五个1997年真实历史台风站点事件案例的固定起报回算和MSE/WHuber对照。
- 坎门已完成数据审计、1996验证、1997全年测试和WINNIE事件回算。
- 后续继续向中国沿海其他具备GESLA与ERA5共同覆盖的站点扩展。

已完成结果表明：Rollout-6通常比普通CNN-GRU具有更稳定的长时效表现；WHuber在部分事件上改善峰值，但收益并不跨站、跨事件一致，因此当前作为补充方案。

## 最重要结果

以下站点是已经完成的参考实验，不代表项目最终只包含这些站点。

| 参考站点 | 一步最佳RMSE | CNN-GRU 72h RMSE | Rollout-6 72h RMSE | 72h降幅 |
|---|---:|---:|---:|---:|
| 厦门 | 3.47 cm | 12.14 cm | **10.00 cm** | 17.6% |
| 连云港 | 5.49 cm | 19.96 cm | **15.71 cm** | 21.3% |
| 北海 | 3.37 cm | 17.16 cm | **14.49 cm** | 15.5% |
| 坎门 | 5.03 cm | 11.60 cm | **9.94 cm** | 14.3% |

1996验证表明，极值损失没有一个跨站统一最优方案：厦门整体72小时RMSE以MSE/MSE最低（9.85 cm），连云港以WMSE/MSE最低（15.86 cm），北海以WHuber/WHuber最低（14.66 cm）。这说明损失函数需要结合站点特征评价，不能只看单个P99指标。

已有1997年历史台风站点事件案例的72小时窗口平均RMSE如下：

| 事件 | CNN-GRU | MSE Rollout-6 | WHuber一步 | WHuber Rollout-6 |
|---|---:|---:|---:|---:|
| ZITA（北海） | 15.33 cm | **14.85 cm** | 16.40 cm | 16.39 cm |
| VICTOR（连云港） | 19.82 cm | 16.07 cm | 19.80 cm | **14.84 cm** |
| WINNIE（连云港） | 26.20 cm | **25.07 cm** | 34.74 cm | 27.30 cm |
| AMBER（厦门） | 13.09 cm | **12.29 cm** | 13.83 cm | 12.91 cm |
| WINNIE（坎门） | 41.64 cm | **20.93 cm** | 26.02 cm | 23.15 cm |

MSE Rollout-6在五个站点事件案例中均优于普通CNN-GRU，并在四个案例取得最低过程RMSE；WHuber Rollout-6仅在VICTOR中明显领先。当前因此以MSE Rollout-6作为稳定主方案，WHuber作为事件相关的峰值补充方案。坎门1997全年WHuber Rollout-6的72小时RMSE为9.87 cm，略低于MSE Rollout-6的9.94 cm，但1996验证期MSE Rollout-6更优，因此不能依据测试年改选方案。

## 当前实验边界

- 每站独立计算UTide、标准化参数和模型权重。
- 验证期用于选型，测试期只用于锁定方案后的评价。
- 当前只在各站既定测试期内筛选台风事件进行测试。
- 暂不开展训练期独立台风留出重训。
- 台风事件结果不能用于返回调整模型和损失参数。
- 新站的数据年份由实际GESLA与ERA5共同覆盖决定，不预先写死。

## 仓库结构

```text
storm_surge_clean/
├── projects/xiamen_short_term/short_term_forecast/  # 中国沿海正式代码
├── reports/experiment_results/                      # 小型结果、图和报告
├── tools/                                           # 跨站比较和资料工具
├── PROJECT_GUIDE_CN.md                              # 项目说明
└── README.md                                        # 仓库入口
```

正式代码目录和Python包名称保留了早期厦门项目命名，以兼容已有实验；它们不代表项目只研究厦门。

## 核心流程

```text
源数据审计
  → GESLA质量控制与训练期UTide
  → ERA5对齐和处理后数据审计
  → 验证期基线、CNN-GRU与Rollout-6
  → 锁定模型
  → 测试期全年1—72小时评价
  → 测试期台风筛选与固定起报回算
  → 小型结果包和科研图表
```

新增站点的具体操作见：

- [中国沿海站点统一实验流程](projects/xiamen_short_term/short_term_forecast/CHINA_COASTAL_WORKFLOW.md)
- [测试期历史台风事件回算](projects/xiamen_short_term/short_term_forecast/HISTORICAL_TYPHOON_EVENTS.md)
- [期刊图件说明](projects/xiamen_short_term/short_term_forecast/JOURNAL_FIGURES_README.md)

## 数据与Git安全

- `F:\ERA5-NEW`、`F:\GESLA` 等原始数据目录只读。
- 不上传NetCDF、GESLA原始文件、NPY、模型权重和大型逐时预测。
- 可提交结果统一放在 `reports/experiment_results/`。
- 每个站点、年份和实验方案使用独立目录，不覆盖已有结果。
- 不使用 `git add .`，只暂存明确检查过的文件。

## 非当前主线

`projects/caribbean_forecast/` 保存早期加勒比实验，`projects/xiamen_reconstruction/` 保存早期日最大增水复现。两者均保留用于追溯，但当前不继续投入。
