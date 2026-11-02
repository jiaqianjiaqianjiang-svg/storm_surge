# 中国沿海小时级短期风暴增水预测

本仓库当前主线是中国沿海多站点小时级短期风暴增水预测。研究使用验潮站观测和 ERA5 大气再分析资料，建立统一的 CNN-GRU、Rollout-6、基线模型和极值损失实验流程，评价未来 1—72 小时增水变化、强增水和历史台风过程。

- 当前工作分支：`codex/china-multistation`
- 详细说明：[`PROJECT_GUIDE_CN.md`](PROJECT_GUIDE_CN.md)

## 当前研究范围

| 站点 | 训练期 | 验证期 | 测试期 | 当前状态 |
|---|---|---|---|---|
| 厦门 Xiamen | 1970—1995 | 1996 | 1997 | 核心模型、72 h 滚动、极值损失和台风案例已完成 |
| 连云港 Lianyungang | 1975—1995 | 1996 | 1997 | 核心模型、72 h 滚动、极值损失和台风案例已完成 |
| 北海 Beihai | 1975—1995 | 1996 | 1997 | 核心模型、72 h 滚动、极值损失和台风案例已完成 |
| 汕尾 Shanwei | 由真实资料共同覆盖决定 | 计划保留 1996 | 计划保留 1997 | 统一流程已接入，等待实验室源数据审计和真实运行 |

统一输入为过去 24 小时 U10、V10、MSL 空间场及过去 24 小时非潮增水；主要评价 1、3、6、12、24、48、72 小时预测。

## 主要代码

正式项目目录：

```text
projects/xiamen_short_term/short_term_forecast/
```

该目录名称源自最初的厦门实验，但其中 `src/xiamen_forecast/` 已扩展为中国沿海多站点共用代码。目前不要再为新站点复制一套独立项目。

主要入口包括：

- `station_config.py`：站点、年份、数据和输出配置；
- `prepare_xiamen.py`、`audit_prepared_dataset.py`：预处理和数据审计；
- `train_xiamen.py`：一步模型训练；
- `train_rollout_temporal.py`：CNN-GRU 六步递归微调；
- `rolling_diagnostics.py`：1—72 小时滚动评价；
- `run_three_station_extreme_loss_suite.py`：三站极值损失实验调度；
- `run_typhoon_events.py`：审核台风事件的固定起报历史回算；
- `run_station_workflow.py`：新增站点标准流程；
- `journal_figures/`：论文和汇报图件。

## 已完成的核心结果

三站 1997 全年结果表明，Rollout-6 相比普通 CNN-GRU 均能降低 72 小时递归误差：

| 站点 | 一步最佳 RMSE | CNN-GRU 72 h RMSE | Rollout-6 72 h RMSE | 降幅 |
|---|---:|---:|---:|---:|
| 厦门 | 3.47 cm | 12.14 cm | **10.00 cm** | 17.6% |
| 连云港 | 5.49 cm | 19.96 cm | **15.71 cm** | 21.3% |
| 北海 | 3.37 cm | 17.16 cm | **14.49 cm** | 15.5% |

三站 1996 极值损失统一验证已经完成。结果显示不同站点的整体误差与强增水误差最优方案并不完全一致，不能强行指定一个适用于所有站点的损失函数。当前正在推进 MSE/MSE 与 WHuber/WHuber 在四场 1997 历史台风中的逐事件对照。

所有 72 小时实验使用未来有效时次的 ERA5 再分析真值，属于**已知未来气象强迫的历史回算**，不是实时业务预报。

## 结果位置

可提交 Git 的精简结果位于：

```text
reports/experiment_results/
```

当前重点目录：

- `xiamen_short_term_1996_seed42/`、`xiamen_short_term_1997_seed42/`；
- `lianyungang_short_term_1996_seed42/`、`lianyungang_short_term_1997_seed42/`；
- `beihai_short_term_1996_seed42/`、`beihai_short_term_1997_seed42/`；
- `china_three_station_comparison_1997_seed42/`；
- `three_station_extreme_loss_validation_1996_seed42/`；
- `china_three_station_event_distribution_v1/`；
- `historical_typhoon_four_case_review/`；
- `typhoon_events_1997_mse_whuber_comparison_seed42/`。

原始 GESLA、ERA5、处理后的大型 NPY、完整逐时输出和模型权重不上传 GitHub。

## 运行环境与数据安全

真实中国沿海数据和 GPU 实验主要在实验室 Windows 电脑运行：

```text
仓库：H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge
环境：conda activate jjq
项目：projects\xiamen_short_term\short_term_forecast
```

Mac 用于代码、测试、文档和精简结果审核，不假设本地具有实验室 F 盘原始数据。任何脚本都不得写入或删除 `F:\ERA5-NEW`、`F:\GESLA` 中的原始文件。

## 当前下一步

1. 在实验室运行三站 MSE/MSE 与 WHuber/WHuber 四场历史台风对照；
2. 检查强增水峰值、峰时和 24/48/72 小时误差是否跨事件一致；
3. 完成汕尾真实源数据审计后，再启动标准核心模型流程；
4. 后续扩展更多中国沿海站点和真正未参与模型开发的台风留出案例。

## 非当前主线

`projects/caribbean_forecast/` 保存早期 Prickly Bay 跨区域实验，代码和结果暂时保留，但当前不继续投入，也不作为仓库首页和近期实验计划的主要内容。`projects/xiamen_reconstruction/` 是更早的日最大增水论文复现，仅用于追溯。
