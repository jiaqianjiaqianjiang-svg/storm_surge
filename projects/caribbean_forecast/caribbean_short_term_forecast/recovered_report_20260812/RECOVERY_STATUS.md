# Prickly Bay实验文件恢复状态（2026-08-12）

## 当前结论

- `storm_surge_clean`中的源代码、配置、测试和Git提交完整，最新本地提交为`1c179de`。
- 本地分支`codex-caribbean-short-term-forecast`仍保留此前6个未推送提交。
- 原有`outputs/`、`models/`、`figures/`和`logs/`运行产物已不存在，这些目录目前只剩`.gitkeep`。
- 这些运行产物被`.gitignore`排除，因此不能从Git历史恢复。
- 当前Windows只识别C、D、E盘；配置指向的`F:/data`外接数据盘没有挂载，暂时无法重跑正式实验。

## 已恢复，可用于今晚汇报

以下文件依据此前已核验并在工作记录中保存的最终指标重建：

- `direct_vs_rolling_selected_leads.csv`
- `era5_ablation_selected_leads.csv`
- `top5_era5_improvement.csv`
- `rmse_direct_vs_rolling_recovered.png`
- `rmse_era5_ablation_recovered.png`
- `top5_era5_improvement_recovered.png`

这些表格和曲线可用于阶段汇报，但属于“根据最终记录恢复的汇报版”，不是从当前逐时预测数组重新计算的原始输出。

## 仍缺失的原始产物

1. 处理后对齐数据：
   - `outputs/processed/prickly_bay/aligned_dataset/atmosphere.npy`（此前约1.16 GiB）
   - `surge.npy`
   - `times.npy`
   - 数据审计、QC和UTide分潮输出
2. 模型权重：Ridge、XGBoost、MLP、GRU、Dual-CNN及消融模型checkpoint。
3. 逐时预测数组和完整1—24小时指标CSV。
4. 原始训练损失记录、实验metadata和自动生成Markdown报告。
5. 根据逐时预测生成的典型强增水过程图。

## 数据盘恢复后的重跑顺序

1. 确认`F:/data/caribbean_tide_gauges/pric/`与`F:/data/ERA5-Caribbean/Regional_Union/`存在。
2. 重新运行`prepare_station.py --station prickly_bay`，恢复完整对齐数据集并执行审计。
3. 重建直接多步线性基线。
4. 重建直接24小时多模型实验。
5. 重建统一的直接—滚动24小时比较。
6. 重建最小ERA5消融实验。
7. 对重建指标与本恢复包中的记录逐项比对；一致后再开展5随机种子和时间块Bootstrap。

正式重跑前不要加载2018用于调参；2018仍只用于锁定方案后的时间外评价。
