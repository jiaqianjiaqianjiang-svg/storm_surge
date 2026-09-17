# 厦门小时级短时风暴潮预报

正式实验统一使用 `src/xiamen_forecast/`。`src/short_term_forecast/` 仅保留期刊绘图和公用指标；已经被新流程替代的旧模型、旧训练器和旧滚动脚本已删除，避免两套入口混用。

## 当前正式数据

实验室电脑上的厦门数据与加勒比数据相互独立：

```text
F:\GESLA\GESLA3\xiamen-376a-chn-uhslc
F:\ERA5-NEW\Xiamen\xiamen_10u_1970_1997.nc
F:\ERA5-NEW\Xiamen\xiamen_v10_1970_1997.nc
F:\ERA5-NEW\Xiamen\xiamen_slp_1970_1997.nc
```

默认正式划分为 `1970—1995` 训练、`1996` 验证、`1997` 独立测试。UTide 也只使用截至 1995 年的观测标定，再重构 1996—1997 年潮汐，避免测试观测影响风暴增水标签。

## 正式 v2 流程

在实验室仓库中进入本目录：

```powershell
cd H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast
```

1. 生成按时间存储的内存映射数据。程序逐年读取三个大 NetCDF，不会一次加载 1970—1997 全部气象场，也不会为每个滑窗重复保存网格。

```powershell
python -m src.xiamen_forecast.prepare_xiamen `
  --era5-dir "F:\ERA5-NEW\Xiamen" `
  --gesla-file "F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
```

2. 训练前审计时间、形状、缺测、物理范围和有效样本：

```powershell
python -m src.xiamen_forecast.audit_prepared_dataset
```

3. 生成 Zero、Persistence 和 Ridge 基线：

```powershell
python -m src.xiamen_forecast.evaluate_baselines
```

4. 训练消融模型。`surge_mlp` 只使用历史增水，`era5_cnn` 只使用 ERA5；它们用于判断两类信息的独立贡献，不属于五模型主体对比：

```powershell
python -m src.xiamen_forecast.train_xiamen --model-type surge_mlp
python -m src.xiamen_forecast.train_xiamen --model-type era5_cnn
```

5. 在完全相同的数据划分、24 小时输入和评价规则下训练五个正式对比模型：

```powershell
python -m src.xiamen_forecast.train_xiamen --model-type cnn --batch-size 256 --epochs 50 --patience 8
python -m src.xiamen_forecast.train_xiamen --model-type cnn_lstm --batch-size 32 --epochs 50 --patience 8
python -m src.xiamen_forecast.train_xiamen --model-type cnn_gru --batch-size 32 --epochs 50 --patience 8
python -m src.xiamen_forecast.train_xiamen --model-type tcn --batch-size 32 --epochs 50 --patience 8
python -m src.xiamen_forecast.train_xiamen --model-type transformer --batch-size 32 --epochs 50 --patience 8
```

也可以一次按顺序运行五个模型；已经生成 `metrics.json` 的完整实验会自动跳过，实验中断后可重复执行同一命令继续后续任务：

```powershell
python -m src.xiamen_forecast.train_model_suite --device cuda
```

`cnn` 是“空间 CNN + 历史增水 MLP”的正式基础模型。旧命令中的 `dual` 仍可读取，但只是 `cnn` 的兼容别名，不应重复训练。所有模型默认在 CUDA 上启用 AMP 混合精度；若显存充足，可逐步把时序模型 batch size 从 32 调到 64 或 128。Windows 的 `--num-workers` 默认是 0，确认运行稳定后可尝试 `--num-workers 2`。

模型使用 early stopping，并输出 1996 验证和 1997 测试结果。指标包括 RMSE、MAE、Bias、Pearson r、R2、RRMSE、Top 10%/5%强增水、快速上涨和独立峰值误差。

五个模型跑完后，先汇总 1996 验证集的一步预测结果：

```powershell
python -m src.xiamen_forecast.compare_models --split validation
```

在模型方案完全确定前，不要用 `--split test` 反复挑模型，以免把 1997 独立测试年变成调参数据。

6. 基础 CNN 完成后，可继续做 6 步递归损失微调。它从已有一步 CNN 权重开始，冻结耗时最大的气象编码器，只调整历史分支和融合层，因此不是重新从头训练：

```powershell
python -m src.xiamen_forecast.train_rollout_cnn --device cuda --rollout-steps 6
```

7. 所有一步模型完成后，在 1996 验证年统一比较 1、3、6、12、24、48、72 小时递归回算：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --device cuda
```

如已完成 6 步微调，可把它加入同一张对比表：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --device cuda --rollout-checkpoint "models\xiamen\formal_seed42\cnn_rollout6\best_model.pth"
```

滚动实验使用未来时次的 ERA5 再分析场，因此应称为“已知未来大气强迫条件下的历史回算”，不能表述为业务实时预报。1997 测试年在模型和方案确定前保持封存。

这一版已经接入 CNN、CNN-LSTM、CNN-GRU、TCN、Transformer 和 CNN 6 步递归微调。加勒比项目中的直接 24 小时表格模型、ERA5 消融扩展和物理残差融合尚未直接复制到厦门；应先完成上述统一模型对比，再根据 1996 验证结果决定是否迁移。

## 测试

```powershell
python -m compileall -q src
python -m pytest tests -q
```

以下内容不会提交到 GitHub：真实 NetCDF、验潮数据、对齐数组、模型权重和新实验输出。
