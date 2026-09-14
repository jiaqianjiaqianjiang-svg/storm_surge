# 厦门小时级短时风暴潮预报

本目录同时保留早期 1985 年 Ridge 流程验证和正式 ERA5 多年预报代码。新实验统一使用 `src/xiamen_forecast/`；旧版代码保留在 `src/short_term_forecast/`，只用于复核此前结果。

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
cd E:\jjq\surge\storm_surge\projects\xiamen_short_term\short_term_forecast
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

4. 分别训练历史增水、ERA5 和双分支一步模型：

```powershell
python -m src.xiamen_forecast.train_xiamen --model-type surge_mlp
python -m src.xiamen_forecast.train_xiamen --model-type era5_cnn
python -m src.xiamen_forecast.train_xiamen --model-type dual
```

模型使用 early stopping，并输出 1996 验证和 1997 测试结果。指标包括 RMSE、MAE、Bias、Pearson r、R2、RRMSE、Top 10%/5%强增水、快速上涨和独立峰值误差。

5. 在 1996 验证年统一比较 1、3、6、12、24、48、72 小时递归回算：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --device cuda
```

滚动实验使用未来时次的 ERA5 再分析场，因此应称为“已知未来大气强迫条件下的历史回算”，不能表述为业务实时预报。1997 测试年在模型和方案确定前保持封存。

这一版先补齐可复现的正式一步预测和统一滚动评价。加勒比项目中的直接 24 小时 XGBoost/GRU、ERA5 消融扩展和物理残差融合尚未直接复制到厦门；应先完成数据审计和三类一步模型，再根据 1996 验证结果决定是否迁移，避免在未经核对的厦门标签上批量跑模型。

## 测试

```powershell
python -m py_compile src\xiamen_forecast\*.py
pytest tests\test_xiamen_v2.py
```

以下内容不会提交到 GitHub：真实 NetCDF、验潮数据、对齐数组、模型权重和新实验输出。

## 早期阶段成果

以下内容是 2026-06-24 整理的 1985 年流程验证成果。

## 文件说明

- `short_term_rolling_baseline.py`：小时风暴增水 Ridge 滚动预测基线。它只验证滑动窗口、递归预测和分时效评价，不是最终 CNN 模型。
- `make_report_figures.py`：生成短时 CNN 流程图、分潮过程图和年度增水概览图。
- `本周汇报_短时风暴潮预测.md`：任务理解、本周进展、初步结果、限制和下一步计划。
- `results/`：脚本生成的 CSV、JSON 和图片，已在 `.gitignore` 中忽略。
- `figures_for_report/`：适合本周汇报正文展示的三张图片。

## 运行方式

在项目根目录执行：

```powershell
python short_term_forecast\short_term_rolling_baseline.py
python short_term_forecast\make_report_figures.py
```

脚本默认读取：

```text
data/xiamen_GESLA/xiamen-376a-chn-uhslc
```

当前试验使用修正后的 UTide 分潮逻辑，从原始 GESLA 小时潮位重新计算风暴增水，不读取已删除的旧 `processed_xiamen_1985` 标签。

## 正式模型方向

最终路线仍是沿用论文的 U10、V10、SLP 三类 `40×40` 气象网格和 CNN，同时维护增水历史窗口，并把新预测增水回填用于下一次预测。Ridge 结果仅作为增水自回归流程基线和持续性模型对照。
