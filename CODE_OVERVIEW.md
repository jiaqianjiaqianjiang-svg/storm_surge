# 代码说明

本仓库现在把代码分成两部分：原论文复现代码和新增的厦门站小时级短时预报代码。真实数据、模型权重和输出结果不放进代码目录。

## 目录结构

```text
src/
  config.py                    # 共享配置：数据路径、站点信息、网格大小、默认预报参数
  test_paths.py                # 检查实验室电脑上的 ERA5/ERA20C/GESLA 路径

  paper_reconstruction/        # 原论文复现：大气场 -> daily maximum storm surge
    preprocess_xiamen.py       # 预处理入口
    train_xiamen.py            # 原论文 CNN 训练入口
    cnn_model.py               # 原论文 CNN 模型
    dataset_builder.py         # 复现任务样本构建
    era20c_loader.py           # ERA20C 读取
    gesla_loader.py            # GESLA 潮位读取
    tide_processing.py         # UTide 去潮、日最大增水
    metrics.py                 # 指标计算
    plot_results.py            # 结果作图

  short_term_forecast/         # 新增短时预报：前 t 小时 -> 下一小时
    forecast_dataset.py        # 构建小时级短时预报样本
    forecast_cnn_model.py      # CNN + 历史增水 MLP 双分支模型
    train_forecast_xiamen.py   # 训练一步预测模型
    rolling_forecast.py        # 使用一步模型做多步滚动预报
    compare_forecast_windows.py# 对比 12h/24h 等输入窗口
```

## 两个任务的区别

原论文复现任务是重建任务：

```text
U10/V10/SLP 大气场 -> 当天 daily maximum storm surge
```

新增短时预报任务是小时级预报：

```text
前 t 小时 ERA5 U10/V10/SLP 40x40
+
前 t 小时 storm surge
-> 下一小时 storm surge
```

其中 SLP 在部分 ERA5 文件中的变量名可能是 `msl`，代码会自动识别。

## 短时预报代码逻辑

1. `config.py`  
   配置真实数据路径和默认参数。当前默认使用 `ERA5`、`hourly`、`input_steps=24`、`horizon=1`。

2. `forecast_dataset.py`  
   读取厦门站 storm surge 和 ERA5 大气场。小时级任务优先使用 `cleaned_surge.csv`；如果没有，则从 GESLA 原始文件读取并按 `use_flag` 过滤无效记录。ERA5 裁剪后如果已经是 `40x40` 就不插值，否则重采样到 `40x40`。

3. `forecast_cnn_model.py`  
   模型有两个分支：CNN 分支处理 `t*3` 个大气场通道，MLP 分支处理前 `t` 小时 storm surge。两个分支拼接后输出下一小时 storm surge。

4. `train_forecast_xiamen.py`  
   按时间顺序划分训练集和验证集，默认前 80% 为训练集、后 20% 为验证集。训练阶段只支持 `horizon=1` 的一步预测。输出 `metrics.json`、`val_predictions.csv`、`loss_history.csv` 和结果图。小时级数组很大，默认不保存 `.npy` 大数组，只有加 `--save-arrays` 才保存。

5. `rolling_forecast.py`  
   加载训练好的一步预测模型进行滚动预报。第一步使用真实历史 storm surge，之后把模型预测值加入历史序列继续预测。气象强迫目前使用历史 ERA5 再分析数据，真实 observed 只用于最后画图对比。

6. `compare_forecast_windows.py`  
   对比不同输入窗口长度，例如 `t=12h` 和 `t=24h` 的验证集效果。

## 短时预报复用了哪些原论文代码思路

短时预报没有直接改写原论文训练脚本，而是在新目录中单独实现。复用和继承的部分主要有：

```text
1. 共享 config.py 中的厦门站经纬度、GESLA 路径、输出目录和 40x40 网格参数。
2. 复用了原论文流程中的 GESLA 读取、缺测值过滤、MAD 异常值过滤和 UTide 去潮思路。
3. 复用了原论文中“站点周围区域裁剪并统一到 40x40 网格”的数据处理思路。
4. 复用了原论文 CNN 从 U10/V10/SLP 空间场提取特征的建模思路。
5. 指标仍使用 Pearson r、RMSE、MAE、RRMSE，方便和之前结果对照。
```

区别是：短时预报的标签不再是当天日最大增水，而是下一小时 storm surge；输入也增加了前 `t` 小时历史 storm surge。

## 常用命令

检查路径：

```bash
python src/test_paths.py
```

训练 24 小时窗口小时级模型：

```bash
python src/short_term_forecast/train_forecast_xiamen.py --start-year 1985 --end-year 1985 --input-steps 24 --epochs 2
```

训练 12 小时窗口小时级模型：

```bash
python src/short_term_forecast/train_forecast_xiamen.py --start-year 1985 --end-year 1985 --input-steps 12 --epochs 2
```

运行 24 小时窗口滚动预报：

```bash
python src/short_term_forecast/rolling_forecast.py --model-path models/forecast_xiamen/ERA5_1985_1985_hourly_t24_h1/best_forecast_cnn.pth --start-date "1985-11-01 00:00" --steps 72
```

对比 12h 和 24h：

```bash
python src/short_term_forecast/compare_forecast_windows.py --runs ERA5_1985_1985_hourly_t12_h1 ERA5_1985_1985_hourly_t24_h1
```

原论文复现预处理：

```bash
python src/paper_reconstruction/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

原论文复现训练：

```bash
python src/paper_reconstruction/train_xiamen.py --epochs 2 --batch-size 16
```

## 不需要发送的内容

发代码给师兄时不要包含以下内容：

```text
outputs/
models/
figures/
data/
__pycache__/
.mplconfig/
final_code_snapshot/
*.npy
*.pth
*.pt
*.nc
*.grb
*.csv
```
