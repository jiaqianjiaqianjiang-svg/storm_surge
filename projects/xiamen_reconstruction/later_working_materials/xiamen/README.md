# Xiamen Storm Surge Reconstruction Code

这个文件夹把厦门站风暴潮重建代码整理成两部分：

```text
01_preprocessing/     数据预处理：GESLA 潮位、UTide 分潮、ERA20C 输入、训练/验证集 .npy
02_model_training/   CNN 训练、5-model ensemble、指标和图片输出
```

真实 ERA20C/GESLA 数据不放在这里，输出结果也不要提交到 Git。

## 1. 预处理

先进入项目根目录：

```powershell
cd "E:\AAAqian\code\storm_surge\code_my\xiamen"
conda activate jjq
```

单年快速测试：

```powershell
python 01_preprocessing\preprocess_xiamen.py --start-year 1985 --end-year 1985
```

完整年份，按论文式前 5 年验证、其余年份训练：

```powershell
python 01_preprocessing\preprocess_xiamen.py --all-years --split-mode first-years --validation-years 5
```

预处理会保存到：

```text
outputs/xiamen/
```

主要文件包括：

```text
X_train.npy
X_val.npy
y_train.npy
y_val.npy
dates_train.npy
dates_val.npy
y_original.npy
dates_all.npy
cleaned_surge.csv
daily_max_surge.csv
y_scaler.json
split_metadata.json
```

## 2. 模型训练

快速测试：

```powershell
python 02_model_training\train_xiamen.py --epochs 2 --batch-size 16 --seeds 0
```

正式训练：

```powershell
python 02_model_training\train_xiamen.py --epochs 100 --batch-size 32 --lr 0.001
```

训练输出：

```text
models/xiamen/model_seed_*.pth
outputs/xiamen/validation_predictions.csv
outputs/xiamen/metrics.json
figures/xiamen/loss_curve.png
figures/xiamen/pred_vs_obs.png
figures/xiamen/scatter.png
```

## 注意

- 路径配置在两个子文件夹各自的 `config.py` 中，默认真实数据路径是：
  - `F:\ERA20C`
  - `F:\GESLA\GESLA3`
- 预处理和训练共用同一个根目录下的 `outputs/xiamen/`。
- 如果修改了 UTide、标签或 ERA 输入逻辑，需要删除旧的 `outputs/xiamen/` 后重新预处理。
