# Storm Surge Xiamen CNN

本项目用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 中厦门站 `Xiamen` 的数据预处理、CNN 训练与验证流程。

仓库只提交代码、配置、README、requirements 和示例 notebook。真实数据、预处理输出、模型权重和图片结果都不提交到 GitHub。

## 数据路径

默认路径集中在 `src/config.py`：

```python
ERA20C_DIR = r"F:\ERA20C"
GESLA_DIR = r"F:\GESLA\GESLA3"
SITE_FILE = r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
SITE_LAT = 24.45
SITE_LON = 118.067
```

ERA-20C 目录结构需要是：

```text
F:\ERA20C\10U\*.grb
F:\ERA20C\10V\*.grb
F:\ERA20C\SLP\*.grb
```

## 环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行：

```bash
conda activate jjq
pip install -r requirements.txt
```

训练阶段需要 PyTorch。如果你已经单独安装了 GPU 版 PyTorch，可以不用重复安装。

## 1. 预处理

预处理入口：

```bash
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

完整年份：

```bash
python src/preprocess_xiamen.py --all-years
```

预处理流程：

1. 读取厦门站 GESLA 文件，自动跳过 `#` 元数据并识别数据行。
2. 清洗潮位数据，删除重复时间、缺测标记和明显坏值。
3. 使用 UTide 做潮汐分离。
4. 计算 `storm surge = observed sea level - predicted tide`。
5. 按天取 `daily maximum storm surge` 作为标签 `y`。
6. 读取 ERA-20C 的 U10、V10、SLP。
7. 自动识别 GRIB 文件中的变量名。
8. 提取厦门站周围 `10°×10°` 区域。
9. 插值到 `40×40` 网格。
10. 对 U10、V10、SLP 分别标准化。
11. 对某一天 `D`，使用 `D-1` 和 `D` 两天共 16 个 3 小时时间片。
12. 构建 CNN 输入 `X shape = (N, 48, 40, 40)`。
13. 构建标签 `y shape = (N,)`。
14. 按时间顺序划分：前 80% 训练集，后 20% 验证集。

预处理输出目录：

```text
outputs/xiamen/
```

主要文件：

```text
X_train.npy        训练集 CNN 输入，shape=(N_train, 48, 40, 40)
y_train.npy        标准化后的训练集标签，shape=(N_train,)
X_val.npy          验证集 CNN 输入，shape=(N_val, 48, 40, 40)
y_val.npy          标准化后的验证集标签，shape=(N_val,)
dates_train.npy    训练集日期
dates_val.npy      验证集日期
y_original.npy     未标准化的全部标签
dates_all.npy      全部样本日期
y_scaler.json      标签标准化 mean/std
daily_max_surge.csv 每日最大风暴潮
cleaned_surge.csv  清洗和潮汐分离后的时间序列
```

## 2. CNN 训练

快速测试：

```bash
python src/train_xiamen.py --epochs 2 --batch-size 16
```

正式训练示例：

```bash
python src/train_xiamen.py --epochs 100 --batch-size 32 --lr 0.001
```

训练脚本会训练 5 个不同随机种子的模型：

```text
seed = 0, 1, 2, 3, 4
```

并对验证集做 5-model averaging ensemble。

模型结构：

```text
Input: (48, 40, 40)
Conv2d(5×5) -> BatchNorm -> ReLU -> MaxPool(2×2)
Conv2d(5×5) -> BatchNorm -> ReLU -> MaxPool(2×2)
Conv2d(5×5) -> BatchNorm -> ReLU -> MaxPool(2×2)
Linear -> ReLU
Linear -> ReLU
Linear -> output
```

训练设置：

```text
loss: MSELoss
optimizer: SGD
```

## 3. 训练输出

模型保存到：

```text
models/xiamen/model_seed_0.pth
models/xiamen/model_seed_1.pth
models/xiamen/model_seed_2.pth
models/xiamen/model_seed_3.pth
models/xiamen/model_seed_4.pth
```

验证结果保存到：

```text
outputs/xiamen/validation_predictions.csv
outputs/xiamen/metrics.json
```

图片保存到：

```text
figures/xiamen/loss_curve.png
figures/xiamen/pred_vs_obs.png
figures/xiamen/scatter.png
```

指标包括：

```text
Pearson correlation r
RMSE
MAE
RRMSE
```

`validation_predictions.csv` 中包含：

```text
date
observed
pred_ensemble
pred_seed_0
pred_seed_1
pred_seed_2
pred_seed_3
pred_seed_4
```

其中 `observed` 和预测值都已经从标准化值反变换回原始 storm surge 单位。

## 4. 查看结果

训练完成后，优先查看：

```text
outputs/xiamen/metrics.json
figures/xiamen/pred_vs_obs.png
figures/xiamen/scatter.png
figures/xiamen/loss_curve.png
```

如果只想确认流程能跑通，先使用：

```bash
python src/train_xiamen.py --epochs 2 --batch-size 16
```

## 5. 不要提交真实数据和结果

`.gitignore` 已排除：

```text
data/
outputs/
models/
figures/
*.grb
*.grib
*.nc
*.npy
*.csv
*.pth
*.pt
*.png
*.zip
```

请不要把 ERA20C、GESLA、`outputs/`、`models/`、`figures/` 或模型权重提交到 GitHub。
