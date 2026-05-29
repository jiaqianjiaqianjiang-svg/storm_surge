# Storm Surge Xiamen CNN

本项目用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 中厦门站 `Xiamen` 的数据预处理、CNN 训练与验证流程。

仓库只提交代码、配置、README、requirements 和示例 notebook。真实 ERA20C/GESLA 数据、预处理输出、模型权重和图片结果都不提交到 GitHub。

## 数据路径

默认路径集中在 [src/config.py](src/config.py)：

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

训练阶段需要 PyTorch。如果你已经单独安装 GPU 版 PyTorch，可以不用重复安装。

## 1. 预处理

单年快速测试：

```bash
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

完整复现数据集：

```bash
python src/preprocess_xiamen.py --all-years
```

划分方式默认是 `--split-mode auto`：

- 当年份足够多时，使用更接近论文验证方式的划分：最早 5 年作为验证集，其余年份作为训练集。
- 当只跑 1985 这种单年测试时，自动退回时间顺序 80/20，只用于确认流程能跑通。

也可以显式指定：

```bash
python src/preprocess_xiamen.py --all-years --split-mode first-years --validation-years 5
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985 --split-mode chronological
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
13. 构建标签 `y shape = (N,)`，并用训练集 mean/std 标准化。

预处理输出目录：

```text
outputs/xiamen/
```

主要文件：

```text
X_train.npy             训练集 CNN 输入，shape=(N_train, 48, 40, 40)
y_train.npy             标准化后的训练集标签，shape=(N_train,)
X_val.npy               验证集 CNN 输入，shape=(N_val, 48, 40, 40)
y_val.npy               标准化后的验证集标签，shape=(N_val,)
dates_train.npy         训练集日期
dates_val.npy           验证集日期
y_original.npy          未标准化的全部标签
y_train_original.npy    未标准化的训练集标签
y_val_original.npy      未标准化的验证集标签
dates_all.npy           全部样本日期
y_scaler.json           标签标准化 mean/std
split_metadata.json     训练/验证划分说明
daily_max_surge.csv     每日最大风暴潮
cleaned_surge.csv       清洗和潮汐分离后的时间序列
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

当前模型训练设置：

```text
loss: MSELoss
optimizer: SGD
```

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

注意：训练时使用标准化后的 `y_train.npy` / `y_val.npy`。评估时脚本会用 `y_scaler.json` 反标准化，再默认乘以 100 转成厘米，与论文的 RMSE/MAE 单位对齐。

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

`metrics.json` 中主要指标包括：

```text
pearson_r
r2
rmse_cm
mae_cm
rrmse_percent
```

`validation_predictions.csv` 中保存两套数值：

```text
y_true_scaled    标准化后的验证标签
y_pred_scaled    标准化后的 ensemble 预测
y_true_raw       反标准化后的原始单位标签
y_pred_raw       反标准化后的原始单位预测
y_true_cm        厘米单位标签
y_pred_cm        厘米单位预测
```

每个 seed 的预测也会保存为：

```text
pred_seed_0_scaled / pred_seed_0_raw / pred_seed_0_cm
...
pred_seed_4_scaled / pred_seed_4_raw / pred_seed_4_cm
```

## 4. 查看结果

训练完成后，优先查看：

```text
outputs/xiamen/metrics.json
outputs/xiamen/validation_predictions.csv
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
