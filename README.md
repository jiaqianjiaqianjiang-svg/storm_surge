# Storm Surge CNN Reconstruction

本项目用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 的风暴潮数据预处理与 CNN 重建流程。

当前代码支持论文技术验证中举例分析的 6 个站点：

```text
Kushiro
Kashiwazaki
Lusi
Xiamen
Geting
Legaspi
```

仓库只提交代码、配置、README、requirements 和示例 notebook。真实 ERA20C/GESLA 数据、预处理输出、模型权重、图片结果都不提交到 GitHub。

## 1. 数据路径

真实路径集中在 [src/config.py](src/config.py)：

```python
ERA20C_DIR = Path(r"F:\ERA20C")
GESLA_DIR = Path(r"F:\GESLA\GESLA3")
```

ERA20C 目录结构需要是：

```text
F:\ERA20C\10U\*.grb
F:\ERA20C\10V\*.grb
F:\ERA20C\SLP\*.grb
```

站点配置在 `SITES` 中。厦门站已经固定文件名：

```python
filename="xiamen-376a-chn-uhslc"
```

其他站点会按站名关键词在 `F:\GESLA\GESLA3` 自动查找文件。如果某个站点提示找不到文件，请先在远程电脑运行：

```powershell
Get-ChildItem F:\GESLA\GESLA3 | Where-Object { $_.Name -match "kushiro|kashiwazaki|lusi|lvsi|lyusi|geting|legaspi|legazpi" } | Select-Object Name
```

然后把真实文件名填到 [src/config.py](src/config.py) 对应站点的 `filename`。

## 2. 环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行：

```powershell
conda activate jjq
pip install -r requirements.txt
```

训练阶段需要 PyTorch。如果已经安装 GPU 版 PyTorch，不需要重复安装。

## 3. 预处理

脚本入口仍然是：

```powershell
python src/preprocess_xiamen.py
```

虽然文件名保留了 `xiamen`，现在已经支持 `--site` 参数。

单年快速测试：

```powershell
python src/preprocess_xiamen.py --site xiamen --start-year 1985 --end-year 1985
python src/preprocess_xiamen.py --site lusi --start-year 1985 --end-year 1985
```

完整年份处理：

```powershell
python src/preprocess_xiamen.py --site xiamen --all-years --split-mode first-years --validation-years 5
python src/preprocess_xiamen.py --site lusi --all-years --split-mode first-years --validation-years 5
```

可选站点：

```text
xiamen
kushiro
kashiwazaki
lusi
geting
legaspi
```

预处理流程：

1. 读取 GESLA 潮位文件，自动跳过元数据。
2. 清洗重复时间、缺测值、use_flag 坏值和明显异常值。
3. 使用 UTide 做调和分析。
4. 计算 `storm_surge = observed_sea_level - predicted_tide`。
5. 按天取 `daily maximum storm surge` 作为标签。
6. 读取 ERA20C 的 U10、V10、SLP。
7. 自动识别 GRIB 变量名。
8. 裁剪站点周围 `10° x 10°` 区域。
9. 插值到 `40 x 40` 网格。
10. 按变量分别标准化。
11. 对某一天 D，使用 D-1 和 D 两天共 16 个 3 小时时间片。
12. 按作者 notebook 风格把 16 个 `40 x 40` 时间片拼成 `160 x 160`。
13. 生成 CNN 输入 `X shape = (N, 3, 160, 160)`。
14. 生成标准化标签 `y shape = (N,)`。

第一次从 GRIB 读取 ERA20C 会比较慢。代码会把每个站点、每个变量、每一年的裁剪插值结果缓存到：

```text
cache/<site>/era20c_yearly/
```

再次运行同一站点时会优先读缓存，速度会快很多。

## 4. 预处理输出

每个站点单独输出，例如：

```text
outputs/xiamen/
outputs/lusi/
outputs/kushiro/
```

主要文件：

```text
X_train.npy             训练集输入，shape=(N_train, 3, 160, 160)
y_train.npy             标准化后的训练集标签
X_val.npy               验证集输入，shape=(N_val, 3, 160, 160)
y_val.npy               标准化后的验证集标签
dates_train.npy         训练集日期
dates_val.npy           验证集日期
y_original.npy          未标准化的全部标签
y_train_original.npy    未标准化的训练集标签
y_val_original.npy      未标准化的验证集标签
dates_all.npy           全部样本日期
y_scaler.json           标签标准化参数
split_metadata.json     数据划分说明
daily_max_surge.csv     每日最大风暴潮
cleaned_surge.csv       清洗和潮汐分离后的时间序列
```

## 5. CNN 训练

快速测试：

```powershell
python src/train_xiamen.py --site xiamen --epochs 2 --batch-size 16 --seeds 0
```

正式训练：

```powershell
python src/train_xiamen.py --site xiamen --epochs 100 --batch-size 32 --lr 0.001 --patience 10
```

其他站点只需要改 `--site`：

```powershell
python src/train_xiamen.py --site lusi --epochs 100 --batch-size 32 --lr 0.001 --patience 10
```

训练设置：

```text
loss: MSELoss
optimizer: SGD
ensemble: 5 个 seed 训练后平均
early stopping: 默认开启
```

模型结构贴近作者 notebook：

```text
Input: (3, 160, 160)
Conv2d(3, 6, 5x5) -> BatchNorm -> ReLU -> MaxPool(2x2)
Conv2d(6, 12, 5x5) -> BatchNorm -> ReLU -> MaxPool(2x2)
Conv2d(12, 6, 5x5) -> BatchNorm -> ReLU -> MaxPool(2x2)
FC -> FC -> FC -> Storm Surge
```

评估时会用 `y_scaler.json` 反标准化，并转换为厘米计算：

```text
Pearson r
R2
RMSE cm
MAE cm
RRMSE percent
Top 5% extreme metrics
```

## 6. 训练输出

每个站点单独保存：

```text
models/<site>/model_seed_0.pth
models/<site>/model_seed_1.pth
models/<site>/model_seed_2.pth
models/<site>/model_seed_3.pth
models/<site>/model_seed_4.pth

outputs/<site>/validation_predictions.csv
outputs/<site>/metrics.json

figures/<site>/loss_curve.png
figures/<site>/pred_vs_obs.png
figures/<site>/scatter.png
```

## 7. 不要提交真实数据和结果

`.gitignore` 已排除：

```text
data/
cache/
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

请不要把 ERA20C、GESLA、`cache/`、`outputs/`、`models/`、`figures/` 或模型权重提交到 GitHub。
