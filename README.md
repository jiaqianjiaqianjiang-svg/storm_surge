# Storm Surge Xiamen Preprocessing

本项目用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 的数据预处理流程。当前代码以厦门站 `Xiamen` 为例，完成从 GESLA 潮位清洗、UTide 潮汐分离、ERA-20C 气象场处理，到 CNN 可用输入数据集与训练/验证划分的完整预处理流程。

本项目只提交代码、配置、README、requirements 和示例 notebook。真实数据、输出数组和模型文件不提交到 GitHub。

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

每个变量每年一个 `.grb` 文件。

## 环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行：

```bash
conda activate jjq
pip install -r requirements.txt
```

如果环境中已经安装了 `numpy pandas matplotlib scipy scikit-learn tqdm xarray netCDF4 utide jupyter ipykernel cfgrib eccodes`，通常可以直接运行。

## 完整预处理流程

1. 读取厦门站 GESLA 文件，自动跳过 `#` 元数据并识别数据起始行。
2. 解析 `date`、`time`、`sea_level`、`qc_flag`、`use_flag`。
3. 删除重复时间、缺测标记和明显坏值。
4. 使用 UTide 对观测期做调和分析。
5. 计算 `storm surge = observed sea level - predicted tide`。
6. 按天取 `daily maximum storm surge` 作为标签 `y`。
7. 读取 ERA-20C 的 U10、V10、SLP。
8. 自动识别 GRIB 文件中的变量名。
9. 提取厦门站周围 `10°×10°` 区域。
10. 将每个变量插值到 `40×40` 网格。
11. 保留 ERA-20C 的 3 小时时间分辨率。
12. 对 U10、V10、SLP 分别计算均值和标准差并标准化。
13. 对某一天 `D`，使用 `D-1` 和 `D` 两天 ERA-20C，共 16 个 3 小时时间片。
14. 每个时间片包含 U10、V10、SLP 三个变量，最终单样本 shape 为 `(48, 40, 40)`。
15. 按时间顺序划分：前 80% 训练集，后 20% 验证集。
16. 保存 `.npy` 与 `.csv` 输出到 `outputs/xiamen/`。

通道顺序为：

```text
U10 的 16 个时间片 -> V10 的 16 个时间片 -> SLP 的 16 个时间片
```

因此：

```text
单个样本 shape = (48, 40, 40)
X shape = (N, 48, 40, 40)
y shape = (N,)
```

注意：当只运行单年 `1985` 时，`1985-01-01` 需要 `1984-12-31` 的 ERA 数据，因此会被自动跳过。这是正常现象。

## 运行命令

优先测试单年 1985：

```bash
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

成功时日志中应能看到类似：

```text
[ERA] 变量 u10 自动识别为: ...
[ERA] 裁剪区域: ...
[ERA] 1985 u10 插值后 shape: ...
[DATASET] X 总 shape: (N, 48, 40, 40)
[DATASET] 训练集 shape: X(..., 48, 40, 40), y(...)
[DATASET] 验证集 shape: X(..., 48, 40, 40), y(...)
[DATASET] 输出文件保存位置: ...\outputs\xiamen
```

运行厦门站全部可用年份：

```bash
python src/preprocess_xiamen.py --all-years
```

也可以显式指定完整范围：

```bash
python src/preprocess_xiamen.py --start-year 1954 --end-year 1997
```

如果某一年 ERA 文件缺失，希望先跳过缺失年份继续运行：

```bash
python src/preprocess_xiamen.py --all-years --skip-missing-era
```

## 输出文件

输出目录：

```text
outputs/xiamen/
```

生成文件：

```text
X_train.npy
y_train.npy
X_val.npy
y_val.npy
dates_train.npy
dates_val.npy
y_original.npy
dates_all.npy
daily_max_surge.csv
cleaned_surge.csv
```

额外还会保存：

```text
y_scaler.json
```

其中 `y_train.npy` 和 `y_val.npy` 是用训练集标签均值和标准差标准化后的标签；`y_original.npy` 保留未标准化的 daily maximum storm surge，便于后续画图和反标准化。

## 日志说明

运行时会打印：

- 使用的年份范围；
- 读取的 GESLA 文件；
- GESLA 清洗记录数；
- UTide 潮汐分离后的记录数；
- daily maximum storm surge 日期范围；
- 读取了哪些 ERA 年份；
- GRIB 中自动识别到的变量名；
- ERA 原始维度；
- 裁剪区域；
- 插值后 shape；
- 每年生成了多少 CNN 样本；
- 跳过了多少日期以及跳过原因；
- 最终 X/y shape；
- 训练集/验证集 shape；
- 输出文件保存位置。

## 不要提交真实数据

`.gitignore` 已排除：

```text
data/
outputs/
*.grb
*.grib
*.idx
*.nc
*.npy
*.npz
*.csv
*.pth
*.pt
*.zip
```

请不要把 ERA20C、GESLA、`outputs/xiamen/` 或模型权重提交到 GitHub。

## 常见问题

如果 GRIB 读取失败，优先检查：

```bash
python src/test_paths.py
```

确认 `F:\ERA20C` 和 `F:\GESLA\GESLA3` 存在。

如果报 `cfgrib` 或 `eccodes` 相关错误，请确认当前 conda 环境中已经安装并能正常导入：

```bash
python -c "import cfgrib, eccodes; print('ok')"
```

本项目当前只做预处理，不训练 CNN。
