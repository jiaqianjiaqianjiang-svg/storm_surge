# Storm Surge Xiamen Preprocessing

本项目用于复现论文 **A dataset of storm surge reconstructions in the Western North Pacific using CNN** 的数据预处理流程。当前版本先以厦门站 `Xiamen` 为例，完成从 GESLA 潮位、UTide 潮汐分离、ERA-20C 气象场处理，到 CNN 可用输入数据集与训练/验证划分的流程。

本项目只提交代码、配置、README、requirements 和示例 notebook。真实数据、输出数组和模型文件不提交到 GitHub。

## 数据路径

默认路径集中在 `src/config.py`：

```python
ERA20C_DIR = r"F:\ERA20C"
GESLA_DIR = r"F:\GESLA\GESLA3"
SITE_FILE = r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
```

ERA-20C 目录结构需要是：

```text
F:\ERA20C\10U\*.grb
F:\ERA20C\10V\*.grb
F:\ERA20C\SLP\*.grb
```

每个变量每年一个 `.grb` 文件。

## 环境

建议在实验室远程电脑的 conda 环境 `jjq` 中运行。当前代码依赖：

```bash
pip install -r requirements.txt
```

如果环境中已经安装了 `numpy pandas matplotlib scipy scikit-learn tqdm xarray netCDF4 utide jupyter ipykernel cfgrib eccodes`，通常可以直接运行。

## 处理流程

1. 读取厦门站 GESLA 文件，自动跳过 `#` 元数据并识别数据起始行。
2. 解析 `date`、`time`、`sea_level`、`qc_flag`、`use_flag`。
3. 删除重复时间、缺测标记和明显坏值。
4. 使用 UTide 对完整观测期做调和分析。
5. 计算 `storm surge = observed sea level - predicted tide`。
6. 按天取 `daily maximum storm surge` 作为标签 `y`。
7. 读取 ERA-20C 的 U10、V10、SLP。
8. 提取厦门站周围 `10°×10°` 区域并插值到 `40×40`。
9. 保留 3 小时时间分辨率，并分别标准化 U10、V10、SLP。
10. 对每一天 D，使用 D-1 与 D 两天共 16 个时间片，构建 `(48, 40, 40)` 输入。
11. 按时间顺序划分：前 80% 训练集，后 20% 验证集。
12. 使用训练集标签的均值和标准差标准化 `y_train`/`y_val`，同时保留原始标签 `y_original.npy`。

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

## 运行命令

测试单年：

```bash
python src/preprocess_xiamen.py --start-year 1985 --end-year 1985
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

运行时日志会打印：

- 使用的年份范围；
- 读取的 GESLA 文件；
- 读取了哪些 ERA 年份；
- 每年生成了多少样本；
- 最终 X/y shape；
- 训练集/验证集 shape；
- 输出文件保存位置。

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
y_scaler.json
daily_max_surge.csv
cleaned_surge.csv
```

这些文件可能很大，已经在 `.gitignore` 中排除，不要提交到 GitHub。

## 注意事项

- `--all-years` 可能生成数 GB 的 `.npy` 文件，请确认磁盘空间充足。
- `cfgrib` 依赖 ecCodes；如果读取 GRIB 报错，优先检查 `cfgrib` 和 `eccodes` 是否可用。
- 本项目当前只做预处理，不训练 CNN。
- 如果需要切换到其他潮位站，请先复制 `config.py` 中的站点路径、经纬度和年份范围，再复用同一套流程。
