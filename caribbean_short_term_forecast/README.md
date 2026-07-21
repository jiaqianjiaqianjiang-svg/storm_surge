# 加勒比地区风暴潮短时滚动预报

这是与现有厦门代码完全独立的方案 B 实现。它复用厦门试验中“UTide 重新分潮、按时间切分、递归回填预测值”的成熟设计，但不会导入、修改或覆盖 `short_term_forecast/`。第一阶段用 Prickly Bay 2011—2018 验证全流程，随后可通过站点配置切换到 Calliaqua、Ganter's Bay、San Juan、Charlotte Amalie 和 Cristobal。

安装独立依赖：`pip install -r caribbean_short_term_forecast/requirements.txt`。

## 预测任务

对目标小时 `T`，模型接收 `T-24h` 至 `T-1h` 的 ERA5 `U10/V10/MSL` 40×40 网格，以及相同 24 小时的历史风暴增水，输出 `T` 的增水。滚动预报将该预测值放回历史窗口，并使用下一时段 ERA5 强迫继续预测，支持 12、24、48、72 小时。

模型包含三组 `Conv2d → BatchNorm → ReLU → MaxPool` 的大气分支、处理增水历史的 MLP 分支，以及融合全连接层。大气输入通道数自动取 `input_steps × 3`，默认形状为 `(batch, 72, 40, 40)`；输出为 `(batch,)`。

## 数据与配置

原始数据只保存在外部盘 `F:\data`，程序不会复制或移动文件。先执行：

```powershell
python caribbean_short_term_forecast\src\inspect_data.py --data-root "F:\data"
```

清单写入 `outputs/data_inventory.csv`，只读取文件路径、名称、后缀与大小，不载入大文件。它会标记可能的 GESLA metadata/单站记录、PSMSL/NOC 文件以及 ERA5 NetCDF/GRIB。若外部盘未连接，会明确提示：`Data root F:\data is not available. Please connect the external drive.`

根据清单把核实后的站点信息填入 `configs/stations.yaml`。Prickly Bay 的经纬度和实际路径目前故意为 `null`；UTide、ERA5 或训练需要这些值时会停止并说明缺少哪一项，不会静默采用猜测值。默认超参数位于 `configs/default.yaml`。

建议的数据盘结构（实际子目录名称可以不同）：

```text
F:\data\
├── GESLA\...                 # metadata、单站记录
├── PSMSL_NOC\...             # Prickly Bay 等站
└── ERA5\prickly_bay\...      # 按年或按月 NetCDF/GRIB
```

所有 `outputs/`、`models/`、`figures/`、`logs/`、缓存、CSV、NetCDF/GRIB、数组和权重均被模块 `.gitignore` 排除。不要把真实数据复制到仓库。

## 预处理流程

1. `tide_gauge_loader.py` 统一读取 GESLA 4.0、PSMSL/NOC，以及常见 CSV、TXT、空格或制表符分隔文件。输出统一字段；通过配置把 m/cm/mm 换算成 m。
2. `tide_quality_control.py` 解析时间、排序、去重、过滤缺测与质量标志，并用宽松的绝对物理范围检查异常。多传感器不会直接混合，而是分别计算完整性和连续性，选择得分最高的通道，生成 QC JSON。
3. `tide_processing.py` 对 QC 后的观测用站点真实纬度重新运行 UTide。PSMSL/NOC 自带 tide estimate 只可辅助比较，不作为正式标签。有效数据不足 30 天时停止，不生成伪结果。输出 `cleaned_water_level.csv`、`tide_reconstruction.csv`、`hourly_storm_surge.csv` 和诊断 JSON。
4. `era5_loader.py` 识别 U10、V10、MSL 及坐标别名，统一 0—360/-180—180 经度和纬度方向，裁剪站点周围约 10°×10°。原裁剪恰好 40×40 时不插值，否则重采样到 40×40。每次只打开一个年/月文件，可按期迭代并缓存到 `outputs/cache/`。
5. `time_alignment.py` 将两类时间显式转换到 UTC 并按整点精确求交，报告覆盖范围、缺失时次和连续段。时区未核实时不会把 naive 时间擅自当成 UTC。
6. `dataset_builder.py` 按需生成 `(72,40,40)`、`(24,)` 和标量标签，跳过含断点或 NaN 的窗口。数据按时间前 80%/后 20% 划分，scaler 只在训练段拟合。大规模实跑应保存分年/月缓存，并用 Dataset、memmap 或分块文件按需读取。

完成站点配置后执行完整预处理：

```powershell
python caribbean_short_term_forecast\src\prepare_station.py --station prickly_bay
```

该入口顺序执行验潮读取、QC、UTide、逐年/月 ERA5 裁剪和整点匹配。它预先创建 `.npy` memory map，再把每个 ERA5 文件的匹配时次增量写入，不会 `np.stack` 多年大气场。生成的 `aligned_dataset/` 包含：

```text
atmosphere.npy  (time, 3, 40, 40), float32，变量顺序 U10/V10/MSL
surge.npy       (time,), float32，单位 m
time.npy        (time,), datetime64，逐小时 UTC
```

默认位置为 `outputs/processed/<station>/aligned_dataset/`，不会被 Git 跟踪。训练也兼容包含同名三个数组的小型 NPZ smoke-test 文件；正式多年训练优先使用 memory map 目录。

## 训练

完整训练：

```powershell
python caribbean_short_term_forecast\src\train_station.py --station prickly_bay --start-year 2011 --end-year 2018 --input-steps 24 --epochs 50 --batch-size 16 --lr 0.001
```

先做 2011 单年 smoke test：

```powershell
python caribbean_short_term_forecast\src\train_station.py --station prickly_bay --start-year 2011 --end-year 2011 --input-steps 24 --epochs 2 --batch-size 8 --smoke-test
```

训练优先使用 CUDA，默认 Adam + MSELoss，也可传 `--optimizer sgd`。固定随机种子并启用 early stopping。checkpoint 保存模型名、输入步数、变量、网格、scaler、站点和训练时段。输出包括 `best_model.pth`、`metrics.json`、`val_predictions.csv`、`loss_history.csv`、`training_config.json` 以及两张验证 PNG。指标和图中的增水统一为 cm：Pearson r、RMSE、MAE、Bias、RRMSE。

## 滚动预报

```powershell
python caribbean_short_term_forecast\src\rolling_forecast.py --station prickly_bay --model-path "caribbean_short_term_forecast\models\prickly_bay\best_model.pth" --start-time "2018-01-01 00:00" --forecast-steps 12
```

未来每个小时必须有 ERA5 强迫，历史 24 小时必须有已知增水。目标时段没有观测仍会输出预测，但跳过相应评价。输出 `rolling_forecast.csv`、`rolling_metrics.json`、`rolling_forecast.png`、`rolling_error.png` 和 `cumulative_error.png`。CSV 字段为 `datetime, lead_time, observed, predicted, error, absolute_error`。所有图片仅为 PNG，默认 400 dpi。

## 测试

```powershell
python -m py_compile caribbean_short_term_forecast\src\*.py
pytest caribbean_short_term_forecast\tests
```

测试使用小型伪验潮表和随机数组，不包含真实站点数据；覆盖 GESLA/CSV 读取、QC、UTC 时间匹配、方案 B 样本形状、断时跳过及 `(2,72,40,40) + (2,24) → (2,)` 模型 forward。

## 排错与后续扩展

外部盘断开时先在资源管理器确认盘符，再重跑 `inspect_data.py`；若盘符变化，显式传新的 `--data-root`，不要修改代码里的路径。GRIB 无法打开时检查 `cfgrib/eccodes`；时间对不上时先核实原始站点时区和日志中的双方覆盖范围；空裁剪通常表示站点经纬度或 ERA5 经度范围配置错误。

站点代码没有复制：`station_registry.py` 提供统一入口，模型 checkpoint 保留站点元数据。后续计划依次为 Prickly Bay 2011 smoke test、2011—2018 正式训练、12/24/48/72h 滚动预报、San Juan 长记录训练，再以 Charlotte Amalie 等站做跨站验证。单站训练已实现；跨站测试、多站联合训练和迁移学习应在现有 Dataset/registry 接口上增加采样策略，不需要另复制一套流水线。
