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

清单写入 `outputs/data_inventory.csv`，只读取文件路径、名称、后缀与大小，不载入大文件，并自动忽略 macOS 生成的全部 `._*` 伴生文件。它会标记可能的 GESLA metadata/单站记录、PSMSL/NOC 文件以及 ERA5 NetCDF/GRIB。若外部盘未连接，会明确提示：`Data root F:\data is not available. Please connect the external drive.`

Prickly Bay 已按 PSMSL/NOC 权威资料配置为 `12.005°N, 61.765°W`、UTC、米；三个传感器路径和 `Regional_Union` ERA5 路径均已写入 `configs/stations.yaml`。三个传感器的高度零点不保证一致，因此正式流程不跨通道拼接水位，当前固定采用 QC 得分最高的雷达通道 `rad`。默认超参数位于 `configs/default.yaml`。

建议的数据盘结构（实际子目录名称可以不同）：

```text
F:\data\
├── GESLA4_0\...                              # metadata、单站记录
├── caribbean_tide_gauges\pric\pric_*.csv    # Prickly Bay 三通道与潮汐估计
└── ERA5-Caribbean\Regional_Union\...         # 按变量、按年份 NetCDF
```

所有 `outputs/`、`models/`、`figures/`、`logs/`、缓存、CSV、NetCDF/GRIB、数组和权重均被模块 `.gitignore` 排除。不要把真实数据复制到仓库。

## 预处理流程

1. `tide_gauge_loader.py` 统一读取 GESLA 4.0、PSMSL/NOC，以及常见 CSV、TXT、空格或制表符分隔文件。输出统一字段；通过配置把 m/cm/mm 换算成 m。
2. `tide_quality_control.py` 解析时间、排序、去重、过滤缺测与质量标志，并用宽松的绝对物理范围检查异常。多传感器不会直接混合，而是分别计算完整性和连续性，选择得分最高的通道，生成 QC JSON。
3. `tide_processing.py` 对 QC 后的观测用站点真实纬度重新运行 UTide。PSMSL/NOC 自带 tide estimate 只可辅助比较，不作为正式标签。有效数据不足 30 天时停止，不生成伪结果。输出 `cleaned_water_level.csv`、`tide_reconstruction.csv`、`hourly_storm_surge.csv` 和诊断 JSON。
4. `era5_loader.py` 识别 U10、V10、MSL 及坐标别名，支持同一年三个变量分文件保存；`prepare_station.py` 先按文件名年份分组，再核对三份文件的时间和网格完全一致后合并。随后统一 0—360/-180—180 经度和纬度方向，裁剪站点周围约 10°×10°，并重采样到 40×40。每次只加载一个年份，避免把多年数据同时放入内存。
5. `time_alignment.py` 将两类时间显式转换到 UTC 并按整点精确求交，报告覆盖范围、缺失时次和连续段。时区未核实时不会把 naive 时间擅自当成 UTC。
6. `dataset_builder.py` 按需生成 `(72,40,40)`、`(24,)` 和标量标签，跳过含断点或 NaN 的窗口。数据按时间前 80%/后 20% 划分，scaler 只在训练段拟合。大规模实跑应保存分年/月缓存，并用 Dataset、memmap 或分块文件按需读取。

完整数据生成后可先执行审计：

```powershell
python caribbean_short_term_forecast\src\audit_prepared_dataset.py
```

审计会检查数组形状、磁盘大小、逐小时连续性、三变量有限值比例与物理范围、MSL是否仍为Pa，以及整体和逐年的有效24小时样本数。

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

正式年份切分训练（推荐）：

```powershell
python caribbean_short_term_forecast\src\train_station.py --station prickly_bay --split-mode years --train-start-year 2011 --train-end-year 2016 --validation-year 2017 --test-year 2018 --model-type dual --input-steps 24 --epochs 50 --batch-size 16 --lr 0.001
```

该模式按目标时刻年份切分，2011—2016用于训练、2017用于验证和early stopping、2018只用于最终测试；三组共用仅在训练年份拟合的scaler。`--model-type` 支持完整双分支 `dual`、只用ERA5的 `era5_cnn` 和只用历史增水的 `surge_mlp`。

先做 2011 单年 smoke test：

```powershell
python caribbean_short_term_forecast\src\train_station.py --station prickly_bay --start-year 2011 --end-year 2011 --input-steps 24 --epochs 2 --batch-size 8 --smoke-test
```

在正式CNN训练前运行简单基线：

```powershell
python caribbean_short_term_forecast\src\evaluate_baselines.py --station prickly_bay --train-start-year 2011 --train-end-year 2016 --validation-year 2017 --test-year 2018
```

基线包括零增水、上一小时持久性，以及岭回归。岭回归输入为过去24小时每个ERA5变量的空间均值、标准差、最小值、最大值，加过去24小时增水，共312个特征。

训练优先使用 CUDA，默认 Adam + MSELoss，也可传 `--optimizer sgd`。固定随机种子并启用 early stopping；正式年份切分只根据2017验证损失选择checkpoint，2018测试集不参与模型选择。checkpoint 保存模型名、输入步数、变量、网格、scaler、站点和训练时段。

每个模型输出 `best_model.pth`、`metrics.json`、`dataset_report.json`、`loss_history.csv`、`loss_curve.png`、训练配置、验证/测试预测CSV、时间序列图和散点图。评价包括全部时段的Pearson r、R²、RMSE、MAE、Bias、RRMSE、相对岭回归技能评分，以及绝对增水Top 10%/5%、快速上涨、分离峰值幅度误差和峰值时间误差。

完成三个seed-42模型后生成统一比较：

```powershell
python caribbean_short_term_forecast\src\summarize_first_round.py
```

## 滚动预报

```powershell
python caribbean_short_term_forecast\src\rolling_forecast.py --station prickly_bay --model-path "caribbean_short_term_forecast\models\prickly_bay\best_model.pth" --start-time "2018-01-01 00:00" --forecast-steps 12
```

对已训练的 seed 42 模型执行严格限制在2017验证集内的共同起报、72小时递归诊断：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\rolling_diagnostics.py --station prickly_bay --device cuda --batch-size 256
```

该实验使用已知未来 ERA5 再分析强迫，名称为“已知未来大气强迫条件下的历史滚动回算”，不能表述为业务预报。脚本硬性禁止加载2018数据；岭回归、surge_mlp和双分支模型均只回填自身预测值。

建立连续未来24小时标签，并仅使用2017验证集比较直接多步线性基线：

```powershell
python caribbean_short_term_forecast\src\direct_multistep_baselines.py --station prickly_bay
```

正则系数使用2011—2015拟合、2016选择，再用2011—2016重拟合。脚本同时比较AR-Ridge、ERA5-Ridge、Combined-Ridge，以及过去ERA5和过去＋未来ERA5两种强迫设置；不读取2018直接多步结果。

使用GPU训练直接24小时XGBoost、MLP、GRU、Dual-CNN-Past和Dual-CNN-Future，并统一在2017比较：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\train_direct_multimodel.py --station prickly_bay --device cuda
```

XGBoost使用24个独立模型；其余网络一次输出连续24小时。所有模型共享相同起报时刻，重点评价1、3、6、12、24小时，脚本硬性禁止加载2018。

训练一步XGBoost并在相同2017起报时刻统一比较Ridge、XGBoost和Dual-CNN的直接预测与滚动24小时预测：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\rolling_24_comparison.py --station prickly_bay --device cuda
```

滚动模型只回填自身预测增水，并逐时读取未来ERA5再分析真值；实验属于已知未来大气强迫历史回算。该入口不会修改旧滚动代码，也不会加载2018。

对XGBoost和匹配结构CNN执行Surge-only、Past-ERA5、Future-ERA5最小消融：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\era5_ablation.py --station prickly_bay --device cuda
```

CNN-Surge保留Dual-CNN相同的增水分支和融合输出头，仅移除ERA5分支。三个信息版本共享相同起报时刻、训练年份、seed、普通MSE和早停标准；不加载2018。

统一比较Persistence、Ridge、XGBoost和Dual-CNN的72小时递归误差：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\rolling_72_comparison.py --station prickly_bay --device cuda
```

四个模型严格使用相同2017起报时刻，输出1、3、6、12、24、48和72小时的RMSE、MAE、Bias、Pearson r、Top 5%及快速上涨指标；脚本不加载2018。

从一步Dual-CNN初始化，冻结ERA5空间编码器并进行6步递归微调：

```powershell
E:\condaData\envs_dirs\mygpu\python.exe caribbean_short_term_forecast\src\train_rollout_dual.py --station prickly_bay --device cuda --rollout-steps 6
```

训练时连续展开6步，对所有步共同计算MSE，并使用从真值历史逐渐过渡到模型预测历史的scheduled sampling。该版本仅用2017递归损失选checkpoint；训练完成后可向`rolling_72_comparison.py`传入`--rollout-checkpoint`，在相同起报时刻与原模型比较。

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
