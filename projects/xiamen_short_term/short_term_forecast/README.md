# 中国沿海多站点小时级短时风暴增水预测

正式实验统一使用 `src/xiamen_forecast/`。这个包名为兼容已有厦门命令而保留，内部流程现已支持 `xiamen`、`lianyungang` 和 `beihai`，没有为新站点复制第二套项目。`src/short_term_forecast/` 仅保留期刊绘图和公用指标。

## 多站点配置与实验边界

站点、经纬度、原始文件名、年份划分和独立输出目录集中定义在 `src/xiamen_forecast/station_config.py`：

| station参数 | 站点 | 数据年份 | 训练 | 验证 | 独立测试 |
|---|---|---:|---:|---:|---:|
| `xiamen` | 厦门 | 1970—1997 | 1970—1995 | 1996 | 1997 |
| `lianyungang` | 连云港 | 1975—1997 | 1975—1995 | 1996 | 1997 |
| `beihai` | 北海 | 1975—1997 | 1975—1995 | 1996 | 1997 |

固定规范如下：

1. 输入为过去24小时的U10、V10、MSL 40×40网格及历史增水；ERA5变量顺序固定为U10、V10、MSL。
2. UTide只用训练期观测标定；标准化参数只用训练集计算。
3. 1996只用于模型选择和早停；1997只在方案锁定后运行一次独立测试。
4. 递归滚动会回填模型自身预测增水，不会用未来真实增水；未来ERA5仍为再分析真值，所以属于历史回算，不是业务实时预报。
5. 原始目录 `F:\ERA5-NEW` 和 `F:\GESLA` 只读。所有处理数组、模型和实验结果都写入本项目自己的 `outputs/`、`models/` 和仓库根目录 `reports/experiment_results/`。
6. 真实NetCDF、GESLA、NPY、模型权重和大型预测数组不提交GitHub。

## 实验室阶段A/B操作

以下命令使用Windows PowerShell，不要写CMD的 `cd /d`。先更新本功能分支并进入项目：

```powershell
conda activate jjq
Set-Location "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git status
git fetch origin
git checkout codex/china-multistation
git pull --ff-only origin codex/china-multistation
Set-Location "projects\xiamen_short_term\short_term_forecast"
python -m compileall -q src
python -m pytest tests -q
```

先处理连云港。命令会从集中配置读取用户给定的三个ERA5文件和GESLA文件，逐年读取ERA5并写入独立目录，不会修改原始数据：

```powershell
python -m src.xiamen_forecast.prepare_xiamen --station lianyungang
python -m src.xiamen_forecast.audit_prepared_dataset --station lianyungang
```

第一条成功后预期生成：

```text
outputs/processed/lianyungang/tide_qc_report.json
outputs/processed/lianyungang/preparation_report.json
outputs/processed/lianyungang/tide/
outputs/processed/lianyungang/aligned_dataset/atmosphere.npy
outputs/processed/lianyungang/aligned_dataset/surge.npy
outputs/processed/lianyungang/aligned_dataset/time.npy
outputs/processed/lianyungang/aligned_dataset/dataset_metadata.json
```

第二条生成 `outputs/processed/lianyungang/prepared_dataset_audit.json`。只有报告的 `status` 为 `passed`，且ERA5三个变量坐标一致、MSL单位为Pa、`5|0`等无效GESLA记录已剔除、三个数据集均有有效样本、1996和1997均有72小时共同起报样本时，才进入训练。严重问题会让命令以错误退出，不会静默插值或伪造结果。

北海命令已经准备好，但默认等待连云港审计确认后再执行：

```powershell
python -m src.xiamen_forecast.prepare_xiamen --station beihai
python -m src.xiamen_forecast.audit_prepared_dataset --station beihai
```

北海对应输出位于 `outputs/processed/beihai/`，不会读取或覆盖连云港、厦门数组。若需要查看失败报告而不让程序抛出异常，可临时加 `--report-only`；这不代表数据合格，也不能据此开始训练。

## 审计确认后的阶段C命令

本节仅供后续使用，本轮不要运行。以连云港为例，先在1996验证集完成基线和核心模型选择：

```powershell
python -m src.xiamen_forecast.evaluate_baselines --station lianyungang --validation-only
python -m src.xiamen_forecast.train_xiamen --station lianyungang --model-type surge_mlp --validation-only --device cuda
python -m src.xiamen_forecast.train_xiamen --station lianyungang --model-type era5_cnn --validation-only --device cuda
python -m src.xiamen_forecast.train_model_suite --station lianyungang --models cnn cnn_gru --device cuda
python -m src.xiamen_forecast.compare_models --station lianyungang --split validation
```

确认CNN-GRU为主模型后再做rollout-6与1996年的1、3、6、12、24、48、72小时诊断：

```powershell
python -m src.xiamen_forecast.train_rollout_temporal --station lianyungang --model-type cnn_gru --rollout-steps 6 --device cuda
python -m src.xiamen_forecast.rolling_diagnostics --station lianyungang --evaluation-year 1996 --split validation --models surge_mlp era5_cnn cnn cnn_gru --device cuda --rollout-checkpoint "models\lianyungang\formal_seed42\cnn_gru_rollout6\best_model.pth"
```

模型方案锁定后，下面这一条才读取1997观测并评估已锁定权重；它不会重新训练：

```powershell
python -m src.xiamen_forecast.run_final_test --station lianyungang --device cuda --language en
```

精简结果最终位于仓库根目录 `reports/experiment_results/lianyungang_short_term_1997_seed42/`。北海只需把所有命令中的 `lianyungang` 换成 `beihai`。

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
cd H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast
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
python -m src.xiamen_forecast.evaluate_baselines --validation-only
```

4. 训练消融模型。`surge_mlp` 只使用历史增水，`era5_cnn` 只使用 ERA5；它们用于判断两类信息的独立贡献，不属于五模型主体对比：

```powershell
python -m src.xiamen_forecast.train_xiamen --model-type surge_mlp --validation-only
python -m src.xiamen_forecast.train_xiamen --model-type era5_cnn --validation-only
```

5. 在完全相同的数据划分、24 小时输入和评价规则下训练五个正式对比模型：

```powershell
python -m src.xiamen_forecast.train_xiamen --model-type cnn --batch-size 256 --epochs 50 --patience 8 --validation-only
python -m src.xiamen_forecast.train_xiamen --model-type cnn_lstm --batch-size 32 --epochs 50 --patience 8 --validation-only
python -m src.xiamen_forecast.train_xiamen --model-type cnn_gru --batch-size 32 --epochs 50 --patience 8 --validation-only
python -m src.xiamen_forecast.train_xiamen --model-type tcn --batch-size 32 --epochs 50 --patience 8 --validation-only
python -m src.xiamen_forecast.train_xiamen --model-type transformer --batch-size 32 --epochs 50 --patience 8 --validation-only
```

也可以一次按顺序运行五个模型；已经生成 `metrics.json` 的完整实验会自动跳过，实验中断后可重复执行同一命令继续后续任务：

```powershell
python -m src.xiamen_forecast.train_model_suite --device cuda
```

`cnn` 是“空间 CNN + 历史增水 MLP”的正式基础模型。旧命令中的 `dual` 仍可读取，但只是 `cnn` 的兼容别名，不应重复训练。所有模型默认在 CUDA 上启用 AMP 混合精度；若显存充足，可逐步把时序模型 batch size 从 32 调到 64 或 128。Windows 的 `--num-workers` 默认是 0，确认运行稳定后可尝试 `--num-workers 2`。

模型使用 early stopping。模型选择阶段加 `--validation-only`，只输出1996验证结果，不加载1997；锁定方案后由 `evaluate_checkpoints` 或 `run_final_test` 评估1997。指标包括RMSE、MAE、Bias、Pearson r、R2、RRMSE、Top 10%/5%强增水、快速上涨和独立峰值误差。

五个模型跑完后，先汇总 1996 验证集的一步预测结果：

```powershell
python -m src.xiamen_forecast.compare_models --split validation
```

在模型方案完全确定前，不要用 `--split test` 反复挑模型，以免把 1997 独立测试年变成调参数据。

6. 1996 验证集上 CNN-GRU 的一步预测 RMSE 最低，因此以它作为主模型进行 6 步递归损失微调。程序从已有 CNN-GRU 权重继续训练，预计算并冻结耗时最大的逐小时气象编码器，只更新 GRU 和回归头，因此不是从头训练。训练前会先记录未微调模型的6步递归基线，teacher forcing在前5轮降到0，之后才启用early stopping；如果微调没有改善，原始权重会自动保留为最佳checkpoint：

```powershell
python -m src.xiamen_forecast.train_rollout_temporal --model-type cnn_gru --device cuda --rollout-steps 6
```

同一入口也支持 `cnn_lstm`、`tcn` 和 `transformer`，但不应在查看1997测试结果后反复挑选；当前主实验只微调由1996验证集选出的CNN-GRU。原来的 `train_rollout_cnn` 保留用于普通CNN对照。

7. 所有一步模型完成后，在 1996 验证年统一比较 1、3、6、12、24、48、72 小时递归回算：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --device cuda
```

如已完成 6 步微调，可把它加入同一张对比表：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --device cuda --rollout-checkpoint "models\xiamen\formal_seed42\cnn_gru_rollout6\best_model.pth"
```

滚动实验使用未来时次的 ERA5 再分析场，因此应称为“已知未来大气强迫条件下的历史回算”，不能表述为业务实时预报。模型和方案已在1996验证集上锁定，1997独立测试已经完成；后续不得再根据1997结果反复选择模型或调整超参数。

模型方案固定后，运行一次1997独立测试总流程。该入口不会重新训练模型，会依次汇总已有一步测试指标、运行1至72小时滚动回算、生成1997测试图，并导出Git安全结果包：

```powershell
python -m src.xiamen_forecast.run_final_test --device cuda --language en
```

1997输出分别保存到：

```text
outputs/experiments/xiamen/model_comparison/test_model_metrics.csv
outputs/experiments/xiamen/rolling_1997_seed42/
outputs/journal_figures/xiamen_1997_seed42/
仓库根目录/reports/experiment_results/xiamen_short_term_1997_seed42/
```

如中途失败，可以重复运行同一命令；已有模型权重不会被改写。1997结果只用于最终独立评价，不应再据此选择模型或调整超参数。

8. 滚动实验完成后，导出可安全提交到Git的小型结果包。脚本只收集指标、摘要和关键图片，不复制模型权重、原始数据或大型逐时预测表：

```powershell
python -m src.xiamen_forecast.export_final_results
```

结果保存到仓库根目录的 `reports/experiment_results/xiamen_short_term_1996_seed42/`。确认内容后，可在当前目录执行：

```powershell
git -C ..\..\.. add reports/experiment_results/xiamen_short_term_1996_seed42
git -C ..\..\.. commit -m "Archive Xiamen short-term forecast results"
git -C ..\..\.. push
```

9. 在实验室电脑上生成厦门正式汇报和论文图。该命令只读取已经存在的指标、预测CSV和图片，不加载大型ERA5数据、不读取模型权重，也不会重新训练：

```powershell
python -m src.short_term_forecast.journal_figures.make_xiamen_journal_figures
```

默认输出到：

```text
outputs/journal_figures/xiamen_1996_seed42/
```

其中包括一步模型对比、RMSE随提前量变化、CNN-GRU递归微调收益、Top 5%强增水与快速上涨指标、训练过程、验证散点、不同提前量滚动散点、残差诊断、峰值分析和已有强事件图。`figure_manifest.csv`记录每张图的状态；缺少某个预测CSV时只跳过对应图，不影响其他图。

如需中文图题：

```powershell
python -m src.short_term_forecast.journal_figures.make_xiamen_journal_figures --language zh
```

图片确认无误后，再运行一次结果导出。导出脚本会把`fig*.png`、manifest、图件说明和诊断JSON复制到Git可提交的精简结果包，不复制逐时预测表：

```powershell
python -m src.xiamen_forecast.export_final_results
```

这一版已经接入 CNN、CNN-LSTM、CNN-GRU、TCN、Transformer，以及普通CNN和时序模型的递归微调。加勒比项目中的直接 24 小时表格模型、ERA5 消融扩展和物理残差融合尚未直接复制到厦门；应先完成上述统一模型对比，再根据 1996 验证结果决定是否迁移。

## 测试

Mac 本地已建立独立环境 `jjq`，位置为：

```text
/Users/jjq/Documents/storm_surge/.venvs/jjq
```

以后在 Mac 上检查代码、运行单元测试或生成轻量图件时，先执行：

```bash
source /Users/jjq/Documents/storm_surge/.venvs/jjq/bin/activate
cd /Users/jjq/Documents/storm_surge/storm_surge_clean/projects/xiamen_short_term/short_term_forecast
python -m compileall -q src
python -m pytest tests -q
```

环境包含 PyTorch、NumPy、Pandas、SciPy、scikit-learn、Matplotlib、xarray、NetCDF4、UTide、cfgrib/eccodes、CDS API、Jupyter 和 pytest。当前 Mac 的 PyTorch 不支持 MPS，因此本地主要用于测试和轻量结果处理；正式模型训练继续使用实验室电脑的 RTX 4090。退出环境使用 `deactivate`。

重新建立该环境时可执行：

```bash
python3 -m venv /Users/jjq/Documents/storm_surge/.venvs/jjq
source /Users/jjq/Documents/storm_surge/.venvs/jjq/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

实验室 Windows 电脑继续使用已有的 Conda `jjq` 环境；两个环境名称相同，但位于不同电脑，互不影响。

通用测试命令：

```powershell
python -m compileall -q src
python -m pytest tests -q
```

以下内容不会提交到 GitHub：真实 NetCDF、验潮数据、对齐数组、模型权重和新实验输出。
