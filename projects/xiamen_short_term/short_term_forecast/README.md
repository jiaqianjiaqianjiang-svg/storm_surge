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

审计通过后，推荐用一条命令完成连云港1996验证流程。它会依次执行基线、Surge-MLP、ERA5-only CNN、Fusion CNN、CNN-LSTM、CNN-GRU、TCN、Transformer、rollout-6、72小时滚动诊断、绘图和精简结果导出，与厦门正式模型体系保持一致：

```powershell
python -m src.xiamen_forecast.run_validation_workflow --station lianyungang --device cuda
```

整个过程可能持续数小时，主要耗时在深度模型训练。命令可以中断后重新执行；再次运行时会根据结果标记跳过已完成步骤，并在新增或更新模型后自动刷新下游比较、滚动、绘图和导出。加 `--force` 才会强制重跑。该入口只使用训练期和1996验证集，不读取1997测试观测。

最终可上传的小型结果包位于仓库根目录：

```text
reports/experiment_results/lianyungang_short_term_1996_seed42/
```

回到仓库根目录后只提交这个目录，不要使用 `git add .`：

```powershell
Set-Location "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git status
git add reports/experiment_results/lianyungang_short_term_1996_seed42
git commit -m "Archive Lianyungang 1996 validation results"
git push origin codex/china-multistation
```

如需逐项排查，仍可使用下面的等价手动命令：

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

## 三站1997独立测试汇总

厦门、连云港和北海的1997精简结果包全部上传后，在仓库根目录运行：

```powershell
python tools\compare_china_short_term_stations.py --language en
```

该命令只读取 `reports/experiment_results/*_short_term_1997_seed42/`，不会加载原始数据、重新训练或改写单站结果。统一输出位于：

```text
reports/experiment_results/china_three_station_comparison_1997_seed42/
```

其中包含一步模型指标、1—72小时滚动指标、P90/P95/P99峰值指标、核心摘要、三张跨站对比图和中文说明。默认推荐英文图片，避免不同电脑缺少中文字体；Windows已安装微软雅黑时可将参数改为 `--language zh`。

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

## 连云港正向极值损失消融（仅1996方法开发）

这一组实验保持CNN-GRU结构、24小时输入、40×40 ERA5、数据划分、优化器、学习率、随机种子和early stopping规则不变，只替换训练损失。`MSE`是原始基线；`WMSE`（weighted MSE，加权均方误差）把训练期P90至P99正向增水的权重从1线性提高到3；`AWMSE`（asymmetric weighted MSE，不对称加权均方误差）进一步惩罚正向强增水低估；`WHuber`（weighted Huber，加权Huber损失）是稳健性对照。

权重阈值只使用1975—1995训练标签并以米计算。若原始P90不大于0，程序使用`q90=max(P90, 0 m)`，因此零及负增水始终保持单位权重。WHuber默认物理转折尺度为5 cm，运行时按训练集增水标准差换算到标准化尺度。所有阈值、单位、样本数和换算值均写入`extreme_loss_config.json`与checkpoint。

新实验仍以1996普通验证MSE早停，同时记录加权训练和验证目标；加权损失不能直接解释为普通RMSE。1997不参与本轮训练、早停或超参数选择。

### 1. 实验室更新与检查

```powershell
conda activate jjq
Set-Location "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git checkout codex/china-multistation
git pull origin codex/china-multistation
Set-Location "projects\xiamen_short_term\short_term_forecast"
python -m compileall -q src
python -m pytest tests -q
```

### 2. 一步预测损失消融

现有`models\lianyungang\formal_seed42\cnn_gru\`是MSE基线，不应覆盖。依次运行三个新实验：

```powershell
python -m src.xiamen_forecast.train_xiamen --station lianyungang --model-type cnn_gru --loss-type wmse --extreme-alpha 2 --batch-size 32 --epochs 50 --patience 8 --lr 0.001 --optimizer adam --seed 42 --device cuda --validation-only

python -m src.xiamen_forecast.train_xiamen --station lianyungang --model-type cnn_gru --loss-type awmse --extreme-alpha 2 --extreme-beta 1 --batch-size 32 --epochs 50 --patience 8 --lr 0.001 --optimizer adam --seed 42 --device cuda --validation-only

python -m src.xiamen_forecast.train_xiamen --station lianyungang --model-type cnn_gru --loss-type whuber --extreme-alpha 2 --huber-delta-cm 5 --batch-size 32 --epochs 50 --patience 8 --lr 0.001 --optimizer adam --seed 42 --device cuda --validation-only
```

输出严格隔离在：

```text
models/lianyungang/formal_seed42/cnn_gru_extreme_wmse/
models/lianyungang/formal_seed42/cnn_gru_extreme_awmse/
models/lianyungang/formal_seed42/cnn_gru_extreme_whuber/
```

汇总全部已完成的一步实验：

```powershell
python -m src.xiamen_forecast.compare_extreme_losses --station lianyungang
```

主要检查：

```text
outputs/experiments/lianyungang/extreme_loss_ablation_1996_seed42/validation_extreme_loss_summary.csv
outputs/experiments/lianyungang/extreme_loss_ablation_1996_seed42/validation_positive_peak_events.csv
outputs/experiments/lianyungang/extreme_loss_ablation_1996_seed42/validation_extreme_loss_details.json
outputs/experiments/lianyungang/extreme_loss_ablation_1996_seed42/experiment_metadata.json
outputs/experiments/lianyungang/extreme_loss_ablation_1996_seed42/validation_extreme_loss_comparison.png
```

先比较整体RMSE代价、正向P99 RMSE、P99 bias、低估比例和独立正向事件峰值误差。预设实用目标是P99 RMSE降低至少10%、整体RMSE恶化不超过5%；这是筛选规则，不是预先保证的结果。

### 3. rollout-6两阶段消融（一步结果确认后再运行）

下列四条路线分别隔离“一步损失”和“递归微调损失”的贡献。第一条已有，无需重跑。其余命令只能在人工确认一步结果后执行：

```powershell
# 已有基线：MSE一步 + MSE rollout-6
# models\lianyungang\formal_seed42\cnn_gru_rollout6\best_model.pth

# WMSE一步 + MSE rollout-6
python -m src.xiamen_forecast.train_rollout_temporal --station lianyungang --model-type cnn_gru --base-checkpoint "models\lianyungang\formal_seed42\cnn_gru_extreme_wmse\best_model.pth" --loss-type mse --rollout-steps 6 --epochs 20 --patience 5 --teacher-forcing-epochs 5 --learning-rate 0.0001 --seed 42 --device cuda

# WMSE一步 + WMSE rollout-6
python -m src.xiamen_forecast.train_rollout_temporal --station lianyungang --model-type cnn_gru --base-checkpoint "models\lianyungang\formal_seed42\cnn_gru_extreme_wmse\best_model.pth" --loss-type wmse --extreme-alpha 2 --rollout-steps 6 --epochs 20 --patience 5 --teacher-forcing-epochs 5 --learning-rate 0.0001 --seed 42 --device cuda

# AWMSE一步 + AWMSE rollout-6
python -m src.xiamen_forecast.train_rollout_temporal --station lianyungang --model-type cnn_gru --base-checkpoint "models\lianyungang\formal_seed42\cnn_gru_extreme_awmse\best_model.pth" --loss-type awmse --extreme-alpha 2 --extreme-beta 1 --rollout-steps 6 --epochs 20 --patience 5 --teacher-forcing-epochs 5 --learning-rate 0.0001 --seed 42 --device cuda
```

自动目录分别为：

```text
cnn_gru_extreme_wmse_rollout6_mse/
cnn_gru_extreme_wmse_rollout6_wmse/
cnn_gru_extreme_awmse_rollout6_awmse/
```

在完全相同的1996共同起报时刻比较已有基线和新checkpoint：

```powershell
python -m src.xiamen_forecast.rolling_diagnostics --station lianyungang --evaluation-year 1996 --split validation --models cnn_gru --device cuda --rollout-checkpoint "models\lianyungang\formal_seed42\cnn_gru_rollout6\best_model.pth" --extra-checkpoint "wmse_one_mse_rollout=models\lianyungang\formal_seed42\cnn_gru_extreme_wmse_rollout6_mse\best_model.pth" --extra-checkpoint "wmse_one_wmse_rollout=models\lianyungang\formal_seed42\cnn_gru_extreme_wmse_rollout6_wmse\best_model.pth" --extra-checkpoint "awmse_one_awmse_rollout=models\lianyungang\formal_seed42\cnn_gru_extreme_awmse_rollout6_awmse\best_model.pth" --output-dir "outputs\experiments\lianyungang\extreme_rollout_ablation_1996_seed42"
```

`rolling_metrics_long.csv`包含1、3、6、12、24、48、72小时整体指标、历史兼容的绝对Top 5%指标，以及新增的正向P90/P95/P99 RMSE、bias、低估比例、平均低估和平均高估幅度。`rolling_predictions_selected_leads.csv`保留模型配对所需的共同起报和验证时刻。

方法选择阶段只提交上述小型CSV、JSON和PNG供检查，不提交GESLA、ERA5、NPY、逐时大型数组或`.pth`权重。只有连云港1996结果通过人工检查后，才考虑rollout消融、其他站复验及回顾性的1997评价。
