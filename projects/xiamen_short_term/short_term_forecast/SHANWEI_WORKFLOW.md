# 汕尾站短时增水实验：实验室操作说明

本流程复用 `src/xiamen_forecast/` 已验证的预处理、质量审计、Persistence/Ridge/CNN-GRU、六步递归微调、1996验证、1997逐时1–72小时滚动回算、IBTrACS候选筛选和固定起报台风案例代码。厦门、连云港、北海的默认完整模型流程不变。汕尾未在Mac上读取到原始数据；**尚无汕尾真实训练或误差结果**。

## 执行原则

- 原始 `F:` 盘GESLA和ERA5只读。所有处理数组位于项目内 `outputs/processed/shanwei/`，模型位于 `models/shanwei/`，不会写入另外三站目录。
- 汕尾的经纬度与GESLA/U10/V10/MSL准确文件名须由实验室核对后输入，不能凭文件夹名称猜测。
- 源审计读取实际时轴和质量标志，要求1996、1997共同有效小时覆盖各不低于80%，且训练期至少有5个年份达到该阈值；训练起点取最早的合格年份，不预设1975年。训练期间个别普通缺测不会自动截断全部早期数据，不能被插值伪造；不连续窗口在样本构建时剔除。处理后审计允许不超过总小时数1%的已标记ERA5未匹配时次，并在`notes`中报告数量；更大缺口仍拦截。后续准备数据审计必须显示 `status=passed`。
- 1996只用于模型选择、早停及六步微调；创建模型锁以后才做1997测试。1997结果不能反馈调参。已知未来ERA5场意味着这是历史回算，不是业务实时预报。
- 下述命令均为PowerShell单行命令，不使用CMD的 `^` 或易断开的续行反引号。已经生成的结果目录不覆盖；中断后先检查文件再决定如何恢复。

## 1. 在实验室定位文件并审计

在PowerShell中运行：

```powershell
conda activate jjq
Set-Location -LiteralPath 'H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast'
git branch --show-current
if ((git branch --show-current).Trim() -ne 'codex/china-multistation') { throw '请先切换到 codex/china-multistation，勿覆盖其他分支' }
git status --short
git fetch origin codex/china-multistation
git merge --ff-only origin/codex/china-multistation
python -m pytest tests/test_new_station_workflow.py tests/test_multistation.py -q
python -m src.xiamen_forecast.audit_station_sources discover --gesla-dir 'F:\GESLA\GESLA3' --era5-dir 'F:\ERA5-NEW\Shanwei'
```

先在输出中核对汕尾GESLA文件和三份ERA5文件对应U10、V10、MSL，核实GESLA元数据中的站点经纬度。若ERA5实际目录不是 `F:\ERA5-NEW\Shanwei`，改用真实目录。以下交互输入需粘贴**完整实际路径与核实后的坐标**：

```powershell
$gesla = Read-Host '汕尾GESLA完整文件路径'
$u10 = Read-Host '汕尾U10 NetCDF完整路径'
$v10 = Read-Host '汕尾V10 NetCDF完整路径'
$msl = Read-Host '汕尾MSL NetCDF完整路径'
$lat = [double](Read-Host 'GESLA核实的纬度，十进制度')
$lon = [double](Read-Host 'GESLA核实的经度，十进制度')
python -m src.xiamen_forecast.audit_station_sources configure --gesla-file $gesla --u10-file $u10 --v10-file $v10 --msl-file $msl --latitude $lat --longitude $lon
```

通过后生成：

- `outputs/source_audits/shanwei/source_audit.json`：原始实际覆盖期、质量标志、每年共同有效小时与选定训练起点。
- `outputs/station_configs/shanwei.json`：本机站点配置。位于Git忽略的 `outputs/`，不携带实验室绝对路径推送到Git。

若坐标、ERA5三变量时轴/网格/单位或1996/1997共同覆盖不满足要求，会明确报错并且**不会生成通过的站点配置**。此时不要运行训练，也不要改阈值“凑通过”；先核对文件和单位。

## 2. 准备与审计

```powershell
python -m src.xiamen_forecast.run_station_workflow prepare --station shanwei
```

逐年读取ERA5，训练期观测拟合UTide，生成 `outputs/processed/shanwei/aligned_dataset/{atmosphere,surge,time}.npy`、数据元信息、GESLA QC及 `outputs/processed/shanwei/prepared_dataset_audit.json`。此步可能较慢但不训练神经网络。**仅在该审计为 `passed` 时继续**。此入口拒绝覆盖已有汕尾处理目录。

## 3. 1996核心模型验证与锁定

```powershell
python -m src.xiamen_forecast.run_station_workflow validation --station shanwei --seed 42 --device cuda --language zh
python -m src.xiamen_forecast.run_station_workflow lock --station shanwei --seed 42
```

第一条只跑Persistence/Ridge、CNN-GRU与CNN-GRU Rollout-6，保存模型到 `models/shanwei/`，完整验证输出在 `outputs/experiments/shanwei/`，精简包在仓库 `reports/experiment_results/shanwei_short_term_1996_seed42/`。第二条记录三个固定模型文件的SHA256，保存在 `models/shanwei/formal_seed42/validation_model_lock.json`。若验证结果不合理，**先停在这里**；不要查看1997后再重新选超参数。

## 4. 1997全年独立测试

```powershell
python -m src.xiamen_forecast.run_station_workflow test --station shanwei --seed 42 --device cuda --language zh
```

该入口检查模型锁，评估同一组已定模型；Ridge复用已保存权重，不重新拟合。输出 `reports/experiment_results/shanwei_short_term_1997_seed42/`，包括一步模型指标、72小时各提前量CSV、论文式模型对比图、CNN-GRU P99分析。不会直接把测试指标用于模型选择。

## 5. 1997台风候选与人工核验

```powershell
$repo = (Resolve-Path -LiteralPath '..\..\..').Path
$events = Join-Path $repo 'reports\experiment_results\shanwei_typhoon_events_1997_seed42'
python -m src.xiamen_forecast.run_station_workflow scan-events --station shanwei --seed 42 --track-csv 'outputs\best_tracks\ibtracs.WP.list.v04r01.csv' --events-dir $events
```

在 `$events\candidate_events.csv` 核对最佳路径、局地风压、增水过程和输入完整性。只有真正具备台风影响证据的案例才把 `review_decision` 改为 `yes`，填写中文 `review_note`；普通大增水或仅仅距站较近不能自动纳入。无合格案例时如实报告，不执行下一条。

```powershell
python -m src.xiamen_forecast.run_station_workflow run-events --station shanwei --seed 42 --device cuda --track-csv 'outputs\best_tracks\ibtracs.WP.list.v04r01.csv' --events-dir $events
python -m src.xiamen_forecast.run_station_workflow report --station shanwei --seed 42 --events-dir $events
```

每场合格案例在 `$events\<event_id>\` 保存三次固定起报的72小时CSV、24/48/72小时累计RMSE/MAE、峰高与峰时误差、最佳路径图、各次起报曲线和中文研究报告。最后一条生成 `reports/experiment_results/shanwei_core_review_1997_seed42/` 的核心模型汇总表、图和中文结论。如果没有合格台风，可单独运行 `report`，报告会说明事件部分待完成。不同起报曲线不能拼接成虚假的连续预报。

## 6. 上传范围

先人工检查结果，再只暂存对应的小型报告目录。不要使用 `git add .`，不要上传 `outputs/`、`models/`、原始GESLA/ERA5或大型NPY。汕尾代码更新应先从 `codex/china-multistation` 同步，确认Git工作区内已有本地改动再拉取，避免覆盖实验室文件。

## 训练期独立台风留出方案

候选短名单见 `reports/experiment_results/independent_typhoon_holdout_plan_v1/candidate_shortlist.csv`，仅是从现有训练期统计按路径距离、数据完整性与较大观测增水筛出的**待人工核验**对象；这些证据本身不足以证明台风归因。确认少量事件后，复制成审核表，将确认的行填写 `review_decision=yes` 和气象/增水证据 `review_note`，再运行：

```powershell
python -m src.xiamen_forecast.plan_typhoon_holdout --reviewed-csv 'H:\审核后的训练期台风清单.csv' --output-csv 'H:\留出方案\fixed_exclusion_windows.csv'
```

脚本会冻结每场完整事件、此前24小时历史和此后72小时预报缓冲，拒绝跨训练/验证边界及重叠事件；**它只生成隔离时间窗，不会训练**。真正的独立留出需另建实验ID与独立处理/模型目录：UTide拟合必须剔除时间窗内潮位；训练标准化、Ridge拟合和CNN-GRU/rollout6训练窗口也必须完全排除这些时次（不仅排除目标时刻）；1996仍用于早停和方案选择；留出事件只能在方案锁定后评价一次。现有三站权重曾见过训练期事件，不能直接拿来称作独立留出测试。连云港仅加入已经选择的WMSE→WMSE作为附加对照，其他损失不重训。本阶段尚未改造上述重训链路，故留出结果**尚未完成**。
