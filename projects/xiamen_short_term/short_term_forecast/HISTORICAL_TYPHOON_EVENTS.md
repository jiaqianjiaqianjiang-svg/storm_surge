# 1997年历史台风事件回算

本工具使用三个中国站已经训练的权重，不重新训练或改变模型。1996年为既有验证年；1997年全年结果已被查看，本轮属于**追加的历史案例分析**，不能写成新盲测。未来气象采用对应有效时次的ERA5再分析，属于**已知未来气象强迫的历史回算**，不是实时预报。

## 资料与筛选规则

- 最佳路径：[NOAA IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive)，使用[西北太平洋CSV](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv)。原始文件放在本项目的 `outputs/best_tracks/`，不会进入Git。运行时记录本地路径和SHA-256；CSV第二行是单位说明，读取时跳过。
- 使用1997年WP盆地的主路径点，不把spur当成另一条台风路径。相邻最佳路径点相隔不超过12小时才逐小时线性插值；记录真实最近路径时刻和为模型起报取整点的`t0`。500 km仅是候选初筛参考，不是台风影响边界。
- 对每个候选检查`t0−72h`到`t0+48h`共120小时的实测增水、ERA5、每次起报前24小时历史及未来72小时观测；还检查所有必需checkpoint。缺测或跨年不自动插值、不悄悄放行。
- `candidate_events.csv`初始全部为`pending`。需结合路径、风场/气压和同期验潮增水，人工在`review_note`写归因与完整性说明，并把`review_decision`改为`yes`或`no`。单凭距离或增水超过P99不能确定台风归因。没有适合事件的站点保留空白，并如实报告。
- 三次固定起报为`t0−72h`、`t0−48h`、`t0−24h`。每次分别连续预测72小时，模型只使用起报前真实增水，之后回填该模型自己的预测。各模型使用同一时刻、同一ERA5和同一观测。不能拼接不同起报曲线。

## 实验室PowerShell命令

在PowerShell中使用 `Set-Location`，不要使用CMD的 `cd /d`。以下整段可从仓库目录开始执行；下载约百MB，若实验室网络无法直连，可在浏览器从上述NOAA页面下载CSV后放到相同目录。

```powershell
conda activate jjq
Set-Location -LiteralPath "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast"
New-Item -ItemType Directory -Force "outputs\best_tracks" | Out-Null
Invoke-WebRequest -Uri "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv" -OutFile "outputs\best_tracks\ibtracs.WP.list.v04r01.csv"
python -m src.xiamen_forecast.run_typhoon_events scan --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv"
```

候选清单位于仓库根目录 `reports/experiment_results/typhoon_events_1997_seed42/candidate_events.csv`。请逐条审查路径、ERA5、验潮记录，填好`review_note`与`review_decision`；Excel编辑后须保存为CSV并保持表头、事件ID、站点和`t0`原值。模型文件若缺失，在`screening_note`中明确列出，先核实权重是否在实验室电脑且确实来自1996验证选定的方案。不要重新训练来补齐。首次`scan`不会覆盖已存在结果目录；若需要改筛选距离，另选一个新的`--results-dir`，保留旧结果。

确认至少一个真实案例后，从同一项目目录运行：

```powershell
python -m src.xiamen_forecast.run_typhoon_events run --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" --device cuda
```

Windows笔记本或无CUDA的机器把`--device cuda`改为`--device cpu`。运行时会核对原CSV的SHA-256及筛选协议，重新检查数据与权重；未标`yes`的候选不会进入正式事件计算。原始ERA5/GESLA、处理后的数组及现有模型权重只读。若某事件的数据不完整或已生成，不会覆盖该事件目录，会在`run_status.json`报告。

## 模型与输出

厦门、北海：Ridge、CNN-GRU、CNN-GRU rollout-6；连云港另含WMSE→MSE、WMSE→WMSE、AWMSE→AWMSE、WHuber→WHuber。Persistence也作为基础对照。所有模型都从各站既有`models/<station>/`加载；若任一必需权重缺失，事件跳过并明确报错，不能冒充已经完成该组合。

结果包为仓库根目录 `reports/experiment_results/typhoon_events_1997_seed42/`：

- `source_and_protocol.json`、`candidate_events.csv`、`confirmed_event_list.csv`：来源、筛选与人工确认记录。
- `run_status.json`、`all_event_metrics.csv`：总体完成状态与全部已完成事件的逐起报、逐模型、逐窗口评价。
- 每个`<station>_<stormSID>/`下有原始路径摘录、完整72小时预报CSV、`event_metrics.csv`、`exact_lead_metrics.csv`、权重来源和哈希、三张真实结果PNG和`研究报告.md`。

图1是真实最佳路径与站点位置（经纬度图，不是海岸线底图）；图2分三行显示三次起报的实测和预测过程，绝不拼接；图3显示精确24/48/72小时提前量RMSE，以及各固定窗口的峰高、峰时误差。`event_metrics.csv`的`rmse_cm`是单次起报后前H小时累计RMSE；`exact_lead_metrics.csv`的RMSE是三次固定起报在恰好H小时处的描述性汇总。三次重叠起报不是三场独立台风，也不能据此构造独立案例置信区间。

峰值始终在同一次起报后的**相同预先固定窗口**内比较；`t0−72h`起报的72小时窗口通常在`t0`前结束，可能不包含整场台风最大峰值。结果应逐事件报告低估幅度、峰时偏移、长时效误差和失败案例。若要核对整场峰值，还必须检查事件窗口与起报覆盖范围，不可直接套用单个窗口内峰值。

精简结果经检查后，仅提交此目录，例如从仓库根目录运行 `git add -- reports/experiment_results/typhoon_events_1997_seed42`，先看 `git diff --cached --stat`；不要把`outputs/`、`models/`或原始最佳路径整文件提交到GitHub。

## AMBER与VICTOR的下一轮核验（尚未确认归因）

原有候选清单显示：厦门AMBER的最近路径时刻为1997-08-29 07:00 UTC，120小时窗口峰值72.44 cm发生于此前7小时；厦门CASS的窗口与AMBER重叠99小时，**两份清单记录的是同一个观测峰值**，不能直接称为AMBER造成。连云港VICTOR最近路径时刻为1997-08-04 06:00 UTC，窗口峰值72.69 cm发生于其后4小时。两项均仍为`pending`。

实验室电脑先更新本分支，再在PowerShell运行只读核验。已有最佳路径文件必须与原筛选时使用的文件SHA-256一致；脚本不会修改原始数据或候选决定。

```powershell
conda activate jjq
Set-Location -LiteralPath "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git pull origin codex/china-multistation
Set-Location -LiteralPath "projects\xiamen_short_term\short_term_forecast"
python -m src.xiamen_forecast.review_typhoon_candidates --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv"
```

核验材料默认生成在仓库根目录`reports/experiment_results/typhoon_candidate_review_1997/`，每场包含逐小时增水/风矢量/气压CSV、最佳路径摘录、风场图、增水与气象过程图、去趋势ACF/频谱图。中央2×2与40×40风场统计仅是网格代理；需人工查阅区域风向、气压演变、最佳路径和原始潮位/潮汐分离质量。ACF或频谱中的周期性不能自行解释为残余潮汐。若归因不充分，继续保留`pending`；不要为了运行模型强行填`yes`。脚本拒绝覆盖已有核验目录，重新核验请指定新的`--output-dir`。

**只有归因与数据核查通过后**，再建立与旧结果隔离的新回算包。`scan`会重新生成候选清单且所有候选为`pending`；只对审查通过的AMBER和/或VICTOR手工填写`review_note`并将`review_decision`改成`yes`，其他候选保持`pending`。特别是AMBER，必须说明与CASS共享观测峰值时能够支持何种程度的归因；若无法区分，暂不纳入正式单台风评价。

```powershell
$caseDir = "..\..\..\reports\experiment_results\typhoon_events_1997_amber_victor_seed42"
python -m src.xiamen_forecast.run_typhoon_events scan --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" --stations xiamen lianyungang --results-dir $caseDir
# 此处先人工核验并编辑 $caseDir\candidate_events.csv；未确认时不要执行下一行。
python -m src.xiamen_forecast.run_typhoon_events run --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" --stations xiamen lianyungang --results-dir $caseDir --device cuda
```

同一事件的三次起报、全部模型、观测和未来ERA5强迫由现有回算程序统一处理，不重新训练。厦门主图比较CNN-GRU与原Rollout-6；连云港主图再加入WMSE→WMSE。Persistence与Ridge留在误差对照图和完整表，其他已训练损失组合保留为消融表格，不作为主图重点。输出包括每场`forecasts_72h.csv`、`event_metrics.csv`、`exact_lead_metrics.csv`和PNG；报告必须注明这是未来ERA5已知的历史回算、1997追加案例而非新盲测。

## 实时预报准备（尚未实现）

仓库已有`tools/realtime_crawlers/gfs_nomads_downloader.py`、`nmc_typhoon_crawler.py`、`nmefc_storm_surge_crawler.py`及若干潮位源检索脚本。GFS下载程序当前默认只取0–24小时，尚未证实能稳定提供完整72小时逐小时、同化并裁剪到与ERA5一致的40×40 U10/V10/MSL输入。须建立GFS循环发布时间、有效时间、缺时次检测和插值/重采样记录，并在训练期之外验证GFS与ERA5变量、单位、网格和标准化差异。未来GFS不能被ERA5真值悄悄替换。

实时运行还缺三个站可靠的最近24小时**实测**潮位、基准面和质量标记流，以及只用运行时可得资料分离非潮增水的程序。NMEFC风暴潮产品是官方**预报**，不能当验潮观测。将来应按发布时刻保存GFS循环号、当时可得的24小时历史、所有72小时预测和输入文件哈希；台风过后再用独立实测检验，才能称为真正提前预报。
