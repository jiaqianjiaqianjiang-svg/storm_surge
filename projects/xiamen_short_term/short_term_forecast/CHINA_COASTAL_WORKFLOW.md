# 中国沿海站点统一实验流程

本说明用于把已经验证的数据处理、模型训练和事件评价流程扩展到更多中国沿海验潮站。站点名单不是固定的；每个新增站都必须先依据实际GESLA与ERA5共同覆盖范围建立配置，再进入训练。

## 1. 当前实验边界

- 每站独立处理数据、计算UTide和标准化参数、训练模型并保存结果。
- 训练、验证、测试按时间严格隔离，具体年份由站点配置决定。
- 当前台风评价只从既定测试期中选取经人工核验的真实事件。
- 暂不开展训练期独立台风留出重训，也不把旧留出计划当成当前任务。
- 未来气象仍使用ERA5再分析，结果属于历史回算，不是实时业务预报。

## 2. 新增站点前检查

新增站点必须先确认：

1. GESLA站点身份、经纬度、基准面、质量标记和逐年有效观测。
2. U10、V10、MSL三个ERA5文件的时间、网格、坐标和单位一致。
3. GESLA与ERA5存在足够长的共同覆盖期，并能划分训练、验证和测试年份。
4. 24小时历史输入和72小时滚动评价具有足够完整窗口。
5. 新站输出目录与已有站点隔离，不覆盖处理数据、模型或结果。

只有源数据审计通过后，才把站点写入集中配置或经过审计的本机站点配置。不得凭文件夹名称猜测坐标、变量或年份。

## 3. 标准执行顺序

以下 `<station>` 代表已注册的站点ID。

### 3.1 准备与审计

```powershell
python -m src.xiamen_forecast.run_station_workflow prepare --station <station>
```

该入口依次完成数据准备和处理后审计。审计状态不是 `passed` 时停止，不进入训练。

### 3.2 验证与模型锁定

```powershell
python -m src.xiamen_forecast.run_station_workflow validation --station <station> --seed 42 --device cuda --language zh
python -m src.xiamen_forecast.run_station_workflow lock --station <station> --seed 42
```

验证阶段运行Persistence、Ridge、CNN-GRU和CNN-GRU Rollout-6。模型选择只使用配置中的验证期；锁定文件记录模型路径与SHA-256，防止测试前后权重被替换。

### 3.3 全年测试

```powershell
python -m src.xiamen_forecast.run_station_workflow test --station <station> --seed 42 --device cuda --language zh
```

测试阶段只评价已经锁定的模型，不重新训练或调参。输出包括一步指标、1—72小时滚动指标、强增水P99分析、模型对比图和Git安全的精简结果包。

### 3.4 测试期台风事件

```powershell
$events = "..\..\..\reports\experiment_results\<station>_typhoon_events_<test_year>_seed42"
python -m src.xiamen_forecast.run_station_workflow scan-events --station <station> --seed 42 --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" --events-dir $events
```

先人工核验候选路径、风场、气压和实测增水，只有 `review_decision=yes` 的事件才运行：

```powershell
python -m src.xiamen_forecast.run_station_workflow run-events --station <station> --seed 42 --device cuda --track-csv "outputs\best_tracks\ibtracs.WP.list.v04r01.csv" --events-dir $events
python -m src.xiamen_forecast.run_station_workflow report --station <station> --seed 42 --events-dir $events
```

每场事件采用多个固定起报，每次独立递归72小时，不拼接曲线，也不使用起报后的真实增水回填。

## 4. 结果目录

```text
outputs/processed/<station>/          # 处理后数组与审计，不上传
outputs/experiments/<station>/        # 完整实验输出，不整体上传
models/<station>/                     # 模型权重，不上传
reports/experiment_results/           # 小型CSV、图和报告，可审核后提交
```

每个结果包都应包含站点、时间划分、随机种子、模型来源和评价口径。目录已存在或配置不一致时不得静默覆盖。

## 5. 当前扩展状态

已有若干站点完成了全流程，可作为方法参考，但不构成最终站点名单。坎门已完成数据审计、1996验证、1997测试和测试期WINNIE事件回算；后续新增站点继续复用同一流程，不再复制独立项目。

## 6. Git与数据安全

- 原始GESLA、ERA5、NPY、完整逐时输出和模型权重不上传GitHub。
- 原始数据目录只读。
- 只提交经过检查的小型结果包、代码和说明文档。
- 不使用 `git add .`，显式指定提交路径。
- 不根据测试期结果返回修改训练参数或重新选择模型。
