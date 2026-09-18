# 风暴增水项目完整说明与结果汇总

更新日期：2026-09-18  
当前正式分支：`codex/xiamen-short-term-parity`  
GitHub：`git@github.com:jiaqianjiaqianjiang-svg/storm_surge.git`

## 1. 一分钟了解这个项目

这个仓库研究的是利用验潮站观测和 ERA5/ERA-20C 大气资料预测风暴增水。目前包含三条相互区分的工作线：

1. **厦门小时级短时预报**：当前最重要、最新的一条工作线。使用 1970—1997 年数据，已完成数据处理、基线、五种神经网络对比、CNN-GRU 多步递归微调和 72 小时历史滚动回算。
2. **加勒比 Prickly Bay 小时级短时预报**：使用 2011—2018 年数据，已完成一步预测、直接 24 小时预测、滚动预测和 ERA5 消融实验。
3. **厦门日最大增水论文复现**：较早的复现工作，使用 ERA-20C。代码保留用于追溯和参考，不应与现在的小时级短时预报混用。

当前厦门短时预报的主要结果是：

- 1996 验证年一步预测中，`CNN-GRU` 最好，RMSE 为 **3.449 cm**，MAE 为 **2.683 cm**，Pearson r 为 **0.990**。
- `CNN-LSTM` 与它非常接近，RMSE 为 **3.460 cm**。
- 对 CNN-GRU 加入 6 步递归损失微调后，模型在 1—72 小时各提前量上均取得最低 RMSE。
- 72 小时 RMSE 从普通 CNN-GRU 的 **11.992 cm** 降到 **9.846 cm**，降低约 **17.9%**。

这里的 72 小时实验使用未来时次的 ERA5 再分析真值，科学上应称为**已知未来大气强迫条件下的历史回算**，不能直接称为实时业务预报。

## 2. 四台电脑、代码和数据分别在哪里

本项目目前涉及四台电脑。它们承担的任务不同，后续安排实验前必须先确认任务应该在哪台电脑执行。

| 电脑 | 当前职责 | 主要项目 | 大型数据状态 |
|---|---|---|---|
| MacBook Air | 代码整理、文档、Git 和精简结果检查 | 全项目协调 | 不保存厦门大数据 |
| 个人 Windows 笔记本 | 加勒比数据处理与实验 | Prickly Bay 短时预报 | 保存加勒比数据和运行产物 |
| 实验室公共 Windows 电脑 | 厦门大型数据处理与 GPU 实验 | 厦门短时预报 | 当前保存厦门数据和完整产物 |
| 个人 Windows 台式机 | 未来计算与厦门数据备份候选 | 暂未正式启用 | 后续可能从公共电脑迁入厦门数据 |

### 2.1 Mac 上的代码仓库

```text
/Users/jjq/Documents/storm_surge/storm_surge_clean
```

Mac 主要用于整理代码、查看结果、编写文档和提交 GitHub，不保存厦门的大型原始数据。

### 2.2 个人 Windows 笔记本

这台电脑负责加勒比 Prickly Bay 项目。已知使用过的代码位置为：

```text
E:\AAAqian\code\storm_surge_clean
```

加勒比 ERA5 数据目录为：

```text
F:\data\ERA5-Caribbean\Regional_Union
```

它主要保存和运行：

- Prickly Bay 2011—2018 年验潮与 ERA5 数据；
- 加勒比预处理结果；
- 一步、直接24小时、滚动预测和 ERA5 消融实验；
- 加勒比 `outputs/`、`models/` 和模型权重。

加勒比和厦门的数据、处理结果及模型权重相互独立，不能因为盘符都叫 `F:` 就认为它们在同一块硬盘或同一台电脑上。

### 2.3 实验室公共 Windows 电脑

这台电脑当前负责厦门项目，因为厦门数据体积较大，并且该电脑配有 RTX 4090 D。代码仓库位置为：

```text
H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge
```

厦门短时预报的实际运行目录：

```text
H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast
```

实验室环境：

```text
Conda 环境：jjq
PyTorch：2.11.0+cu128
CUDA：12.8，可用
GPU：NVIDIA GeForce RTX 4090 D
```

厦门原始数据位置如下。

验潮数据：

```text
F:\GESLA\GESLA3\xiamen-376a-chn-uhslc
```

ERA5 小时数据：

```text
F:\ERA5-NEW\Xiamen\xiamen_10u_1970_1997.nc
F:\ERA5-NEW\Xiamen\xiamen_v10_1970_1997.nc
F:\ERA5-NEW\Xiamen\xiamen_slp_1970_1997.nc
```

这些原始数据目前只在实验室公共电脑的数据盘保存，不上传 GitHub。程序读取原文件并将处理结果写入项目自己的 `outputs/`，不会修改原始 NetCDF 或 GESLA 文件。

这台电脑主要执行：

- 厦门 1970—1997 年数据准备和审计；
- 厦门五种神经网络训练；
- CNN-GRU 多步递归微调；
- 1—72 小时滚动诊断；
- 保存厦门完整 `outputs/`、`models/` 和 checkpoint；
- 导出精简结果并上传 GitHub。

因为这是公共电脑，厦门原始数据、处理数据、模型权重和关键结果不能只保留这一份。

### 2.4 个人 Windows 台式机

这台电脑目前还没有正式承担项目任务，代码目录、Conda 环境、GPU 和数据目录尚未登记。后续可能把实验室公共电脑上的厦门数据和运行产物迁移过来。

建议未来把它规划为：

- 厦门大型数据的第二份本地备份；
- 厦门后续多随机种子、1997 独立测试等计算任务的运行机；
- 公共电脑不可用时的替代计算环境；
- 长期保存 `outputs/`、`models/` 和最终实验档案。

迁移前先确认台式机磁盘容量、GPU、CUDA 和 Conda 环境。大型数据应通过移动硬盘或局域网复制，并用文件数量、总大小和校验值确认完整性，不通过 GitHub 传输。

### 2.5 四台电脑的协作原则

```text
Mac
  负责代码、文档、GitHub和精简结果审核

个人Windows笔记本
  负责加勒比原始数据、模型和实验产物

实验室公共Windows电脑
  当前负责厦门原始数据、模型和GPU实验

个人Windows台式机
  未来承接厦门数据备份和后续计算

GitHub
  在四台电脑之间同步代码、文档和小型结果，不同步原始数据与大型权重
```

任何电脑开始工作前，都先执行 `git status`，确认没有未提交修改，再拉取当前正式分支。任何实验结束后，都应把精简指标和报告导出到 GitHub，同时把完整结果保存在对应的数据电脑上。

大型文件至少应有两份副本。当前最需要补强的是厦门资料：它们现在主要位于实验室公共电脑，后续应复制到个人 Windows 台式机或独立硬盘。

## 3. 仓库结构

```text
storm_surge_clean/
├── PROJECT_GUIDE_CN.md                    本文档
├── README.md                              仓库总索引
├── projects/
│   ├── xiamen_short_term/
│   │   └── short_term_forecast/           厦门小时级短时预报正式项目
│   ├── caribbean_forecast/
│   │   └── caribbean_short_term_forecast/ 加勒比短时预报项目
│   └── xiamen_reconstruction/             厦门日最大增水论文复现归档
├── reports/
│   ├── experiment_results/                可提交 Git 的精简实验结果
│   └── project_documents/                 Word、PPT 等项目材料
├── references/original_paper_code/        原论文代码，只读参考
├── tools/                                  实时采集和数据导出工具
└── archive/                                归档目录
```

日常继续厦门短时预报时，只需要进入：

```text
projects/xiamen_short_term/short_term_forecast/
```

不要在 `references/` 或 `xiamen_reconstruction/` 中继续开发小时级短时预报。

## 4. 厦门小时级短时预报

### 4.1 研究问题

厦门短时预报要回答的是：利用过去 24 小时的风暴增水历史和 ERA5 大气场，能否预测下一小时风暴增水；一步模型递归使用时，误差会怎样累积；在训练中加入多步递归误差后，能否提高 6—72 小时稳定性。

每个样本使用：

- 过去 24 小时的 U10、V10、海平面气压空间场；
- 过去 24 小时的风暴增水；
- 目标为下一小时风暴增水。

所有风暴增水结果统一用 cm 报告。

### 4.2 年份划分与防止数据泄漏

| 数据用途 | 年份 |
|---|---|
| 训练集 | 1970—1995 |
| 验证集 | 1996 |
| 独立测试集 | 1997 |

关键原则：

1. 数据按年份和时间顺序划分，不随机打乱年份。
2. 标准化参数只由 1970—1995 训练数据拟合。
3. UTide 只使用截至 1995 年的验潮观测标定，再重构 1996—1997 潮汐。
4. 当前模型选择、递归微调和 72 小时比较都只使用 1996 验证年。
5. 1997 测试年应在方案最终锁定后只评价一次，不应用来反复挑模型。

### 4.3 数据处理流程

厦门正式数据流程为：

```text
GESLA 原始水位
  -> 读取、时间解析和质量控制
  -> UTide 分潮
  -> 观测水位减去天文潮位
  -> 风暴增水标签

ERA5 U10/V10/MSL
  -> 按时间读取
  -> 站点附近空间网格整理
  -> 与风暴增水按 UTC 整点对齐
  -> 按年份保存可按需读取的数据
  -> 生成过去24小时训练窗口
```

程序按年份和小块读取大型 ERA5 文件，并使用内存映射或按需索引，避免一次把 1970—1997 年全部空间场装入内存。数据准备主要受硬盘读取、NetCDF 解压、时间对齐和 UTide 计算限制，因此这一步以 CPU 和磁盘为主，GPU 不会明显加速。

### 4.4 已比较的模型

| 模型 | 输入 | 用途 |
|---|---|---|
| Persistence | 最后一个历史增水值 | 强自相关基线 |
| Ridge | 历史增水和 ERA5 统计信息 | 线性基线 |
| Surge-MLP | 只用历史增水 | 判断历史自相关贡献 |
| ERA5-CNN | 只用 ERA5 空间场 | 判断单独天气场能力 |
| CNN | ERA5 空间编码和历史增水 | 基础融合网络 |
| CNN-LSTM | ERA5 空间编码、增水历史和 LSTM | 正式时序模型 |
| CNN-GRU | ERA5 空间编码、增水历史和 GRU | 正式时序模型及当前主模型 |
| TCN | ERA5 编码和时序卷积 | 正式时序模型 |
| Transformer | ERA5 编码和注意力时序建模 | 正式时序模型 |
| CNN-GRU rollout-6 | CNN-GRU 加 6 步递归训练 | 降低多步误差累积 |

`Surge-MLP` 和 `ERA5-CNN` 是消融模型，不属于五个正式深度模型。五模型主体对比是 CNN、CNN-LSTM、CNN-GRU、TCN 和 Transformer。

### 4.5 一步预测结果

评价对象为 1996 验证年，共 8784 个样本：

| 排名 | 模型 | Pearson r | RMSE (cm) | MAE (cm) | Bias (cm) |
|---:|---|---:|---:|---:|---:|
| 1 | CNN-GRU | 0.9897 | **3.449** | **2.683** | -0.207 |
| 2 | CNN-LSTM | 0.9896 | 3.460 | 2.691 | 0.121 |
| 3 | CNN | 0.9875 | 3.808 | 2.958 | 0.181 |
| 4 | Transformer | 0.9862 | 4.000 | 3.100 | 0.306 |
| 5 | Surge-MLP | 0.9842 | 4.279 | 3.313 | -0.115 |
| 6 | Ridge | 0.9815 | 4.603 | 3.567 | -0.024 |
| 7 | TCN | 0.9815 | 4.615 | 3.558 | -0.111 |
| 8 | Persistence | 0.8756 | 12.003 | 9.693 | 0.002 |
| 9 | ERA5-CNN | 0.7969 | 14.575 | 11.573 | 0.518 |

这组结果说明：

1. 历史增水本身包含很强的信息，因为 Surge-MLP 和 Ridge 明显优于 ERA5-only。
2. ERA5-only 无法准确恢复当前海洋状态，但把 ERA5 与历史增水融合后，CNN、CNN-LSTM 和 CNN-GRU 均有明显提升。
3. CNN-GRU 是当前验证集最优模型，但只比 CNN-LSTM 低 0.010 cm，不能把两者描述成存在很大的性能差距。
4. TCN 在当前配置下没有超过 Ridge，说明模型复杂并不必然带来提升。

### 4.6 6 步递归微调

普通一步模型训练时只优化下一小时，递归 6、12 或 72 次以后会不断把自己的预测重新作为输入，因此会积累误差。

`CNN-GRU rollout-6` 从已有 CNN-GRU 权重继续训练，不是从头训练。训练时：

1. 先预计算并冻结耗时最大的 ERA5 空间编码器；
2. 主要更新 GRU 和回归头；
3. 对连续 6 步递归预测共同计算损失；
4. teacher forcing 在前 5 个 epoch 逐步下降到 0；
5. 递归验证始终不使用未来真实增水；
6. 如果微调没有优于原模型，会自动保留原始 checkpoint。

本次结果：

| 指标 | 数值 |
|---|---:|
| 原 CNN-GRU 6步递归验证 RMSE | 6.1515 cm |
| 微调后最佳 RMSE | **5.6536 cm** |
| 最佳 epoch | 3 |
| RMSE 改善 | 约 8.1% |
| 是否优于原模型 | 是 |

teacher forcing 只发生在训练阶段：它表示训练早期有一定概率把真实前一步值送回模型，帮助训练稳定；随后概率降为 0，让模型适应完全使用自身预测。验证和最终滚动比较不使用未来真实增水，因此不会因此产生评价泄漏。

### 4.7 1—72 小时滚动结果

滚动实验使用 1996 年共同的 8713 个起报时刻：

| 提前量 | Ridge | CNN | CNN-LSTM | CNN-GRU | Transformer | CNN-GRU rollout-6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 h | 4.612 | 3.815 | 3.465 | 3.454 | 4.005 | **3.408** |
| 3 h | 8.897 | 7.070 | 6.484 | 6.552 | 7.605 | **5.991** |
| 6 h | 8.991 | 7.282 | 6.841 | 6.934 | 7.853 | **6.317** |
| 12 h | 9.327 | 7.572 | 7.117 | 7.156 | 8.167 | **6.568** |
| 24 h | 10.536 | 8.726 | 7.949 | 8.068 | 9.230 | **7.166** |
| 48 h | 12.676 | 11.775 | 10.092 | 10.210 | 11.924 | **8.629** |
| 72 h | 14.146 | 13.937 | 11.806 | 11.992 | 14.056 | **9.846** |

CNN-GRU rollout-6 相对普通 CNN-GRU 的 RMSE 降幅：

| 提前量 | 降幅 |
|---:|---:|
| 1 h | 1.34% |
| 3 h | 8.56% |
| 6 h | 8.90% |
| 12 h | 8.22% |
| 24 h | 11.18% |
| 48 h | 15.48% |
| 72 h | 17.90% |

重要解释：多步微调的优势随提前量增加而扩大，说明它确实减轻了递归误差积累。但滚动过程中逐时使用了未来 ERA5 再分析场，因此结果用于模型诊断和历史回算，不是使用天气预报产品的真实业务预报。

### 4.8 厦门代码位置

以下路径均相对于：

```text
projects/xiamen_short_term/short_term_forecast/
```

| 文件 | 作用 |
|---|---|
| `src/config.py` | 项目路径、站点、变量和默认参数 |
| `src/xiamen_forecast/prepare_xiamen.py` | 厦门完整数据准备入口 |
| `src/xiamen_forecast/tide_gauge_loader.py` | 读取 GESLA 验潮数据 |
| `src/xiamen_forecast/tide_quality_control.py` | 验潮质量控制 |
| `src/xiamen_forecast/tide_processing.py` | UTide 分潮和风暴增水计算 |
| `src/xiamen_forecast/era5_loader.py` | ERA5 变量和网格读取 |
| `src/xiamen_forecast/time_alignment.py` | ERA5 与验潮时间对齐 |
| `src/xiamen_forecast/audit_prepared_dataset.py` | 训练前数据审计 |
| `src/xiamen_forecast/dataset_builder.py` | 24小时窗口、年份切分、训练集标准化 |
| `src/xiamen_forecast/forecast_model.py` | CNN、消融模型的基础结构 |
| `src/xiamen_forecast/temporal_models.py` | CNN-LSTM、CNN-GRU、TCN、Transformer |
| `src/xiamen_forecast/train_xiamen.py` | 单个模型训练入口 |
| `src/xiamen_forecast/train_model_suite.py` | 五个正式模型顺序训练入口 |
| `src/xiamen_forecast/evaluate_baselines.py` | Persistence 和 Ridge 基线 |
| `src/xiamen_forecast/evaluate.py` | 统一评价指标和强事件指标 |
| `src/xiamen_forecast/compare_models.py` | 一步模型统一汇总比较 |
| `src/xiamen_forecast/train_rollout_temporal.py` | 时序模型多步递归微调 |
| `src/xiamen_forecast/rolling_diagnostics.py` | 1—72小时统一滚动诊断 |
| `src/xiamen_forecast/export_final_results.py` | 导出可提交 Git 的精简结果包 |
| `src/short_term_forecast/journal_figures/` | 通用期刊绘图工具 |
| `tests/` | 数据、模型、滚动和绘图测试 |

### 4.9 厦门结果位置

实验室电脑上的完整运行产物通常位于：

```text
projects\xiamen_short_term\short_term_forecast\outputs\
projects\xiamen_short_term\short_term_forecast\models\
```

这些目录包括大型处理数据、逐时预测和模型权重，默认不会上传 GitHub。

已经上传 GitHub 的精简结果包位于：

```text
reports/experiment_results/xiamen_short_term_1996_seed42/
```

其中重要文件：

| 文件 | 内容 |
|---|---|
| `RESULTS_SUMMARY.md` | 厦门实验结果简表 |
| `validation_model_metrics.csv` | 一步模型统一指标 |
| `validation_model_comparison.png` | 一步模型四面板对比图 |
| `rolling_rmse_table.csv` | 各提前量 RMSE 宽表 |
| `rolling_metrics_long.csv` | 各模型、提前量和指标的长表 |
| `rolling_diagnostic_report.md` | 滚动实验说明与限制 |
| `rmse_vs_lead.png` | RMSE 随提前量变化图 |
| `strong_event_*.png` | 三个强增水事件曲线 |
| `rollout_training_metadata.json` | 递归微调元数据和最佳结果 |
| `rollout_loss_history.csv` | 微调训练历史 |
| `rollout_loss_curve.png` | 微调损失曲线 |
| `metrics_*.json` | 各神经网络详细一步指标 |

## 5. 加勒比 Prickly Bay 短时预报

### 5.1 数据和年份

| 项目 | 内容 |
|---|---|
| 站点 | Prickly Bay, Grenada |
| 坐标 | 12.005°N, 61.765°W |
| 验潮范围 | 2011-05-14 至 2018-10-04 |
| 训练年份 | 2011—2016 |
| 验证年份 | 2017 |
| 固定测试年份 | 2018 |
| ERA5 输入 | U10、V10、MSL，40×40 网格 |
| 输入历史 | 24小时 |

三个验潮传感器中，`rad` 通道质量评分最高，因此作为正式通道。ERA5 与验潮最终对齐得到 64,823 个小时，其中 59,330 条具有有效增水观测。

### 5.2 已完成实验

1. 验潮传感器质量控制和正式通道选择。
2. UTide 重新分潮并构建独立风暴增水标签。
3. 2011—2018 年 ERA5 合并、网格整理和逐小时对齐。
4. Persistence、Ridge、Surge-MLP、ERA5-CNN 和 Dual-CNN 一步预测。
5. Ridge、XGBoost、MLP、GRU 和 Dual-CNN 直接 24 小时预测。
6. 直接预测与递归滚动预测比较。
7. 历史增水、过去 ERA5 和未来 ERA5 的最小消融。
8. 1—72 小时滚动诊断和强事件指标。

### 5.3 主要结果

一步预测：

| 模型 | 2017验证 RMSE (cm) | 2018测试 RMSE (cm) |
|---|---:|---:|
| Persistence | 1.029 | 0.999 |
| Ridge | **0.880** | **0.816** |
| Surge-MLP | 0.926 | 0.876 |
| ERA5-CNN | 4.031 | 4.809 |
| Dual-CNN | 0.915 | 0.856 |

直接 24 小时预测的代表性结果：

| 模型 | 1 h | 6 h | 12 h | 24 h |
|---|---:|---:|---:|---:|
| Persistence | 1.031 | 2.864 | 2.902 | 2.718 |
| Combined-Ridge | **0.910** | **2.099** | 2.351 | 2.694 |
| XGBoost | 0.967 | 2.116 | **2.229** | **2.411** |
| Dual-CNN-Future | 1.142 | 2.155 | 2.408 | 2.553 |

加勒比当前可成立的主要结论：

1. 1 小时预测主要依靠历史增水自相关，Ridge 是最稳的一步模型。
2. 12—24 小时直接预测中，XGBoost 表现最好。
3. XGBoost 直接 24 小时 RMSE 为 2.411 cm，优于滚动 XGBoost 的 2.597 cm，说明长时间递归会积累误差。
4. ERA5 对 XGBoost 在 6—24 小时和 Top 5% 强增水样本上具有明确增量价值。
5. 只使用 ERA5 无法准确预测绝对增水，历史海洋状态仍是核心输入。
6. 当前 CNN 对 ERA5 空间场的利用不够稳定，尚不能声称稳定优于 Ridge 或 XGBoost。

详细技术报告位于：

```text
projects/caribbean_forecast/caribbean_short_term_forecast/PROJECT_PROGRESS_DETAILED_REPORT_20260812.md
```

### 5.4 加勒比代码位置

代码根目录：

```text
projects/caribbean_forecast/caribbean_short_term_forecast/
```

主要入口包括：

| 文件 | 作用 |
|---|---|
| `src/prepare_station.py` | 数据准备主入口 |
| `src/audit_prepared_dataset.py` | 数据审计 |
| `src/evaluate_baselines.py` | 一步基线 |
| `src/train_station.py` | 一步神经网络训练 |
| `src/direct_multistep_baselines.py` | 直接多步线性基线 |
| `src/train_direct_multimodel.py` | 直接24小时多模型训练 |
| `src/rolling_24_comparison.py` | 直接与滚动24小时比较 |
| `src/rolling_72_comparison.py` | 72小时滚动比较 |
| `src/era5_ablation.py` | ERA5消融 |
| `src/train_rollout_dual.py` | Dual-CNN多步递归训练 |
| `src/build_physics_features.py` | 物理特征构建 |
| `src/train_residual_physics_xgb.py` | 物理残差XGBoost |

## 6. 厦门与加勒比结果为什么差很多

不能直接用两边 RMSE 的绝对数值判断哪套代码更好，原因包括：

1. 两个站点的风暴增水幅度、潮汐环境和天气过程不同。
2. 厦门使用 1970—1997 年，Prickly Bay 使用 2011—2018 年。
3. 有效样本量、缺测比例、ERA5区域和观测质量不同。
4. 两边具体模型结构和实验发展阶段不同。
5. 厦门当前表格是 1996 验证结果；加勒比一步结果同时记录了 2017 验证和 2018 测试。

可以比较的是研究结论：两个站点都表明历史增水具有很强自相关；ERA5-only 都不够；大气信息需要与历史增水结合；递归训练或直接多步训练是控制长提前量误差的重要方向。

## 7. 厦门项目从头运行的正确顺序

先进入实验室电脑项目目录并激活环境：

```bat
conda activate jjq
cd /d "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge\projects\xiamen_short_term\short_term_forecast"
```

### 第一步：检查代码

```bat
python -m compileall -q src
python -m pytest tests -q
```

最近一次完整测试结果为：

```text
24 passed
```

### 第二步：准备和审计数据

只有在处理数据不存在或原始数据发生变化时才需要重跑：

```bat
python -m src.xiamen_forecast.prepare_xiamen --era5-dir "F:\ERA5-NEW\Xiamen" --gesla-file "F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
python -m src.xiamen_forecast.audit_prepared_dataset
```

### 第三步：基线和消融

```bat
python -m src.xiamen_forecast.evaluate_baselines
python -m src.xiamen_forecast.train_xiamen --model-type surge_mlp
python -m src.xiamen_forecast.train_xiamen --model-type era5_cnn
```

### 第四步：五个正式模型

```bat
python -m src.xiamen_forecast.train_model_suite --device cuda
python -m src.xiamen_forecast.compare_models --split validation
```

已经存在完整 `metrics.json` 的实验会自动跳过，因此命令中断后可以重新运行。

### 第五步：CNN-GRU 递归微调

```bat
python -m src.xiamen_forecast.train_rollout_temporal --model-type cnn_gru --device cuda --rollout-steps 6
```

### 第六步：72小时统一滚动诊断

```bat
python -m src.xiamen_forecast.rolling_diagnostics --device cuda --rollout-checkpoint "models\xiamen\formal_seed42\cnn_gru_rollout6\best_model.pth"
```

### 第七步：导出小型结果包

```bat
python -m src.xiamen_forecast.export_final_results
```

当前上述七步已经完成。除非数据、模型方案或随机种子改变，不需要重复训练。

## 8. GitHub 更新方式

实验室公共电脑更新代码：

```bat
cd /d "H:\02_代码与模型\蒋佳倩_2026-2029_软件工程硕士\storm_surge"
git checkout codex/xiamen-short-term-parity
git pull
```

个人 Windows 笔记本更新代码时，先进入该电脑上实际存在的仓库目录。历史位置是：

```bat
cd /d "E:\AAAqian\code\storm_surge_clean"
git checkout codex/xiamen-short-term-parity
git pull
```

Mac 更新代码：

```bash
cd /Users/jjq/Documents/storm_surge/storm_surge_clean
git checkout codex/xiamen-short-term-parity
git pull
```

个人 Windows 台式机第一次启用时，应从 GitHub 重新克隆当前正式分支，不要复制另一台电脑的整个 `.git` 目录：

```bat
git clone --branch codex/xiamen-short-term-parity git@github.com:jiaqianjiaqianjiang-svg/storm_surge.git storm_surge
```

台式机的最终代码路径、数据路径和 Conda 环境确定后，应补充到本文档。

大型原始数据、`outputs/`、`models/`、checkpoint 和大型逐时预测通常被 `.gitignore` 排除。GitHub 主要保存代码、文档和经过筛选的小型结果摘要，不能代替数据盘备份。

上传结果前先检查：

```bat
git status --short
```

不要使用 `git add .` 把不清楚的文件一次全部加入。优先明确指定代码或 `reports/experiment_results/` 中的精简结果目录。

## 9. 当前已经完成和还没有完成的工作

### 9.1 已完成

1. 厦门 1970—1997 验潮和 ERA5 数据准备流程。
2. 训练集限定 UTide、按年份切分和训练集标准化等防泄漏措施。
3. Persistence、Ridge 和两个消融模型。
4. CNN、CNN-LSTM、CNN-GRU、TCN、Transformer 五模型统一比较。
5. 1996 验证年一步结果汇总。
6. CNN-GRU 六步递归微调。
7. 1996 验证年 1—72 小时滚动比较。
8. 强事件诊断、结果图和精简结果归档。
9. 通用期刊绘图模块。
10. 加勒比 Prickly Bay 的完整数据、一步、直接24小时、滚动和消融流程。

### 9.2 尚未完成或尚未作为最终结论

1. **厦门 1997 独立测试年最终评价**：应先完全锁定模型和评价方案，再只执行一次。
2. **多个随机种子**：当前厦门正式结果主要是 seed 42，论文中最好补充多个 seed 的均值和标准差。
3. **统计显著性和时间块 Bootstrap**：逐小时样本高度相关，不能把每个小时当成完全独立样本。
4. **真实业务预报验证**：当前未来大气输入是 ERA5 再分析，不是实时数值天气预报。
5. **跨站点统一模型实验**：厦门和加勒比目前是两套独立实验，还没有训练一个跨站点模型。
6. **最终论文图表筛选**：绘图工具已具备，但最终投稿版式和图号尚未锁定。
7. **加勒比多随机种子和 2018 直接24小时最终封闭测试**：详细计划已写在加勒比进展报告中。

这些未完成项属于论文增强和最终验证，并不表示当前代码主流程不可运行。厦门核心流程已经跑通并得到可复核结果。

## 10. 汇报时可以直接使用的表述

可以直白地概括为：

> 本阶段完成了厦门站 1970—1997 年验潮与 ERA5 小时资料的质量控制、分潮、时间对齐和短时预报数据集构建。使用 1970—1995 年训练、1996 年验证，并保留 1997 年作为独立测试。统一比较了 Persistence、Ridge、消融模型以及 CNN、CNN-LSTM、CNN-GRU、TCN 和 Transformer。1996 年一步预测中 CNN-GRU 最优，RMSE 为 3.449 cm，CNN-LSTM 为 3.460 cm。随后对 CNN-GRU 进行 6 步递归损失微调，在 1—72 小时历史回算中均取得最低 RMSE，72 小时 RMSE 从 11.992 cm 降至 9.846 cm。结果说明历史增水是短期预测的核心信息，ERA5 与历史状态融合能够进一步提高厦门预测效果，多步训练可以明显减轻长提前量递归误差。

同时必须补充：

> 当前滚动实验使用已知未来 ERA5 再分析强迫，属于历史回算，不是实时业务预报。1997 独立测试、多随机种子和统计显著性检验仍是下一阶段工作。

## 11. 新接手者最短阅读顺序

第一次看到仓库时，按以下顺序阅读即可：

1. 本文档：了解整体目标、结果和路径。
2. `projects/xiamen_short_term/short_term_forecast/README.md`：查看厦门实际运行命令。
3. `reports/experiment_results/xiamen_short_term_1996_seed42/RESULTS_SUMMARY.md`：查看厦门精简结果。
4. `reports/experiment_results/xiamen_short_term_1996_seed42/rolling_diagnostic_report.md`：理解滚动实验边界。
5. `projects/caribbean_forecast/caribbean_short_term_forecast/PROJECT_PROGRESS_DETAILED_REPORT_20260812.md`：了解加勒比完整实验。
6. 最后再根据需要阅读 `src/` 中的具体实现。

如果只继续厦门短时预报，先不要阅读旧论文复现 notebook，也不要修改原始数据；从厦门项目 README、当前结果包和 `src/xiamen_forecast/` 开始即可。
