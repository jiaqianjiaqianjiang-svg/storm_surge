# Storm Surge 工作区索引

本仓库按“研究项目、参考资料、工具、报告、归档”分类。整理过程只移动和归类文件，没有删除原有内容。

## 主要项目

### 1. 厦门论文复现

位置：`projects/xiamen_reconstruction/`

- `git_baseline/`：原来位于仓库根目录、已经被 Git 跟踪的 Python 复现版本。
- `later_working_materials/`：原来的 `code_my/`，包括后续整理版代码和实际运行过的 notebook。

两个版本存在实质差异，目前均保留。选择最终版本前不要互相覆盖。

### 2. 厦门短时预测

位置：`projects/xiamen_short_term/short_term_forecast/`

包含小时风暴增水 Ridge 滚动基线、汇报图片、CSV 和阶段总结。从 `projects/xiamen_short_term/` 目录运行原 README 中的命令。

### 3. 加勒比短时预测

位置：`projects/caribbean_forecast/caribbean_short_term_forecast/`

当前主要站点为 Prickly Bay，包含数据预处理、模型训练、直接与滚动预测、ERA5 消融、物理特征实验、测试、模型和实验产物。从 `projects/caribbean_forecast/` 目录运行项目 README 中的命令。

注意：`outputs/` 中约 1.3 GB 的处理数据和实验结果、`models/` 中的权重均不随普通 Git 提交保存，应单独备份。

## 参考资料

位置：`references/original_paper_code/`

保存原论文代码压缩包、解压后的原始预处理 notebook 和模型训练 notebook。建议作为只读参考，不在这里继续开发。

## 工具

- `tools/realtime_crawlers/`：浙江潮位、台风、GFS 等实时数据采集工具。
- `tools/zhejiang_export/zhejiang_excel_export_20260812/`：浙江台风数据检查和 Excel 导出脚本。

## 报告

位置：`reports/project_documents/`

保存原来的 Word、PowerPoint 项目资料。加勒比项目自己的技术报告仍留在对应项目目录内，便于与实验结果互相对应。

## 归档

位置：`archive/`

暂时不放入任何待删除文件。后续确认厦门最终版本后，可由维护者自行将旧版本移入此处或手动删除。

## 仓库级文件

- `.gitignore`：统一忽略数据、模型、图片、缓存和运行产物。
- `.gitattributes`：Git 属性设置。
- `skills-lock.json`：本地工具技能锁定信息。
