# 厦门短时风暴潮预测试验

本目录集中保存 2026-06-24 整理的短时预测阶段成果。

## 文件说明

- `short_term_rolling_baseline.py`：小时风暴增水 Ridge 滚动预测基线。它只验证滑动窗口、递归预测和分时效评价，不是最终 CNN 模型。
- `make_report_figures.py`：生成短时 CNN 流程图、分潮过程图和年度增水概览图。
- `本周汇报_短时风暴潮预测.md`：任务理解、本周进展、初步结果、限制和下一步计划。
- `results/`：脚本生成的 CSV、JSON 和图片，已在 `.gitignore` 中忽略。
- `figures_for_report/`：适合本周汇报正文展示的三张图片。

## 运行方式

在项目根目录执行：

```powershell
python short_term_forecast\short_term_rolling_baseline.py
python short_term_forecast\make_report_figures.py
```

脚本默认读取：

```text
data/xiamen_GESLA/xiamen-376a-chn-uhslc
```

当前试验使用修正后的 UTide 分潮逻辑，从原始 GESLA 小时潮位重新计算风暴增水，不读取已删除的旧 `processed_xiamen_1985` 标签。

## 正式模型方向

最终路线仍是沿用论文的 U10、V10、SLP 三类 `40×40` 气象网格和 CNN，同时维护增水历史窗口，并把新预测增水回填用于下一次预测。Ridge 结果仅作为增水自回归流程基线和持续性模型对照。
