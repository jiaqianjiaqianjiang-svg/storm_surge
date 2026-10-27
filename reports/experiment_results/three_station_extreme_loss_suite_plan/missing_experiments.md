# 三站极值损失待核验与预计补充清单

本清单依据Git中的小型结果包生成。模型权重和完整预测文件被Git忽略，必须以实验室状态审计为准；“预计补充”不等于已经确认实验室缺失。

## 连云港

- Git已有MSE/MSE、WMSE/MSE、WMSE/WMSE、AWMSE/AWMSE、WHuber/WHuber的1996结果证据。
- 预计不需要重训。实验室先核对checkpoint、`training_config.json`、`training_metadata.json`和`extreme_loss_config.json`。
- 需要用统一入口重新生成含独立正向峰值指标的共同起报滚动表；这一步是推理和汇总，不是重训。

## 厦门

- Git已有MSE一步CNN-GRU与MSE/MSE Rollout-6结果。
- 若实验室没有现成对应权重，预计补充一步WMSE、AWMSE、WHuber，共3个一步模型。
- 预计补充WMSE/MSE、WMSE/WMSE、AWMSE/AWMSE、WHuber/WHuber，共4个Rollout-6模型。
- MSE/MSE优先复用，不因Mac看不到权重而重训。

## 北海

- Git已有MSE一步CNN-GRU与MSE/MSE Rollout-6结果。
- 若实验室没有现成对应权重，预计补充一步WMSE、AWMSE、WHuber，共3个一步模型。
- 预计补充WMSE/MSE、WMSE/WMSE、AWMSE/AWMSE、WHuber/WHuber，共4个Rollout-6模型。
- MSE/MSE优先复用，不因Mac看不到权重而重训。

## 三站共同待办

1. 运行实验室状态审计，区分`compatible`、`legacy_metadata`、`absent`、`incomplete`和`configuration_mismatch`。
2. 只补`absent`方案；旧完整结果经人工核对后可显式复用。
3. 逐站生成1996共同时间戳一步比较和共同起报72小时滚动比较。
4. 生成跨站CSV、PNG和中文报告，再根据1996规则逐站固定方案。
5. 固定方案前不运行新的1997评价，也不使用1997反向调整损失参数。
