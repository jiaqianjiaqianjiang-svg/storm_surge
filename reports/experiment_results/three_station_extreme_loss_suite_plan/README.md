# 三站极值损失实验准备状态

本目录记录代码仓库在Mac上的可验证证据，不代表实验室被Git忽略的模型权重已经缺失。

## 已确认的小型结果

| 站点 | MSE/MSE | WMSE/MSE | WMSE/WMSE | AWMSE/AWMSE | WHuber/WHuber |
|---|---|---|---|---|---|
| 厦门 | 已有1996一步和Rollout精简结果 | 未见Git精简结果 | 未见Git精简结果 | 未见Git精简结果 | 未见Git精简结果 |
| 连云港 | 已有1996一步和Rollout精简结果 | 已有 | 已有 | 已有 | 已有 |
| 北海 | 已有1996一步和Rollout精简结果 | 未见Git精简结果 | 未见Git精简结果 | 未见Git精简结果 | 未见Git精简结果 |

`mac_repository_status.csv`逐方案记录Mac上checkpoint、配置、验证预测、滚动结果和Git精简证据。Mac不保存实验室大型文件，因此`laboratory_ignored_checkpoint_status`统一为“unknown until laboratory dry-run”。本目录是已完成阶段的归档记录；如需复核旧实验，应在实验室使用 `run_three_station_extreme_loss_suite --dry-run` 审计实际权重，不应仅凭Mac文件判断是否需要补训。后续新增站点按项目的 `CHINA_COASTAL_WORKFLOW.md` 执行，不再复制这套阶段性说明。

## 当前边界

- 代码入口和合成测试已准备；尚未在Mac执行真实训练。
- 1996统一三站结果包要等实验室补齐缺失权重并重新执行共同时间戳/共同起报评价后生成。
- 1997不参与本轮参数或损失函数选择。
- 独立台风留出重训暂缓；汕尾沿用现有核心流程。
