# 测试期台风：双Rollout模型对比

仅使用归档预测；原图和旧结果未改动。未来气象为ERA5再分析，1997为追加历史案例分析。
阈值来源逐行记录；没有样本的P95/P99指标保持空白。不同起报重叠，不能作为独立台风样本。

## beihai_1997232N17115

- MSE Rollout-6：三次起报前72h窗口RMSE均值 14.85 cm；峰高绝对误差均值 18.70 cm。
- WHuber Rollout-6：三次起报前72h窗口RMSE均值 16.39 cm；峰高绝对误差均值 20.43 cm。
- P95 RMSE（有样本起报描述性均值）：MSE 19.85 cm，WHuber 17.35 cm；WHuber较低。
- P95低估比例（有样本起报描述性均值）：MSE 87.5%，WHuber 78.1%。
- P99 RMSE（有样本起报描述性均值）：MSE 21.07 cm，WHuber 18.05 cm；WHuber较低。
- P99低估比例（有样本起报描述性均值）：MSE 100.0%，WHuber 100.0%。

## kanmen_1997217N06168

- MSE Rollout-6：三次起报前72h窗口RMSE均值 20.93 cm；峰高绝对误差均值 29.76 cm。
- WHuber Rollout-6：三次起报前72h窗口RMSE均值 23.15 cm；峰高绝对误差均值 32.58 cm。
- P95 RMSE（有样本起报描述性均值）：MSE 23.03 cm，WHuber 26.11 cm；MSE较低。
- P95低估比例（有样本起报描述性均值）：MSE 83.3%，WHuber 79.6%。
- P99 RMSE（有样本起报描述性均值）：MSE 24.77 cm，WHuber 30.91 cm；MSE较低。
- P99低估比例（有样本起报描述性均值）：MSE 79.4%，WHuber 84.9%。

## lianyungang_1997210N15120

- MSE Rollout-6：三次起报前72h窗口RMSE均值 16.07 cm；峰高绝对误差均值 22.42 cm。
- WHuber Rollout-6：三次起报前72h窗口RMSE均值 14.84 cm；峰高绝对误差均值 17.52 cm。
- P95 RMSE（有样本起报描述性均值）：MSE 22.88 cm，WHuber 18.93 cm；WHuber较低。
- P95低估比例（有样本起报描述性均值）：MSE 88.6%，WHuber 85.6%。
- P99 RMSE（有样本起报描述性均值）：MSE 45.76 cm，WHuber 33.73 cm；WHuber较低。
- P99低估比例（有样本起报描述性均值）：MSE 100.0%，WHuber 100.0%。

## lianyungang_1997217N06168

- MSE Rollout-6：三次起报前72h窗口RMSE均值 25.07 cm；峰高绝对误差均值 17.45 cm。
- WHuber Rollout-6：三次起报前72h窗口RMSE均值 27.30 cm；峰高绝对误差均值 43.02 cm。
- P95 RMSE（有样本起报描述性均值）：MSE 21.69 cm，WHuber 27.76 cm；MSE较低。
- P95低估比例（有样本起报描述性均值）：MSE 43.4%，WHuber 74.6%。
- P99 RMSE（有样本起报描述性均值）：MSE 23.52 cm，WHuber 31.61 cm；MSE较低。
- P99低估比例（有样本起报描述性均值）：MSE 39.4%，WHuber 91.0%。

## xiamen_1997232N14136

- MSE Rollout-6：三次起报前72h窗口RMSE均值 12.29 cm；峰高绝对误差均值 16.37 cm。
- WHuber Rollout-6：三次起报前72h窗口RMSE均值 12.91 cm；峰高绝对误差均值 9.67 cm。
- P95 RMSE（有样本起报描述性均值）：MSE 16.62 cm，WHuber 14.53 cm；WHuber较低。
- P95低估比例（有样本起报描述性均值）：MSE 95.6%，WHuber 51.3%。
- P99 RMSE（有样本起报描述性均值）：MSE 20.94 cm，WHuber 16.45 cm；WHuber较低。
- P99低估比例（有样本起报描述性均值）：MSE 100.0%，WHuber 28.9%。

每场的峰时偏差和24/48/72小时累计RMSE见总表和事件报告。不同站点阈值来源可能不同，不对事件样本作独立显著性推断。
