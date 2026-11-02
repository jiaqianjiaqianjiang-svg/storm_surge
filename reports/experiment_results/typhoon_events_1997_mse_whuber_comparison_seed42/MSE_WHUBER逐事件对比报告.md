# 1997年历史台风 MSE/WHuber 逐场对照

本结果使用未来已知的ERA5再分析气象强迫，属于历史回算，不是实时业务预报。1997年已用于既有全年评价，因此这里只是追加历史案例分析，不是全新盲测。

三次固定起报（t0-72 h、t0-48 h、t0-24 h）均独立递归72小时；起报之后不使用未来真实增水回填。表中每个数值是同一台风三个重叠起报的描述性平均，不能当作三场独立事件。

| 事件 | 站点 | 模型 | 时效(h) | 窗口RMSE(cm) | 精确时效RMSE(cm) | 峰高绝对误差(cm) | 峰时误差(h) |
|---|---|---|---:|---:|---:|---:|---:|
| ZITA | beihai | cnn_gru | 24 | 7.66 | 19.31 | 5.20 | +1.0 |
| ZITA | beihai | cnn_gru | 48 | 12.03 | 27.67 | 7.58 | +0.0 |
| ZITA | beihai | cnn_gru | 72 | 15.33 | 37.98 | 10.79 | +0.3 |
| ZITA | beihai | cnn_gru_rollout6 | 24 | 8.02 | 19.55 | 4.01 | +1.0 |
| ZITA | beihai | cnn_gru_rollout6 | 48 | 12.11 | 24.73 | 11.19 | +0.0 |
| ZITA | beihai | cnn_gru_rollout6 | 72 | 14.85 | 29.64 | 18.70 | +0.3 |
| ZITA | beihai | whuber_one_step | 24 | 9.21 | 18.24 | 4.67 | +0.7 |
| ZITA | beihai | whuber_one_step | 48 | 13.32 | 26.47 | 14.41 | +1.3 |
| ZITA | beihai | whuber_one_step | 72 | 16.40 | 35.33 | 25.11 | +3.0 |
| ZITA | beihai | whuber_rollout6 | 24 | 9.14 | 17.09 | 4.93 | +1.0 |
| ZITA | beihai | whuber_rollout6 | 48 | 13.36 | 27.92 | 12.19 | +0.3 |
| ZITA | beihai | whuber_rollout6 | 72 | 16.39 | 38.98 | 20.43 | +2.0 |
| VICTOR | lianyungang | cnn_gru | 24 | 10.49 | 4.73 | 12.17 | +0.3 |
| VICTOR | lianyungang | cnn_gru | 48 | 16.12 | 6.15 | 20.81 | -11.3 |
| VICTOR | lianyungang | cnn_gru | 72 | 19.82 | 7.84 | 28.23 | -30.7 |
| VICTOR | lianyungang | cnn_gru_rollout6 | 24 | 10.75 | 5.83 | 8.18 | +4.7 |
| VICTOR | lianyungang | cnn_gru_rollout6 | 48 | 14.65 | 7.62 | 15.01 | +1.0 |
| VICTOR | lianyungang | cnn_gru_rollout6 | 72 | 16.07 | 10.30 | 22.42 | -18.3 |
| VICTOR | lianyungang | whuber_one_step | 24 | 9.60 | 7.32 | 7.37 | +5.0 |
| VICTOR | lianyungang | whuber_one_step | 48 | 15.79 | 16.60 | 10.14 | +18.3 |
| VICTOR | lianyungang | whuber_one_step | 72 | 19.80 | 29.41 | 12.89 | +15.7 |
| VICTOR | lianyungang | whuber_rollout6 | 24 | 9.80 | 9.10 | 7.38 | +4.7 |
| VICTOR | lianyungang | whuber_rollout6 | 48 | 13.03 | 13.81 | 11.78 | +4.7 |
| VICTOR | lianyungang | whuber_rollout6 | 72 | 14.84 | 16.33 | 17.52 | -14.7 |
| WINNIE | lianyungang | cnn_gru | 24 | 19.71 | 26.80 | 22.79 | -7.3 |
| WINNIE | lianyungang | cnn_gru | 48 | 24.95 | 31.07 | 37.84 | -15.0 |
| WINNIE | lianyungang | cnn_gru | 72 | 26.20 | 37.37 | 40.60 | +1.3 |
| WINNIE | lianyungang | cnn_gru_rollout6 | 24 | 17.82 | 21.63 | 1.28 | -6.0 |
| WINNIE | lianyungang | cnn_gru_rollout6 | 48 | 22.18 | 23.39 | 9.25 | +2.0 |
| WINNIE | lianyungang | cnn_gru_rollout6 | 72 | 25.07 | 58.19 | 17.45 | +2.3 |
| WINNIE | lianyungang | whuber_one_step | 24 | 24.75 | 32.92 | 27.64 | -9.0 |
| WINNIE | lianyungang | whuber_one_step | 48 | 31.66 | 32.10 | 51.96 | -4.0 |
| WINNIE | lianyungang | whuber_one_step | 72 | 34.74 | 51.93 | 62.36 | +4.7 |
| WINNIE | lianyungang | whuber_rollout6 | 24 | 20.56 | 32.82 | 21.51 | +2.7 |
| WINNIE | lianyungang | whuber_rollout6 | 48 | 24.79 | 29.91 | 35.45 | +5.3 |
| WINNIE | lianyungang | whuber_rollout6 | 72 | 27.30 | 52.65 | 43.02 | +5.7 |
| AMBER | xiamen | cnn_gru | 24 | 7.90 | 8.11 | 3.95 | +6.0 |
| AMBER | xiamen | cnn_gru | 48 | 11.62 | 10.84 | 10.51 | +5.7 |
| AMBER | xiamen | cnn_gru | 72 | 13.09 | 9.57 | 14.74 | +2.0 |
| AMBER | xiamen | cnn_gru_rollout6 | 24 | 7.55 | 8.26 | 6.96 | +5.7 |
| AMBER | xiamen | cnn_gru_rollout6 | 48 | 10.59 | 13.79 | 12.66 | +6.7 |
| AMBER | xiamen | cnn_gru_rollout6 | 72 | 12.29 | 12.46 | 16.37 | +2.0 |
| AMBER | xiamen | whuber_one_step | 24 | 6.77 | 9.86 | 5.25 | +0.0 |
| AMBER | xiamen | whuber_one_step | 48 | 10.25 | 10.56 | 3.78 | -0.3 |
| AMBER | xiamen | whuber_one_step | 72 | 13.83 | 7.28 | 4.41 | +0.3 |
| AMBER | xiamen | whuber_rollout6 | 24 | 6.83 | 8.38 | 7.35 | +0.3 |
| AMBER | xiamen | whuber_rollout6 | 48 | 10.24 | 10.21 | 7.70 | +0.3 |
| AMBER | xiamen | whuber_rollout6 | 72 | 12.91 | 9.73 | 9.67 | +0.0 |

## 解释限制

- `mean_window_rmse_cm`是每次起报后前24/48/72小时累计RMSE的平均；`exact_lead_rmse_cm`是恰好该提前量处三个误差的描述性RMSE，两者不能混用。
- 峰值指标始终在同一起报、同一固定窗口内比较；不同起报曲线没有拼接。
- 本对照不用于重新选择损失函数或调整WHuber参数。台风归因继续沿用已审核清单中的人工说明。
