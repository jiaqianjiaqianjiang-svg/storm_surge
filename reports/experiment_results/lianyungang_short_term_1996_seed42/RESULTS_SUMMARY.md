# Lianyungang Short-Term Forecast Results

## Experiment

- Evaluation year: 1996
- Dataset split: validation
- Random seed: 42
- Input window: 24 hours
- Rolling evaluation: known future ERA5 forcing; historical hindcast, not an operational forecast
- Test-year observations were not loaded during rollout fine-tuning

## One-Step Validation Ranking

| rank | model | rmse_cm | mae_cm | pearson_r | skill_score_vs_ridge |
| --- | --- | --- | --- | --- | --- |
| 1 | cnn_lstm | 5.216 | 3.900 | 0.983 | 0.237 |
| 2 | cnn_gru | 5.273 | 3.910 | 0.983 | 0.220 |
| 3 | cnn | 5.476 | 4.073 | 0.982 | 0.159 |
| 4 | surge_mlp | 5.738 | 4.232 | 0.979 | 0.077 |
| 5 | tcn | 5.771 | 4.376 | 0.979 | 0.066 |
| 6 | transformer | 5.882 | 4.362 | 0.979 | 0.029 |
| 7 | ridge | 5.971 | 4.441 | 0.978 | 0.000 |
| 8 | persistence | 10.020 | 7.794 | 0.938 | -1.816 |
| 9 | era5_cnn | 17.546 | 14.138 | 0.788 | -7.635 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8713 | 10.036 | 5.980 | 5.748 | 17.577 | 5.486 | 5.222 | 5.283 | 5.783 | 5.891 | 5.371 |
| 3 | 8713 | 23.581 | 12.472 | 12.575 | 17.576 | 10.949 | 10.534 | 10.730 | 12.750 | 13.996 | 10.199 |
| 6 | 8713 | 30.451 | 14.852 | 16.328 | 17.575 | 12.763 | 12.226 | 12.533 | 14.578 | 18.661 | 11.643 |
| 12 | 8713 | 21.210 | 14.914 | 17.568 | 17.578 | 12.973 | 12.244 | 12.747 | 15.287 | 20.309 | 11.704 |
| 24 | 8713 | 23.367 | 16.262 | 20.164 | 17.568 | 14.348 | 13.335 | 13.837 | 17.079 | 22.936 | 12.727 |
| 48 | 8713 | 28.574 | 18.408 | 22.561 | 17.562 | 17.978 | 16.033 | 16.931 | 20.225 | 27.491 | 14.993 |
| 72 | 8713 | 30.946 | 19.296 | 23.698 | 17.556 | 20.107 | 18.151 | 19.309 | 21.711 | 30.558 | 16.301 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.283 | 5.371 | -1.661 |
| 3 | 10.730 | 10.199 | 4.951 |
| 6 | 12.533 | 11.643 | 7.100 |
| 12 | 12.747 | 11.704 | 8.178 |
| 24 | 13.837 | 12.727 | 8.021 |
| 48 | 16.931 | 14.993 | 11.447 |
| 72 | 19.309 | 16.301 | 15.581 |

## Rollout Training

- Base recursive validation RMSE: 10.5140 cm
- Best recursive validation RMSE: 9.9145 cm
- Best epoch: 14
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
