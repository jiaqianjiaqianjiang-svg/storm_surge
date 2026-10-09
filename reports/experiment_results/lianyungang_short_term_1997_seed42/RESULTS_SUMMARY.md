# Lianyungang Short-Term Forecast Results

## Experiment

- Evaluation year: 1997
- Dataset split: test
- Random seed: 42
- Input window: 24 hours
- Rolling evaluation: known future ERA5 forcing; historical hindcast, not an operational forecast
- Test-year observations were not loaded during rollout fine-tuning

## One-Step Test Ranking

| rank | model | rmse_cm | mae_cm | pearson_r | skill_score_vs_ridge |
| --- | --- | --- | --- | --- | --- |
| 1 | cnn_lstm | 5.488 | 3.992 | 0.979 | 0.201 |
| 2 | cnn_gru | 5.495 | 3.996 | 0.979 | 0.199 |
| 3 | cnn | 5.744 | 4.167 | 0.977 | 0.124 |
| 4 | tcn | 6.005 | 4.444 | 0.975 | 0.043 |
| 5 | surge_mlp | 6.037 | 4.348 | 0.974 | 0.033 |
| 6 | transformer | 6.071 | 4.415 | 0.975 | 0.022 |
| 7 | ridge | 6.138 | 4.426 | 0.973 | 0.000 |
| 8 | persistence | 9.482 | 7.311 | 0.936 | -1.387 |
| 9 | era5_cnn | 17.154 | 13.741 | 0.781 | -6.810 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8681 | 9.493 | 6.150 | 6.047 | 17.152 | 5.754 | 5.498 | 5.504 | 6.012 | 6.082 | 5.523 |
| 3 | 8681 | 21.750 | 11.985 | 12.560 | 17.150 | 10.931 | 10.604 | 10.711 | 12.593 | 13.455 | 10.134 |
| 6 | 8681 | 27.856 | 14.116 | 16.134 | 17.152 | 12.735 | 12.189 | 12.457 | 14.407 | 17.331 | 11.430 |
| 12 | 8681 | 20.870 | 14.447 | 17.357 | 17.154 | 13.119 | 12.286 | 12.747 | 15.121 | 18.977 | 11.519 |
| 24 | 8681 | 22.872 | 15.832 | 19.517 | 17.144 | 14.524 | 13.405 | 13.851 | 17.146 | 21.516 | 12.342 |
| 48 | 8681 | 27.074 | 17.877 | 21.559 | 17.160 | 18.666 | 16.502 | 17.400 | 20.403 | 24.942 | 14.471 |
| 72 | 8681 | 29.181 | 18.798 | 22.458 | 17.116 | 20.908 | 18.535 | 19.963 | 21.977 | 27.832 | 15.707 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.504 | 5.523 | -0.337 |
| 3 | 10.711 | 10.134 | 5.392 |
| 6 | 12.457 | 11.430 | 8.243 |
| 12 | 12.747 | 11.519 | 9.631 |
| 24 | 13.851 | 12.342 | 10.892 |
| 48 | 17.400 | 14.471 | 16.835 |
| 72 | 19.963 | 15.707 | 21.322 |

## Rollout Training

- Base recursive validation RMSE: 10.5140 cm
- Best recursive validation RMSE: 9.9145 cm
- Best epoch: 14
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
