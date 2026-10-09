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
| 1 | cnn_gru | 5.495 | 3.996 | 0.979 | 0.199 |
| 2 | cnn | 5.744 | 4.167 | 0.977 | 0.124 |
| 3 | surge_mlp | 6.037 | 4.348 | 0.974 | 0.033 |
| 4 | ridge | 6.138 | 4.426 | 0.973 | 0.000 |
| 5 | persistence | 9.482 | 7.311 | 0.936 | -1.387 |
| 6 | era5_cnn | 17.154 | 13.741 | 0.781 | -6.810 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_gru | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8681 | 9.493 | 6.150 | 6.047 | 17.152 | 5.754 | 5.504 | 5.523 |
| 3 | 8681 | 21.750 | 11.985 | 12.560 | 17.150 | 10.931 | 10.711 | 10.134 |
| 6 | 8681 | 27.856 | 14.116 | 16.134 | 17.152 | 12.735 | 12.457 | 11.431 |
| 12 | 8681 | 20.870 | 14.447 | 17.357 | 17.154 | 13.119 | 12.747 | 11.519 |
| 24 | 8681 | 22.872 | 15.832 | 19.517 | 17.144 | 14.524 | 13.851 | 12.342 |
| 48 | 8681 | 27.074 | 17.877 | 21.559 | 17.160 | 18.666 | 17.400 | 14.471 |
| 72 | 8681 | 29.181 | 18.798 | 22.458 | 17.116 | 20.908 | 19.963 | 15.707 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.504 | 5.523 | -0.344 |
| 3 | 10.711 | 10.134 | 5.393 |
| 6 | 12.457 | 11.431 | 8.242 |
| 12 | 12.747 | 11.519 | 9.632 |
| 24 | 13.851 | 12.342 | 10.895 |
| 48 | 17.400 | 14.471 | 16.833 |
| 72 | 19.963 | 15.707 | 21.320 |

## Rollout Training

- Base recursive validation RMSE: 10.5139 cm
- Best recursive validation RMSE: 9.9145 cm
- Best epoch: 14
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
