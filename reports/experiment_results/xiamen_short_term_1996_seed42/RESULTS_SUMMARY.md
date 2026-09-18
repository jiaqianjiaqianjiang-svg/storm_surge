# Xiamen Short-Term Forecast Results

## Experiment

- Validation year: 1996
- Random seed: 42
- Input window: 24 hours
- Rolling evaluation: known future ERA5 forcing; historical hindcast, not an operational forecast
- Test-year observations were not loaded during rollout fine-tuning

## One-Step Validation Ranking

| rank | model | rmse_cm | mae_cm | pearson_r | skill_score_vs_ridge |
| --- | --- | --- | --- | --- | --- |
| 1 | cnn_gru | 3.449 | 2.683 | 0.990 | 0.439 |
| 2 | cnn_lstm | 3.460 | 2.691 | 0.990 | 0.435 |
| 3 | cnn | 3.808 | 2.958 | 0.987 | 0.316 |
| 4 | transformer | 4.000 | 3.100 | 0.986 | 0.245 |
| 5 | surge_mlp | 4.279 | 3.313 | 0.984 | 0.136 |
| 6 | ridge | 4.603 | 3.567 | 0.982 | 0.000 |
| 7 | tcn | 4.615 | 3.558 | 0.982 | -0.005 |
| 8 | persistence | 12.003 | 9.693 | 0.876 | -5.798 |
| 9 | era5_cnn | 14.575 | 11.573 | 0.797 | -9.025 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8713 | 12.009 | 4.612 | 4.286 | 14.591 | 3.815 | 3.465 | 3.454 | 4.623 | 4.005 | 3.408 |
| 3 | 8713 | 20.976 | 8.897 | 9.294 | 14.590 | 7.070 | 6.484 | 6.552 | 10.463 | 7.605 | 5.991 |
| 6 | 8713 | 24.878 | 8.991 | 10.572 | 14.590 | 7.282 | 6.841 | 6.934 | 12.009 | 7.853 | 6.317 |
| 12 | 8713 | 14.894 | 9.327 | 12.743 | 14.592 | 7.572 | 7.117 | 7.156 | 13.935 | 8.167 | 6.568 |
| 24 | 8713 | 18.977 | 10.536 | 16.921 | 14.594 | 8.726 | 7.949 | 8.068 | 19.296 | 9.230 | 7.166 |
| 48 | 8713 | 25.704 | 12.676 | 20.432 | 14.595 | 11.775 | 10.092 | 10.210 | 27.532 | 11.924 | 8.629 |
| 72 | 8713 | 28.518 | 14.146 | 22.378 | 14.591 | 13.937 | 11.806 | 11.992 | 35.462 | 14.056 | 9.846 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 3.454 | 3.408 | 1.339 |
| 3 | 6.552 | 5.991 | 8.556 |
| 6 | 6.934 | 6.317 | 8.898 |
| 12 | 7.156 | 6.568 | 8.223 |
| 24 | 8.068 | 7.166 | 11.182 |
| 48 | 10.210 | 8.629 | 15.482 |
| 72 | 11.992 | 9.846 | 17.896 |

## Rollout Training

- Base recursive validation RMSE: 6.1515 cm
- Best recursive validation RMSE: 5.6536 cm
- Best epoch: 3
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
