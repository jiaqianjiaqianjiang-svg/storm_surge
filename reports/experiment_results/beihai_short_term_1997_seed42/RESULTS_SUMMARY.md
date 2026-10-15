# Beihai Short-Term Forecast Results

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
| 1 | cnn | 3.374 | 2.470 | 0.983 | 0.172 |
| 2 | cnn_lstm | 3.407 | 2.484 | 0.983 | 0.156 |
| 3 | cnn_gru | 3.408 | 2.477 | 0.983 | 0.155 |
| 4 | transformer | 3.408 | 2.502 | 0.983 | 0.155 |
| 5 | surge_mlp | 3.660 | 2.638 | 0.980 | 0.026 |
| 6 | ridge | 3.708 | 2.678 | 0.980 | 0.000 |
| 7 | tcn | 4.049 | 2.974 | 0.976 | -0.192 |
| 8 | persistence | 6.137 | 4.782 | 0.945 | -1.740 |
| 9 | era5_cnn | 15.352 | 12.263 | 0.585 | -16.144 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8681 | 6.145 | 3.717 | 3.669 | 15.393 | 3.382 | 3.414 | 3.417 | 4.058 | 3.416 | 3.648 |
| 3 | 8681 | 15.124 | 8.219 | 8.300 | 15.393 | 7.012 | 7.180 | 7.174 | 11.118 | 7.195 | 6.807 |
| 6 | 8681 | 22.009 | 10.930 | 11.667 | 15.392 | 8.675 | 9.137 | 9.187 | 17.077 | 9.157 | 8.423 |
| 12 | 8681 | 24.410 | 11.785 | 13.766 | 15.391 | 8.849 | 9.482 | 9.771 | 19.342 | 9.590 | 8.839 |
| 24 | 8681 | 17.169 | 12.176 | 14.448 | 15.384 | 9.189 | 9.786 | 10.086 | 19.675 | 9.783 | 9.112 |
| 48 | 8681 | 22.132 | 15.554 | 17.617 | 15.388 | 12.685 | 13.614 | 14.511 | 19.962 | 14.256 | 12.485 |
| 72 | 8681 | 24.926 | 17.025 | 19.166 | 15.370 | 14.792 | 15.848 | 17.158 | 19.475 | 17.323 | 14.492 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 3.417 | 3.648 | -6.780 |
| 3 | 7.174 | 6.807 | 5.120 |
| 6 | 9.187 | 8.423 | 8.321 |
| 12 | 9.771 | 8.839 | 9.537 |
| 24 | 10.086 | 9.112 | 9.658 |
| 48 | 14.511 | 12.485 | 13.960 |
| 72 | 17.158 | 14.492 | 15.534 |

## Rollout Training

- Base recursive validation RMSE: 7.7839 cm
- Best recursive validation RMSE: 7.3611 cm
- Best epoch: 16
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
