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
| 1 | cnn_gru | 5.273 | 3.910 | 0.983 | 0.220 |
| 2 | cnn | 5.476 | 4.073 | 0.982 | 0.159 |
| 3 | surge_mlp | 5.738 | 4.232 | 0.979 | 0.077 |
| 4 | ridge | 5.971 | 4.441 | 0.978 | 0.000 |
| 5 | persistence | 10.020 | 7.794 | 0.938 | -1.816 |
| 6 | era5_cnn | 17.546 | 14.138 | 0.788 | -7.635 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_gru | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8713 | 10.036 | 5.980 | 5.748 | 17.577 | 5.486 | 5.283 | 5.371 |
| 3 | 8713 | 23.581 | 12.472 | 12.575 | 17.576 | 10.949 | 10.730 | 10.199 |
| 6 | 8713 | 30.451 | 14.852 | 16.328 | 17.575 | 12.763 | 12.533 | 11.643 |
| 12 | 8713 | 21.210 | 14.914 | 17.568 | 17.578 | 12.973 | 12.747 | 11.705 |
| 24 | 8713 | 23.367 | 16.262 | 20.164 | 17.568 | 14.348 | 13.837 | 12.729 |
| 48 | 8713 | 28.574 | 18.408 | 22.561 | 17.562 | 17.978 | 16.931 | 14.997 |
| 72 | 8713 | 30.946 | 19.296 | 23.698 | 17.556 | 20.107 | 19.309 | 16.306 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.283 | 5.371 | -1.665 |
| 3 | 10.730 | 10.199 | 4.955 |
| 6 | 12.533 | 11.643 | 7.103 |
| 12 | 12.747 | 11.705 | 8.177 |
| 24 | 13.837 | 12.729 | 8.006 |
| 48 | 16.931 | 14.997 | 11.423 |
| 72 | 19.309 | 16.306 | 15.555 |

## Rollout Training

- Base recursive validation RMSE: 10.5139 cm
- Best recursive validation RMSE: 9.9145 cm
- Best epoch: 14
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
