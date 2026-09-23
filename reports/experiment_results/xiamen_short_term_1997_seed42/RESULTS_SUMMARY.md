# Xiamen Short-Term Forecast Results

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
| 1 | cnn_lstm | 3.469 | 2.674 | 0.986 | 0.461 |
| 2 | cnn_gru | 3.480 | 2.692 | 0.986 | 0.457 |
| 3 | cnn | 3.921 | 3.036 | 0.982 | 0.311 |
| 4 | transformer | 4.059 | 3.133 | 0.980 | 0.262 |
| 5 | surge_mlp | 4.294 | 3.320 | 0.978 | 0.174 |
| 6 | tcn | 4.597 | 3.545 | 0.975 | 0.053 |
| 7 | ridge | 4.725 | 3.660 | 0.973 | 0.000 |
| 8 | persistence | 12.037 | 9.702 | 0.828 | -5.491 |
| 9 | era5_cnn | 14.040 | 11.059 | 0.735 | -7.831 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8681 | 12.044 | 4.715 | 4.298 | 14.054 | 3.924 | 3.473 | 3.484 | 4.597 | 4.062 | 3.461 |
| 3 | 8681 | 20.002 | 8.873 | 8.907 | 14.053 | 7.270 | 6.317 | 6.407 | 10.110 | 7.486 | 6.079 |
| 6 | 8681 | 23.857 | 8.981 | 10.144 | 14.054 | 7.427 | 6.693 | 6.789 | 11.451 | 7.708 | 6.362 |
| 12 | 8681 | 14.348 | 9.352 | 11.810 | 14.052 | 7.670 | 7.013 | 7.108 | 13.259 | 8.007 | 6.566 |
| 24 | 8681 | 18.060 | 10.527 | 15.439 | 14.053 | 8.718 | 7.872 | 7.932 | 17.929 | 9.300 | 7.106 |
| 48 | 8681 | 24.513 | 12.870 | 18.599 | 14.017 | 11.606 | 9.794 | 10.118 | 25.279 | 12.145 | 8.670 |
| 72 | 8681 | 26.869 | 14.510 | 20.661 | 13.977 | 13.723 | 11.564 | 12.139 | 32.069 | 14.325 | 10.003 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 3.484 | 3.461 | 0.657 |
| 3 | 6.407 | 6.079 | 5.124 |
| 6 | 6.789 | 6.362 | 6.288 |
| 12 | 7.108 | 6.566 | 7.632 |
| 24 | 7.932 | 7.106 | 10.409 |
| 48 | 10.118 | 8.670 | 14.314 |
| 72 | 12.139 | 10.003 | 17.596 |

## Rollout Training

- Base recursive validation RMSE: 6.1515 cm
- Best recursive validation RMSE: 5.6536 cm
- Best epoch: 3
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
