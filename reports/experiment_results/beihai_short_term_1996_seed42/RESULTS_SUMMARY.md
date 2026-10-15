# Beihai Short-Term Forecast Results

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
| 1 | cnn_lstm | 3.437 | 2.446 | 0.986 | 0.169 |
| 2 | cnn | 3.462 | 2.456 | 0.985 | 0.157 |
| 3 | cnn_gru | 3.464 | 2.447 | 0.985 | 0.156 |
| 4 | transformer | 3.518 | 2.501 | 0.985 | 0.129 |
| 5 | ridge | 3.770 | 2.641 | 0.982 | 0.000 |
| 6 | surge_mlp | 3.773 | 2.648 | 0.982 | -0.001 |
| 7 | tcn | 4.048 | 2.915 | 0.980 | -0.153 |
| 8 | persistence | 6.019 | 4.644 | 0.956 | -1.549 |
| 9 | era5_cnn | 14.705 | 11.567 | 0.688 | -14.213 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | surge_mlp | era5_cnn | cnn | cnn_lstm | cnn_gru | tcn | transformer | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8713 | 6.033 | 3.782 | 3.785 | 14.721 | 3.471 | 3.447 | 3.475 | 4.059 | 3.529 | 3.823 |
| 3 | 8713 | 14.724 | 8.721 | 8.989 | 14.724 | 7.439 | 7.482 | 7.537 | 11.314 | 7.676 | 7.232 |
| 6 | 8713 | 21.236 | 11.665 | 12.800 | 14.731 | 9.204 | 9.684 | 9.881 | 17.541 | 9.973 | 9.076 |
| 12 | 8713 | 24.114 | 12.790 | 15.411 | 14.733 | 9.384 | 10.183 | 10.698 | 20.357 | 10.568 | 9.622 |
| 24 | 8713 | 18.915 | 13.215 | 16.127 | 14.726 | 9.629 | 10.476 | 11.091 | 21.043 | 10.729 | 9.944 |
| 48 | 8713 | 24.240 | 16.430 | 19.632 | 14.726 | 13.080 | 14.259 | 15.617 | 21.352 | 14.957 | 13.201 |
| 72 | 8713 | 25.476 | 17.640 | 20.500 | 14.741 | 14.757 | 16.429 | 18.416 | 20.896 | 17.112 | 14.874 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 3.475 | 3.823 | -10.019 |
| 3 | 7.537 | 7.232 | 4.042 |
| 6 | 9.881 | 9.076 | 8.150 |
| 12 | 10.698 | 9.622 | 10.061 |
| 24 | 11.091 | 9.944 | 10.338 |
| 48 | 15.617 | 13.201 | 15.473 |
| 72 | 18.416 | 14.874 | 19.231 |

## Rollout Training

- Base recursive validation RMSE: 7.7839 cm
- Best recursive validation RMSE: 7.3611 cm
- Best epoch: 16
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
