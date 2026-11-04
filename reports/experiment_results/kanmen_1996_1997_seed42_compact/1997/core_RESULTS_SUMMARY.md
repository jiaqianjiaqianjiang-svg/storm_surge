# Kanmen Short-Term Forecast Results

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
| 1 | cnn_gru | 5.027 | 3.807 | 0.958 | 0.046 |
| 2 | ridge | 5.145 | 3.888 | 0.956 | 0.000 |
| 3 | persistence | 6.429 | 5.001 | 0.932 | -0.561 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | cnn_gru | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- |
| 1 | 8681 | 6.422 | 5.135 | 5.020 | 4.970 |
| 3 | 8681 | 12.092 | 7.362 | 7.044 | 6.806 |
| 6 | 8681 | 14.894 | 7.964 | 7.520 | 7.138 |
| 12 | 8681 | 12.447 | 8.701 | 8.022 | 7.454 |
| 24 | 8681 | 15.715 | 9.765 | 9.027 | 8.075 |
| 48 | 8681 | 18.400 | 10.947 | 10.534 | 9.167 |
| 72 | 8681 | 20.214 | 11.851 | 11.600 | 9.936 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.020 | 4.970 | 0.989 |
| 3 | 7.044 | 6.806 | 3.378 |
| 6 | 7.520 | 7.138 | 5.081 |
| 12 | 8.022 | 7.454 | 7.088 |
| 24 | 9.027 | 8.075 | 10.542 |
| 48 | 10.534 | 9.167 | 12.980 |
| 72 | 11.600 | 9.936 | 14.347 |

## Rollout Training

- Base recursive validation RMSE: 6.7740 cm
- Best recursive validation RMSE: 6.6328 cm
- Best epoch: 9
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
