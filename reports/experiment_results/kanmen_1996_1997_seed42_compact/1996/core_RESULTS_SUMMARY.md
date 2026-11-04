# Kanmen Short-Term Forecast Results

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
| 1 | cnn_gru | 5.067 | 3.800 | 0.968 | 0.041 |
| 2 | ridge | 5.173 | 3.894 | 0.966 | 0.000 |
| 3 | persistence | 6.514 | 4.968 | 0.948 | -0.585 |

## Recursive RMSE

| lead_hours | valid_samples | persistence | ridge | cnn_gru | cnn_gru_rollout6 |
| --- | --- | --- | --- | --- | --- |
| 1 | 8713 | 6.511 | 5.171 | 5.064 | 5.046 |
| 3 | 8713 | 12.208 | 7.388 | 6.980 | 6.843 |
| 6 | 8713 | 14.914 | 7.912 | 7.323 | 7.135 |
| 12 | 8713 | 12.891 | 8.524 | 7.685 | 7.429 |
| 24 | 8713 | 17.041 | 9.684 | 8.608 | 8.146 |
| 48 | 8713 | 19.595 | 10.590 | 9.841 | 9.198 |
| 72 | 8713 | 21.605 | 11.349 | 10.705 | 9.824 |

## Rollout Improvement Over Base Model

| lead_hours | cnn_gru | cnn_gru_rollout6 | rmse_reduction_percent |
| --- | --- | --- | --- |
| 1 | 5.064 | 5.046 | 0.349 |
| 3 | 6.980 | 6.843 | 1.954 |
| 6 | 7.323 | 7.135 | 2.569 |
| 12 | 7.685 | 7.429 | 3.336 |
| 24 | 8.608 | 8.146 | 5.361 |
| 48 | 9.841 | 9.198 | 6.535 |
| 72 | 10.705 | 9.824 | 8.230 |

## Rollout Training

- Base recursive validation RMSE: 6.7740 cm
- Best recursive validation RMSE: 6.6328 cm
- Best epoch: 9
- Improved over base: True

Only compact metrics, reports, and figures are archived here. Raw data, prediction tables,
and model weights remain excluded from Git.
