# Xiamen journal figure guide

These figures use existing 1997 test and historical-hindcast results. No model was retrained.
The rolling experiment uses known future ERA5 reanalysis forcing and is not an operational forecast.

## Completed figures

- `fig01_one_step_model_comparison`
- `fig02_rmse_by_lead`
- `fig03_rollout_improvement`
- `fig04_extreme_and_rapid_rise_metrics`
- `fig05_rollout_training`
- `fig06_validation_scatter`
- `fig07_rollout_scatter_by_lead`
- `fig08_cnn_gru_residual_diagnostics`
- `fig09_cnn_gru_peak_analysis`
- `fig10_strong_event_1`
- `fig10_strong_event_2`
- `fig10_strong_event_3`

## Recommended presentation order

1. `fig01_one_step_model_comparison`: one-hour model selection.
2. `fig02_rmse_by_lead`: error growth across forecast lead times.
3. `fig03_rollout_improvement`: direct evidence that rollout training reduces accumulated error.
4. `fig04_extreme_and_rapid_rise_metrics`: performance for strong surges and rapid rises.
5. `fig06_validation_scatter` and `fig07_rollout_scatter_by_lead`: agreement and spread diagnostics.
6. `fig08_cnn_gru_residual_diagnostics` and `fig09_cnn_gru_peak_analysis`: detailed error diagnostics.
7. `fig10_strong_event_*`: representative strong-event hindcasts.
