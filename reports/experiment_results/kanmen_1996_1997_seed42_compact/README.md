# Kanmen 1996-1997 Compact Results

Station: Kanmen, 28.088 N, 121.282 E
Training: 1975-1995
Validation: 1996
Historical test: 1997
Seed: 42

Models:
- CNN-GRU MSE
- CNN-GRU MSE Rollout-6
- CNN-GRU WHuber
- CNN-GRU WHuber Rollout-6

Contents:
- 1996/: rolling validation metrics and figures
- 1997/: rolling historical test metrics and figures
- typhoon_1997/: WINNIE event metrics and original figures
- model_locks/: checkpoint identity and SHA256 records
- data_audit/: prepared dataset audit

No ERA5, GESLA raw data, trained checkpoint weights,
or large prepared arrays are included.

Forecasts use known future ERA5 reanalysis forcing,
not operational meteorological forecasts.
