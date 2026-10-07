# Station mechanism comparison

This module analyses frozen Xiamen 1997 and Prickly Bay 2018 results. It does
not train models or modify existing outputs.

From the repository root:

```bash
python -m analysis.station_mechanism_comparison.run_analysis
```

To complete the observed-series ACF and scale analysis on the laboratory PC:

```powershell
python -m analysis.station_mechanism_comparison.run_analysis `
  --xiamen-dataset "H:\path\to\xiamen\aligned_dataset" `
  --prickly-dataset "E:\path\to\prickly_bay\aligned_dataset"
```

Each dataset directory must contain `time.npy` and `surge.npy`. Missing raw
series or prediction-level files are reported and skipped rather than inferred
from aggregate metrics.

Raw station data do not need to be copied between computers. A Git-safe result
directory containing `station_acf_full.csv`, `station_surge_scale.csv`, and
`station_analysis_audit.json` can be merged on another computer:

```bash
python -m analysis.station_mechanism_comparison.run_analysis \
  --xiamen-derived-dir reports/experiment_results/station_mechanism_xiamen_1997 \
  --output-dir reports/experiment_results/station_mechanism_comparison_complete
```
