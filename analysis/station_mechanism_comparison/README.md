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
