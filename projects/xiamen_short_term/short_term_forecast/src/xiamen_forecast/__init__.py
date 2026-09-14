"""Leakage-safe hourly storm-surge forecasting components for Xiamen.

Submodules are intentionally imported lazily so data-inspection commands do
not initialise PyTorch or other heavy numerical runtimes. Multi-year ERA5
windows are read on demand from prepared memory-mapped arrays.
"""

__all__: list[str] = []
