"""Xiamen short-term forecast project package."""

import os
import sys


# Conda NumPy/Matplotlib and PyTorch can load separate Intel OpenMP runtimes on
# Windows. Set this before importing either stack so plotting does not abort the
# whole process after a model module has initialized PyTorch.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
