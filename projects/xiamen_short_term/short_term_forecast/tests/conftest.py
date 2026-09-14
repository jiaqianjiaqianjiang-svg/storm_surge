"""Make direct ``pytest tests`` work with Windows conda launchers."""

import os
from pathlib import Path
import sys


# This must be set before pytest imports test modules containing NumPy,
# Matplotlib, and PyTorch. The same runtime guard lives in ``src/__init__.py``
# for normal ``python -m src...`` commands.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
