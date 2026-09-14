"""快速检查实验室电脑上的真实数据路径是否存在。"""

from pathlib import Path

import sys


ERA20C_DIR = Path(r"F:\ERA20C")
ERA5_DIR = Path(r"F:\ERA5")
ERA5_ALL_DIR = Path(r"F:\ERA5-ALL")
ERA5_NEW_DIR = Path(r"F:\ERA5-NEW")
GESLA_DIR = Path(r"F:\GESLA\GESLA3")

print("Python:", sys.version)
print("Current working directory:", Path.cwd())
print("ERA20C exists:", ERA20C_DIR.exists())
print("ERA5 exists:", ERA5_DIR.exists())
print("ERA5-ALL exists:", ERA5_ALL_DIR.exists())
print("ERA5-NEW exists:", ERA5_NEW_DIR.exists())
print("GESLA exists:", GESLA_DIR.exists())
