"""快速检查实验室电脑上的真实数据路径是否存在。"""

from pathlib import Path

import sys


ERA20C_DIR = Path(r"F:\ERA20C")
GESLA_DIR = Path(r"F:\GESLA\GESLA3")

print("Python:", sys.version)
print("Current working directory:", Path.cwd())
print("ERA20C exists:", ERA20C_DIR.exists())
print("GESLA exists:", GESLA_DIR.exists())
