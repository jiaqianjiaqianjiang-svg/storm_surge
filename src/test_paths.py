import os
import sys

print("Python:", sys.version)
print("Current working directory:", os.getcwd())

ERA20C_DIR = r"F:\ERA20C"
GESLA_DIR = r"F:\GESLA\GESLA3"

print("ERA20C exists:", os.path.exists(ERA20C_DIR))
print("GESLA exists:", os.path.exists(GESLA_DIR))
