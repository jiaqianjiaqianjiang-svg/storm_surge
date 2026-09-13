"""项目集中配置。

本文件只保存路径、站点信息、网格大小等配置项，不读取真实数据。
在实验室远程电脑运行时，通常只需要检查这里的路径是否正确。
"""

from pathlib import Path


# =========================
# 1. 真实数据路径
# =========================

# ERA-20C 根目录。目录下应包含 10U、10V、SLP 三个子目录。
ERA20C_DIR = Path(r"F:\ERA20C")

# GESLA-3 根目录。
GESLA_DIR = Path(r"F:\GESLA\GESLA3")


# =========================
# 2. 厦门站信息
# =========================

SITE_NAME = "Xiamen"
SITE_FILE = Path(r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc")
SITE_LAT = 24.45
SITE_LON = 118.067

# GESLA 厦门站大致可用年份。--all-years 会使用这个范围。
XIAMEN_START_YEAR = 1954
XIAMEN_END_YEAR = 1997


# =========================
# 3. ERA-20C 变量目录
# =========================

ERA20C_VARIABLE_DIRS = {
    "u10": ERA20C_DIR / "10U",
    "v10": ERA20C_DIR / "10V",
    "slp": ERA20C_DIR / "SLP",
}

# cfgrib/xarray 读入后，不同文件里的变量名可能略有不同。
# 这里按优先级列出候选名称。
ERA20C_VARIABLE_CANDIDATES = {
    "u10": ("u10", "10u", "u", "var165"),
    "v10": ("v10", "10v", "v", "var166"),
    "slp": ("msl", "slp", "sp", "var151"),
}


# =========================
# 4. CNN 输入参数
# =========================

# 站点周围 10°×10° 区域，即经纬度各向外扩展 5°。
REGION_HALF_SIZE_DEG = 5.0

# 插值后的空间网格。
GRID_SIZE = 40

# ERA-20C 3 小时一个时间片，一天 8 个，两天 16 个。
HOURS_PER_STEP = 3
STEPS_PER_DAY = 8
INPUT_DAYS = 2
STEPS_PER_SAMPLE = STEPS_PER_DAY * INPUT_DAYS

# 三个变量拼接后，单个样本通道数为 16 × 3 = 48。
VARIABLE_ORDER = ("u10", "v10", "slp")
INPUT_CHANNELS = STEPS_PER_SAMPLE * len(VARIABLE_ORDER)


# =========================
# 5. 输出目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
XIAMEN_OUTPUT_DIR = OUTPUT_ROOT / "xiamen"


# =========================
# 6. 清洗参数
# =========================

# GESLA 常见缺测标记。
MISSING_VALUE_MARKERS = {-99, -999, -9999, 9999, 99999}

# 宽松物理范围，单位沿用原始 GESLA 文件。这里只用于去除明显坏值，
# 不应删除真实极端风暴潮。
SEA_LEVEL_ABS_LIMIT = 10_000.0

# MAD 异常检测阈值。阈值较大，目标是只去掉明显离群坏点。
OBS_MAD_THRESHOLD = 15.0
SURGE_MAD_THRESHOLD = 15.0
