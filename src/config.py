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
REPO_ROOT = PROJECT_ROOT
DATA_ROOT = REPO_ROOT / "data"
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


# =========================
# 7. 短时预报配置
# =========================

# 短时预报固定使用 ERA5 小时级数据；原论文复现部分仍使用上面的 ERA20C 配置。
FORECAST_DATA_SOURCE = "ERA5"
FORECAST_INPUT_STEPS = 24
FORECAST_HORIZON = 1
FORECAST_FREQUENCY = "hourly"

# 短时预报使用前 t 个时刻的 U10/V10/SLP 和前 t 个 storm surge 历史值。
# ERA5 文件中的海平面气压变量名通常是 msl，这里对外统一称为 slp，
# 读取时仍会自动识别 msl/slp/sp/var151。
VARIABLES = ["u10", "v10", "slp"]

# ERA5 目录在实验室电脑上可能有多个版本，代码会按顺序检查并打印实际使用路径。
ERA5_DIR = Path(r"F:\ERA5")
ERA5_ALL_DIR = Path(r"F:\ERA5-ALL")
ERA5_NEW_DIR = Path(r"F:\ERA5-NEW")

# 仓库内小样例路径，仅用于没有 F 盘数据时的流程验收。真实实验优先使用上面的 F 盘路径。
LOCAL_GESLA_DIR = DATA_ROOT / "xiamen_GESLA"
LOCAL_SITE_FILE = LOCAL_GESLA_DIR / "xiamen-376a-chn-uhslc"

FORECAST_OUTPUT_ROOT = OUTPUT_ROOT / "forecast_xiamen"
FORECAST_MODEL_ROOT = PROJECT_ROOT / "models" / "forecast_xiamen"
FORECAST_FIGURE_ROOT = PROJECT_ROOT / "figures" / "forecast_xiamen"

ERA_VARIABLE_CANDIDATES = {
    "u10": ("u10", "10u", "u", "var165"),
    "v10": ("v10", "10v", "v", "var166"),
    "slp": ("msl", "slp", "sp", "var151"),
}

ERA_FILE_HINTS = {
    "u10": ("u10", "10u", "165"),
    "v10": ("v10", "10v", "166"),
    "slp": ("slp", "msl", "151"),
}
