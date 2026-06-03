"""项目集中配置。

本文件只保存路径、站点信息、网格大小等配置项，不读取真实数据。
在实验室远程电脑运行时，通常只需要检查这里的路径是否正确。
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SiteConfig:
    """单个验潮站的运行配置。"""

    key: str
    name: str
    lat: float
    lon: float
    start_year: int
    end_year: int
    filename: str | None = None
    search_terms: tuple[str, ...] = ()

    @property
    def file_path(self) -> Path:
        """返回 GESLA 文件路径。"""

        if self.filename:
            return GESLA_DIR / self.filename
        return resolve_gesla_site_file(self)

    @property
    def output_dir(self) -> Path:
        """返回该站点输出目录。"""

        return OUTPUT_ROOT / self.key

    @property
    def cache_dir(self) -> Path:
        """返回该站点 ERA 缓存目录。"""

        return CACHE_ROOT / self.key / "era20c_yearly"


SITES: dict[str, SiteConfig] = {
    "xiamen": SiteConfig(
        key="xiamen",
        name="Xiamen",
        lat=24.45,
        lon=118.067,
        start_year=1954,
        end_year=1997,
        filename="xiamen-376a-chn-uhslc",
        search_terms=("xiamen",),
    ),
    "kushiro": SiteConfig(
        key="kushiro",
        name="Kushiro",
        lat=42.98,
        lon=144.37,
        start_year=1900,
        end_year=2010,
        search_terms=("kushiro",),
    ),
    "kashiwazaki": SiteConfig(
        key="kashiwazaki",
        name="Kashiwazaki",
        lat=37.36,
        lon=138.55,
        start_year=1900,
        end_year=2010,
        search_terms=("kashiwazaki",),
    ),
    "lusi": SiteConfig(
        key="lusi",
        name="Lusi",
        lat=32.06,
        lon=121.60,
        start_year=1900,
        end_year=2010,
        search_terms=("lusi", "lvsi", "lyusi"),
    ),
    "geting": SiteConfig(
        key="geting",
        name="Geting",
        lat=5.53,
        lon=102.11,
        start_year=1900,
        end_year=2010,
        search_terms=("geting",),
    ),
    "legaspi": SiteConfig(
        key="legaspi",
        name="Legaspi",
        lat=13.15,
        lon=123.75,
        start_year=1900,
        end_year=2010,
        search_terms=("legaspi", "legazpi"),
    ),
}


def get_site_config(site_key: str) -> SiteConfig:
    """按命令行站点名返回配置。"""

    key = site_key.lower()
    if key not in SITES:
        raise KeyError(f"未知站点 {site_key}，可选: {', '.join(sorted(SITES))}")
    return SITES[key]


def resolve_gesla_site_file(site: SiteConfig) -> Path:
    """在 GESLA_DIR 中按关键词自动寻找站点文件。"""

    candidates: list[Path] = []
    for term in site.search_terms or (site.key,):
        candidates.extend(sorted(GESLA_DIR.glob(f"*{term}*")))
    candidates = [
        path
        for path in candidates
        if path.is_file()
        # macOS 解压到 Windows 时常会产生 ``._文件名`` 资源附属文件。
        # 这类文件不是 GESLA 潮位数据，里面通常没有数据行，必须忽略。
        and not path.name.startswith("._")
        and not path.name.startswith(".")
    ]
    candidates = sorted(set(candidates), key=lambda path: path.name.lower())
    if not candidates:
        raise FileNotFoundError(
            f"找不到 {site.name} 的 GESLA 文件。请在 src/config.py 的 SITES['{site.key}'] "
            f"中填写 filename。已自动忽略 ._ 开头的系统附属文件。"
            f"搜索目录: {GESLA_DIR}; 搜索词: {site.search_terms}"
        )
    if len(candidates) > 1:
        print(f"[CONFIG] {site.name} 匹配到多个 GESLA 文件，使用第一个: {candidates[0]}")
    return candidates[0]


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
TIME_TILE_ROWS = 4
TIME_TILE_COLS = 4
INPUT_CHANNELS = len(VARIABLE_ORDER)
MODEL_GRID_SIZE = GRID_SIZE * TIME_TILE_ROWS


# =========================
# 5. 输出目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
XIAMEN_OUTPUT_DIR = OUTPUT_ROOT / "xiamen"
CACHE_ROOT = PROJECT_ROOT / "cache"
ERA20C_CACHE_DIR = CACHE_ROOT / "xiamen" / "era20c_yearly"


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
