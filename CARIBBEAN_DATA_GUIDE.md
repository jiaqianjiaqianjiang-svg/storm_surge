# 加勒比风暴潮推广：数据清单

## 1. 已下载的验潮站小时数据

来源：PSMSL/NOC Caribbean Tide Gauge Data Portal。

| 代码 | 站点 | 经纬度 | 推荐观测通道 | 实际记录时段 | 完整率 | 本地目录 |
|---|---|---:|---|---|---:|---|
| `calq` | Calliaqua, Saint Vincent and the Grenadines | 13.130, -61.196 | `calq_rad.csv` | 2013-12-12 至 2017-11-20 | 97.4% | `data/caribbean_tide_gauges/calq/` |
| `stlu` | Ganter's Bay, Saint Lucia | 14.017, -61.000 | `stlu_pr1.csv` | 2016-10-07 至 2018-12-31 | 98.1% | `data/caribbean_tide_gauges/stlu/` |
| `pric` | Prickly Bay, Grenada | 12.005, -61.765 | `pric_rad.csv` | 2011-05-14 至 2018-10-04 | 91.5% | `data/caribbean_tide_gauges/pric/` |

每站还带有 `*_tide_estimate.csv`。传感器与潮汐估计的基准面不一定相同，不能直接逐值相减；更稳妥的方式是延续当前项目流程，对选定的原始观测通道用 UTide 重新调和分析，再得到非潮汐残差。

这三个站的时段较短，适合作为跨区域迁移/独立验证集。若要从零训练单站模型，优先使用长记录站。

## 2. 推荐的长记录验潮站

优先从最新版 GESLA-4.1 元数据中筛选加勒比边界内记录；2026 年 7 月发布的完整包约 7.94 GB，解压约 69 GB，因此先下载元数据 CSV、筛站后再决定是否取完整包。

第一批建议站点：

| 站点 | 经纬度 | 推荐理由 |
|---|---:|---|
| San Juan, Puerto Rico (`NOAA 9755371`) | 18.459, -66.116 | 1970 年代起的长记录；飓风样本丰富；NOAA API 可继续更新 |
| Charlotte Amalie, USVI (`NOAA 9751639`) | 18.336, -64.920 | 1975 年建站；有长期水位与极值产品 |
| Magueyes Island, Puerto Rico (`NOAA 9759110`) | 17.970, -67.046 | 长记录、基准信息完整，适合长期对比 |
| Fort-de-France, Martinique | 14.600, -61.083 | 历史小时数据较长，可补充东加勒比法属岛屿 |
| Cristobal, Panama | 9.350, -79.917 | 20 世纪初开始的超长记录，适合检验长期泛化；位置属于加勒比侧 |

## 3. 气象强迫数据

统一使用 ERA5 hourly data on single levels：

- `10m_u_component_of_wind`，单位 m/s；
- `10m_v_component_of_wind`，单位 m/s；
- `mean_sea_level_pressure`，单位 Pa；
- 时间分辨率 1 小时，规则经纬网格 0.25°，1940 年至今。

下载脚本：`scripts/download_era5_caribbean.py`。脚本按站点、年份、变量拆分请求，并截取站点周围 10°×10°，与当前模型的空间输入定义一致。

准备 CDS 账号、接受 ERA5 数据条款并配置 API token 后，例如：

```bash
python scripts/download_era5_caribbean.py --station pric --start-year 2011 --end-year 2018 --output-root F:\ERA5-NEW
python scripts/download_era5_caribbean.py --station san_juan --start-year 1977 --end-year 2025 --output-root F:\ERA5-NEW
```

CDS API 客户端要求 `cdsapi>=0.7.7`。

## 4. 官方入口

- ERA5：https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- CDS API 配置：https://cds.climate.copernicus.eu/how-to-api
- GESLA-4.1 下载：https://gesla787883612.wordpress.com/downloads/
- GESLA 格式：https://gesla787883612.wordpress.com/format/
- 加勒比验潮站目录：https://psmsl.org/cme/catalogue.php
- 三站小时数据：https://psmsl.org/cme/downloaddata.php
- NOAA CO-OPS API：https://api.tidesandcurrents.noaa.gov/api/prod/

## 5. 接入当前代码前的必要改造

当前 `src/config.py`、`resolve_site_file()`、输出路径仍硬编码为厦门。开始训练加勒比站之前，需要把站名、经纬度、GESLA/CSV 路径和输出目录改成站点配置，而不是直接替换厦门常量。否则不同站点的缓存、模型和结果会互相覆盖。
