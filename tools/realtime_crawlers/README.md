# 实时风暴潮数据爬取

当前目标是给短时风暴潮预测准备实时输入。优先级如下：

1. 实时大气强迫：`U10、V10、SLP/PRMSL`
2. 台风过程信息：中心位置、中心气压、最大风速、风圈半径、预报路径
3. 潮位实况：观测水位、天文潮、残差水位
4. 外海水位/海流边界或对照产品

目前本文件夹已经实现：

- NOAA/NCEP GFS/NOMADS：下载 `U10、V10、PRMSL` 的 GRIB2 区域裁剪文件
- 中央气象台台风网：爬取台风路径、强度和机构预报路径
- 国家海洋环境预报中心 NMEFC：爬取台风风暴潮站点预报、天文潮、增水、总潮位

潮位站、水位、流量、雨量属于水文/海洋观测数据，需要继续确认公开接口或账号权限。

## 1. GFS/NOMADS 实时大气强迫

数据源：NOAA/NCEP NOMADS
入口：https://nomads.ncep.noaa.gov/

脚本：

```powershell
python real_time_data_crawler\gfs_nomads_downloader.py
```

默认下载最近可用 GFS cycle 的未来 0-24 小时，区域为：

```text
leftlon=110
rightlon=130
bottomlat=15
toplat=35
```

也可以指定巴威/浙闽台附近区域、时间和预报时效：

```powershell
python real_time_data_crawler\gfs_nomads_downloader.py --date 20260706 --cycle 06 --hours 0-24 --leftlon 110 --rightlon 130 --bottomlat 15 --toplat 35
```

输出到：

```text
real_time_data_crawler/output/gfs/
```

变量：

- `UGRD` at `10 m above ground`：10 m U 风
- `VGRD` at `10 m above ground`：10 m V 风
- `PRMSL` at `mean sea level`：海平面气压

这些 GRIB2 文件就是后续模型实时强迫场的来源。后面如果要接你复现论文的 CNN，需要再写一步 GRIB2 读取、区域插值/重采样到模型需要的网格。

## 2. 中央气象台台风路径

数据源：中央气象台台风网
页面：http://typhoon.nmc.cn/web.html

已定位到的接口：

- 台风列表：`http://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_default`
- 指定年份台风列表：`http://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_2026`
- 最新年份：`http://typhoon.nmc.cn/weatherservice/typhoon/jsons/maxYear`
- 指定台风路径：`http://typhoon.nmc.cn/weatherservice/typhoon/jsons/view_台风ID`

脚本：

```powershell
python real_time_data_crawler\nmc_typhoon_crawler.py --name 巴威
```

只抓当前活跃台风：

```powershell
python real_time_data_crawler\nmc_typhoon_crawler.py --active
```

### 可以爬到什么

这个网站主要提供台风路径和强度信息，适合作为风暴潮实时预测的气象/台风过程输入补充：

- 台风中心经纬度
- 中心气压
- 最大风速
- 移动方向、移动速度
- 风圈半径
- 中央气象台等机构的未来 12/24/36/48/72/96/120 小时预报路径

它不是验潮站网站，暂时不能直接提供潮位、水位、流量等水文站观测数据。

它不是验潮站网站，暂时不能直接提供潮位、水位、流量等水文站观测数据。

### 输出文件

默认输出到 `real_time_data_crawler/output/`：

- `typhoons_年份.csv`：该年份台风列表
- `raw_view_台风ID_名称.json`：官方接口原始数据
- `track_台风ID_名称.csv`：整理后的历史路径
- `forecast_台风ID_名称.csv`：整理后的机构预报路径

## 3. 潮位实况和水文数据

这部分公开接口没有 GFS/NOMADS 那么稳定。当前已补两个脚本：

- `nmefc_storm_surge_crawler.py`：国家海洋环境预报中心台风风暴潮站点预报爬虫
- `cwa_tide_crawler.py`：台湾 CWA 开放资料 API 下载器，需要 CWA 授权码和资料集编号
- `official_tide_source_search.py`：官方站点检索器，用关键词搜索潮位站/水位/风暴潮相关页面

### 3.1 国家海洋环境预报中心 NMEFC

页面：

```text
https://www.nmefc.cn/stormSurgeViews/typhoon
```

这个页面背后的接口在：

```text
https://www.nmefc.cn/api
```

已确认可爬的接口：

- `/data/init/typhoon`：当前台风风暴潮过程
- `/cms/dimsite/typhoon`：验潮站/潮位站点列表
- `/cms/dimsite/tree/typhoon`：按省份分组的站点树
- `/data/site/info`：单站最大增水、最高潮位、最低潮位、警戒级别、概率
- `/data/typhoon/statistics`：单站多情景的天文潮、增水、总潮位时间序列
- `/data/typhoon/path`：台风预报路径情景

运行当前台风过程：

```powershell
python real_time_data_crawler\nmefc_storm_surge_crawler.py
```

抓某一个站点，并下载时间序列：

```powershell
python real_time_data_crawler\nmefc_storm_surge_crawler.py --site 石头埠 --statistics
```

按省份筛选：

```powershell
python real_time_data_crawler\nmefc_storm_surge_crawler.py --province 福建 --all-sites
```

指定台风编号和路径情景：

```powershell
python real_time_data_crawler\nmefc_storm_surge_crawler.py --ty-code TY2610 --path-number 1 --site 石头埠 --statistics
```

输出到：

```text
real_time_data_crawler/output/nmefc/
```

主要输出：

- `nmefc_typhoon_init.json`：当前台风风暴潮过程信息
- `nmefc_typhoon_sites_all.csv`：全部站点列表
- `nmefc_typhoon_sites_selected.csv`：筛选后的站点
- `nmefc_typhoon_site_info_台风编号_p001.csv`：站点最大增水、最高潮位、最低潮位、警戒级别
- `nmefc_typhoon_timeseries_台风编号.csv`：天文潮、增水、总潮位时间序列
- `nmefc_typhoon_forecast_path_台风编号_p001.csv`：预报路径

注意：这里拿到的是 NMEFC 的风暴潮预报产品，不是原始验潮站实测数据。但它包含站点尺度的：

- `tide_cm`：天文潮
- `surge_cm`：增水
- `water_level_cm`：总潮位

这些数据可以用于官方产品对照、误差订正参考、站点输出展示，不能直接当“观测真值”。

### 3.2 台湾 CWA 潮位/海象资料

CWA 开放资料平台：

```text
https://opendata.cwa.gov.tw/
```

先在平台检索这些关键词：

```text
潮位站
潮高
浮標站與潮位站觀測資料
海象監測
```

拿到资料集编号和授权码后运行：

```powershell
python real_time_data_crawler\cwa_tide_crawler.py --dataset-id 资料集编号 --authorization 你的CWA授权码
```

也可以把授权码放到环境变量：

```powershell
$env:CWA_AUTHORIZATION="你的CWA授权码"
python real_time_data_crawler\cwa_tide_crawler.py --dataset-id 资料集编号
```

如果资料集支持按站点筛选，可以追加参数：

```powershell
python real_time_data_crawler\cwa_tide_crawler.py --dataset-id 资料集编号 --param StationID=站号
```

输出：

```text
real_time_data_crawler/output/cwa/
```

会保存：

- `cwa_资料集编号_raw.json`：原始 JSON
- `cwa_资料集编号_observations.csv`：一行一个站点-时刻的观测表
- `cwa_资料集编号_flattened.csv`：无法识别为海象观测结构时的兜底扁平化 CSV

当前已成功下载：

```text
real_time_data_crawler/output/cwa_o_b0075/cwa_O-B0075-001_raw.json
real_time_data_crawler/output/cwa_o_b0075/cwa_O-B0075-001_observations.csv
```

数据量：

```text
4165 条观测
85 个站号
```

关键字段：

```text
StationID
DateTime
TideHeight
TideLevel
StationPressure
WindSpeed
WindDirection
```

### 3.3 官方站点检索

- 台湾 CWA：适合查台湾周边潮位站，通常需要开放资料 API 授权码
- IOC/GLOSS：适合查全球海平面站，站点覆盖不一定满足浙江/福建近岸需求
- 地方水文/海洋/港口平台：可能有潮位、水位、流量、雨量，但接口公开性不稳定
- 国家海洋环境预报中心：适合做官方风暴潮预警/预报对照，不一定提供逐站高频实测

运行：

```powershell
python real_time_data_crawler\official_tide_source_search.py
```

默认会从这些官方入口开始找：

```text
https://opendata.cwa.gov.tw/
https://data.gov.tw/
https://www.nmefc.cn/
```

默认关键词：

```text
潮位、潮高、潮汐、水位、验潮、潮位站、浮标、海象、风暴潮、storm surge、tide、water level
```

输出：

```text
real_time_data_crawler/output/tide_source_matches.csv
```

也可以只查国家海洋预报台：

```powershell
python real_time_data_crawler\official_tide_source_search.py --start-url https://www.nmefc.cn/ --keyword 风暴潮 --keyword 潮位 --max-pages 100
```

## 5. 巴威附近实测潮位查找优先级

根据中央气象台巴威路径和 NMEFC 沿海站点表，已生成候选站：

```text
real_time_data_crawler/output/bavi_nearby_tide_station_candidates.csv
```

当前最接近巴威后期路径的中国沿海站点主要在闽东北到浙南：

```text
沙埕S、秦屿、三沙、鳌江S、瑞安S、温州S、坎门、海门Z、大陈、健跳、石浦
```

实测潮位查找顺序：

1. 温州市公共数据开放平台：优先查 `实时简报信息`、`潮位内容`、`瑞安市水利局`、`鳌江`、`瑞安`、`温州`
2. 台州市公共数据开放平台：优先查 `水雨情`、`潮位`、`海门`、`坎门`、`大陈`、`健跳`
3. 福建公共数据/水文水资源系统：优先查 `沙埕`、`秦屿`、`三沙`、`福鼎`、`宁德`、`水雨情`
4. 台湾 CWA 与 IOC/GLOSS：用于台湾附近或公开国际站点补充验证

注意区分：

- 实测潮位：可以用于实时校正，至少需要 `观测时间、站名/站码、实测潮位/水位、基准面`
- 天文潮汐表：不能当实测值，不含台风增减水
- NMEFC 风暴潮页面：是官方预报产品，适合作为对照，不是观测真值

NMEFC 巴威 `TY2609` 已另行保存：

```text
real_time_data_crawler/output/nmefc_bavi_ty2609_fujian/
real_time_data_crawler/output/nmefc_bavi_ty2609_zhejiang/
```

目前可用内容：

- `nmefc_typhoon_products_TY2609.csv`：巴威风暴潮格点产品文件列表
- `nmefc_typhoon_site_info_TY2609_p001.csv`：福建/浙江站点警戒与概率摘要

限制：

- `/data/typhoon/path?tyCode=TY2609` 返回为空
- `/data/typhoon/statistics?tyCode=TY2609` 对站点未返回完整天文潮/增水/总潮位序列
- 因此巴威的 NMEFC 可作为产品文件与站点摘要对照，暂不能像当前过程 `TY2610` 那样直接拿到完整站点时间序列

建议后续补的字段：

- `station_id`
- `station_name`
- `lon`
- `lat`
- `time`
- `observed_water_level`
- `astronomical_tide`
- `surge_residual = observed_water_level - astronomical_tide`
- `quality_flag`

## 4. 和短时风暴潮预测的关系

最直接的实时流程：

```text
1. GFS/NOMADS:
   下载 U10, V10, PRMSL

2. 中央气象台台风网:
   下载 center_lon, center_lat, Pc, Vmax, wind radii, forecast track

3. 模型输入:
   如果沿用复现论文，CNN 气象场仍然是 U10、V10、SLP/PRMSL

4. 短时滚动:
   预测增水后，把预测值回填到下一窗口的增水历史

5. 潮位站:
   有实况后做 bias correction 或验证
```
