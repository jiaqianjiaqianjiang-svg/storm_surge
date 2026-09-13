# Runtime Data Layout

Use this layout from now on:

- Code repository: `E:\AAAqian\code\storm_surge_clean`
- Runtime data root: `E:\AAAqian\storm_surge_runtime_data`

Zhejiang water and rainfall data:

- Database: `E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\zhejiang_water_data.db`
- Logs: `E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\logs`
- Debug HTML/screenshots: `E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\debug`
- CSV exports: `E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\exports`
- Backups: `E:\AAAqian\storm_surge_runtime_data\zhejiang_water_data\backups`

Typhoon and forcing data:

- NMC typhoon tracks: `E:\AAAqian\storm_surge_runtime_data\typhoon_data\nmc`
- GFS files: `E:\AAAqian\storm_surge_runtime_data\typhoon_data\gfs`
- NMEFC products: `E:\AAAqian\storm_surge_runtime_data\typhoon_data\nmefc`
- CWA tide outputs: `E:\AAAqian\storm_surge_runtime_data\typhoon_data\cwa`

Normal run location:

```powershell
cd "E:\AAAqian\code\storm_surge_clean\real_time_data_crawler\water_level\爬虫_浙江全省"
E:\condaData\envs_dirs\mygpu\python.exe -m jupyter nbconvert --to notebook --execute "浙江全省.ipynb" --output "浙江全省_长期运行结果.ipynb" --ExecutePreprocessor.timeout=-1
```

Generated data should not be saved inside the Git repository. The code uses `runtime_paths.py` and Notebook path settings to write outputs under `storm_surge_runtime_data` by default.
