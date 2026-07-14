# Storm Surge Forecast

本项目用于厦门站风暴潮预测研究，包含两条代码线：

- `src/paper_reconstruction/`：原论文复现，任务是 `U10/V10/SLP -> daily maximum storm surge`。
- `src/short_term_forecast/`：当前主要工作，任务是 `前 t 小时 ERA5 + 前 t 小时 storm surge -> 下一小时 storm surge`，并通过 rolling forecast 扩展到多步预报。

仓库只提交代码、配置、README、测试和轻量说明文档。真实数据、预处理输出、模型权重、图片结果和临时代码包不提交到 GitHub。

## 数据路径

默认路径集中在 `src/config.py`。实验室 Windows 电脑通常使用：

```python
ERA20C_DIR = r"F:\ERA20C"
GESLA_DIR = r"F:\GESLA\GESLA3"
ERA5_DIR = r"F:\ERA5"
ERA5_ALL_DIR = r"F:\ERA5-ALL"
ERA5_NEW_DIR = r"F:\ERA5-NEW"
SITE_FILE = r"F:\GESLA\GESLA3\xiamen-376a-chn-uhslc"
```

运行前可先检查路径：

```bash
python src/test_paths.py
```

## 环境

建议在实验室远程电脑的 conda 环境中运行：

```bash
conda activate jjq
pip install -r requirements.txt
```

训练阶段需要 PyTorch。若已经单独安装 GPU 版 PyTorch，可以不用重复安装。

## 原论文复现

预处理：

```bash
python src/paper_reconstruction/preprocess_xiamen.py --start-year 1985 --end-year 1985
```

训练：

```bash
python src/paper_reconstruction/train_xiamen.py --epochs 2 --batch-size 16
```

完整年份可使用脚本参数 `--all-years`，具体参数见对应脚本帮助。

## 短时风暴潮预报

训练 24 小时输入窗口 CNN：

```bash
python src/short_term_forecast/train_forecast_xiamen.py --data-source ERA5 --frequency hourly --start-year 1985 --end-year 1985 --input-steps 24 --epochs 2 --batch-size 8 --model cnn
```

滚动预报：

```bash
python src/short_term_forecast/rolling_forecast.py --data-source ERA5 --frequency hourly --input-steps 24 --forecast-steps 12 --model cnn
```

常用分析：

```bash
python src/short_term_forecast/persistence_baseline.py --input-steps 24 --forecast-steps 12
python src/short_term_forecast/lead_time_error_analysis.py --forecast-csv <path_to_rolling_forecast.csv>
python src/short_term_forecast/peak_error_analysis.py --forecast-csv <path_to_rolling_forecast.csv> --percentile 95
python src/short_term_forecast/compare_models.py --root-dir outputs/short_term_forecast/xiamen
```

## 期刊级绘图工具

期刊绘图工具位于：

```text
src/short_term_forecast/journal_figures/
```

这些脚本不重新训练模型，只读取远程电脑上已有的 `metrics.json`、`val_predictions.csv`、`rolling_forecast.csv` 等结果文件。统一入口示例：

```bash
python -m src.short_term_forecast.journal_figures.make_all_journal_figures ^
  --results-root "E:\path\to\outputs" ^
  --output-dir "E:\path\to\journal_figures"
```

详细说明见 `JOURNAL_FIGURES_README.md`。

## 不提交到 GitHub

`.gitignore` 已排除：

```text
data/
outputs/
runs/
checkpoints/
models/
figures/
final_code_snapshot/
.mplconfig/
*.grb
*.grib
*.idx
*.nc
*.npy
*.npz
*.csv
*.jsonl
*.pth
*.pt
*.ckpt
*.h5
*.hdf5
*.zip
*.rar
*.7z
```

