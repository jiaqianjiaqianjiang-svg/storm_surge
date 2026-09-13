# 厦门复现代码版本说明

本目录保留两套厦门论文复现材料，整理时未合并或删除任何一套。

## git_baseline

这是整理前位于仓库根目录的版本，也是原来被 Git 正式跟踪的版本。

- `src/`：模块化 Python 代码。
- `notebooks/`：未执行的命令行调用演示。
- `README.md`：该版本原说明。
- `requirements.txt`：该版本依赖。

使用时先进入此目录：

```powershell
cd projects\xiamen_reconstruction\git_baseline
```

## later_working_materials

这是整理前的 `code_my/`，包含：

- `Data_preprocessing.ipynb` 和 `Model_training.ipynb`：实际执行过的厦门工作 notebook。
- `xiamen/01_preprocessing/`：后续模块化预处理代码。
- `xiamen/02_model_training/`：后续模块化训练代码。
- `xiamen/01_xiamen_preprocess_all_in_one.py`：两文件版本的预处理入口。
- `xiamen/02_xiamen_train_all_in_one.py`：两文件版本的训练入口。

使用后续整理版时进入：

```powershell
cd projects\xiamen_reconstruction\later_working_materials\xiamen
```

## 当前处理原则

两套版本的数据输入形状、切分和评估实现并不完全一致。正式决定采用哪套以前，两边都保留，避免覆盖或删除。
