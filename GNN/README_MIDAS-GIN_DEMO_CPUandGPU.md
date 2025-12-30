MIDAS-GIN (CPU and GPU, Py39): Training and Permutation-Importance Demos
================================================================================

MIDAS  
2025-12-23

This project contains a CPU based workflow and an optional single GPU variant to **train a GIN** (Graph
Isomorphism Network) on MIDAS-style features. It also provides information to run
**permutation-importance analyses** on the graph or features on CPU only.

It targets **Python 3.9** with **PyTorch 2.8.0** and **PyTorch Geometric
2.6.1**, using **CPU wheels** in the CPU environments and **CUDA wheels**
in the GPU environment. PyTorch and PyG are installed *after* the conda
environment is created (see sections on CPU and GPU setup).

**Hardware Support:**
- **CPU:** Standard execution (Parallelised Cross-Validation).
- **GPU (CUDA):** Validated on the **UCL Myriad cluster** (Linux/NVIDIA). 
  **Requires CUDA 12.6**.

For pathway-rewiring permutations, it uses
**[xswap](https://github.com/hetio/xswap)**.

--------------------------------------------------------------------------------

# Paper

This code allows for the reproduction of the results published in:
    
https://www.researchsquare.com/article/rs-5499857/v1
    

The demo data frames provided here follow a similar structure to the data used in 
the paper.

See the paper's data availability.

--------------------------------------------------------------------------------

# Repository layout

- **Train CV GIN, log Optuna study, and save fold models:**
  - `scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py`  (supports `--device cpu` or `--device gpu`)
- **GLOBAL shuffle feature values to assess importance:**
  - `scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py`
- **Global (GiG) degree-preserving rewiring of all network edges:**
  - `scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py`
- **Pathway (GiG) degree-preserving rewiring within pathway subgraphs:**
  - `scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py`
- **Demo gene-pathway membership CSV:**
  - `data/demo/GeneSet_Demo_PathwaysNEW.csv`
- **Demo train/test data and graph:**
  - `data/demo/demo_MIDAS_InputMatrix_train.csv`
  - `data/demo/demo_MIDAS_InputMatrix_test.csv`
  - `data/demo/demo_MIDAS_InputMatrix_Graph.pkl`

- **Base conda envs (Python 3.9). Install torch/PyG after activation:**
  - `env/environment_midas-gin_demo_nmi_cpu_py39.yml`  CPU env for macOS and Linux (creates env `midas-gin-all` by default).
  - `env/environment_midas-gin_demo_nmi_cpu_py39_windows.yml`  CPU env for Windows with pinned versions for the demo.
  - `env/environment_midas-gin_demo_nmi_gpu_py39.yml`  optional GPU env for Linux or clusters. It mirrors the CPU env but is used with CUDA wheels for torch and PyG (see GPU setup below).

------------------------------------------------------------------------

# Requirements

- **OS:** CPU versions tested in macOS and Windows  
- **Python:** 3.9 (via conda)  
- **No GPU is required** for the demos. The main training script can optionally use a single GPU via `--device gpu`, but all permutation importance scripts run on CPU.


------------------------------------------------------------------------

# Create the conda environment

## CPU

> The YAML purposely **does not** install PyTorch or PyG. You will
> install them *after* activation to ensure correct wheels.

- **On Mac:**

``` bash
# From this project folder
conda env create -f env/environment_midas-gin_demo_nmi_cpu_py39.yml
conda activate midas-gin-cpu
```

- **On Windows:**

  - Note, in Windows, the following package versions should be used for
    the demo.
    - scikit-learn==1.2.2
    - optuna==3.0.3
    - sqlalchemy==1.4.44

These are specified in the
environment_midas-gin_demo_nmi_cpu_py39_windows.yml file.

``` bash
# From this project folder
conda env create -f env/environment_midas-gin_demo_nmi_cpu_py39_windows.yml
conda activate midas-gin-cpu
```

If you used a custom name in the YAML, activate that name instead.

# GPU

The creation of the GPU environment is similar:

``` bash
# From this project folder
conda env create -f env/environment_midas-gin_demo_nmi_gpu_py39.yml
conda activate midas-gin-gpu
```

If you used a custom name in the YAML, activate that name instead.

------------------------------------------------------------------------

# Install PyTorch and PyG wheels (AFTER activation)

# CPU only (recommended for generic testing)

``` bash
# PyTorch 2.8.0 CPU (macOS arm64 & Linux CPU wheels are available)

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric core + extensions built for torch 2.8.0 (CPU)

pip install torch-geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

> If you see build errors for the PyG extensions, re-check the ‘-f’ URL
> matches your **torch==2.8.0** and that you are in the **activated
> env**. pyg_lib is not installed. torch_sparse is used instead.

# GPU

# 1. Install PyTorch compatible with CUDA 12.6

``` bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

# 2. Install PyTorch Geometric

``` bash
pip install torch_geometric
```

# 3. Install PyG dependencies for CUDA 12.6 (ensure the torch version in the URL matches your installed torch version)

``` bash
pip install torch_scatter torch_sparse -f [https://data.pyg.org/whl/torch-2.8.0+cu126.html](https://data.pyg.org/whl/torch-2.5.0+cu126.html)
``` 


------------------------------------------------------------------------

# Install ‘xswap’ (for pathway rewiring during importance calculations)

‘xswap’ provides degree-preserving edge permutations used by the GiG and
Local scripts.

``` bash
pip install git+https://github.com/hetio/xswap.git
```

- **Troubleshooting ‘xswap’ build:**
  - Ensure you are on **Python 3.9** inside the activated environment.
  - Make sure the C/C++ toolchain is present.

------------------------------------------------------------------------

# Data expectations

## Train/Test CSVs

- **Index:** first column = **gene ID** (unique).
- **Columns:** numeric features plus a binary label column (default
  name: ‘targ_stat’, configurable via ‘–label_col’).

## Graph pickle (‘–graph_pkl’)

- A **pickled pandas DataFrame** edge-list with two columns representing
  endpoints (e.g., ‘source_name’, ‘target_name’, or aliases like
  ‘src’/‘dst’).
- The script enforces:
  - Node set matches the union of train+test CSV indices.
  - No dangling nodes (degree 0) or isolated cliques after filtering.

## Gene sets for GiG / Local (pathway rewiring)

- A CSV with at least:
  - ‘gene_expl’,‘gene’ and ‘PathwayID’

------------------------------------------------------------------------

# Workflow overview (Optuna DB & models)

## **Train + tune** with:

    scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py

This creates:

- An **Optuna study** stored in
  ‘output_dir/optuna_runs/<study_name>.optuna.db’ if ‘output_dir' is provided. Otherwise the script saves in the root directory of scripts under results/train/optuna_runs
- Per-fold best model weights in
  ’output_dir/models/Best_GINBiDir_train_foldID\_\*.pth’ if ‘output_dir' is provided. Otherwise it saves in the root directory of scripts under results/train/models
- Predictions with fold models (under results/train/predictions if ‘output_dir' is not provided) and log files (under results/train/logs if ‘output_dir' is not provided)


## **Permutation-importance (CPU only)** scripts **reuse** the same study DB and saved models. Run with:

    scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py

or

    scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py

or

    scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py

If you **do not** pass ‘–optuna_db’, they expect the **default** path
'above to exist'results/train/optuna_runs' where the ‘<study_name>.optuna.db’ file is to exist.


------------------------------------------------------------------------

# How to run

Adjust paths to your files. Use the same ‘–study_name’, ‘–output_dir’,
and ‘–label_col’ across all scripts. Change directory to where the scripts, env and data folders are.



## Train (CPU only) & save models (and Optuna study):

- **On Mac:**

``` bash
python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv  \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 --device cpu
```

- **On Windows:**

``` bash
python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py^
 --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
 --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
 --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
 --study_name MIDAS_GIN_demo^
 --n_trials 1 --epochs 2 --cv_splits 5 --cv_repeats 1 --device cpu
```

- **Outputs:**

  - ‘output_dir/optuna_runs/MIDAS_GIN_demo.optuna.db’  
  - ‘output_dir/models/Best_GINBiDir_train_foldID\_<k>GINBiDir_Optuna_MIDAS_demo.pth’  
  - Logs under ‘output_dir/logs/’
  - Predictions with fold models for training folds, test folds and
    held-out are saved in ‘output_dir/predictions’

## GPU Mode (Sequential Folds - Cluster Safe):

``` bash
python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --device gpu
```
------------------------------------------------------------------------

## Global feature shuffling (CPU only)

- **On Mac:**

``` bash
python scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --label_col targ_stat \
  --device cpu \
  --seeds 0:499 --cv_splits 5 --cv_repeats 1
```

- **On Windows:**

``` bash
python scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py^
 --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
 --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
 --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
 --study_name MIDAS_GIN_demo^
 --label_col targ_stat^
 --device cpu^
 --seeds 0:499 --cv_splits 5 --cv_repeats 1
```

- **Output**

  - CSV files written under:

    - <output_dir>/interpretability/permFeatImp/

  - Per fold, the script creates up to three files (names may vary
    slightly by implementation):

    - GINBiDir_KNN{KNN}\_foldID{fold}*CVTrainSet_permScores*{seedspan}.csv  
    - GINBiDir_KNN{KNN}\_foldID{fold}*CVTestSet_permScores*{seedspan}.csv  
    - GINBiDir_KNN{KNN}\_foldID{fold}*HeldOut_permScores*{seedspan}.csv

  - Each file is a tidy table containing (typical) columns such as:

    - CV_fold, permutation, bce_loss, perm_roc_auc and is intended for
      downstream aggregation/plotting across seeds and folds.

------------------------------------------------------------------------

## GiG Global (CPU only)

- **On Mac:**

``` bash
python scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --label_col targ_stat \
  --device cpu \
  --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

- **On Windows:**

``` bash
python scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py^
 --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
 --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
 --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
 --study_name MIDAS_GIN_demo^
 --label_col targ_stat^
 --device cpu^
 --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

- **Output:**

  - CSV files written under:
    <output_dir>/interpretability/permEdgeGlobalImp/

  - Per fold, the script creates up to three files (names may vary
    slightly by implementation):

    - GINBiDir_KNN{KNN}\_foldID{fold}*CVTrainSet_permEdgeScores*{seedspan}.csv  
    - GINBiDir_KNN{KNN}\_foldID{fold}*CVTestSet_permEdgeScores*{seedspan}.csv  
    - GINBiDir_KNN{KNN}\_foldID{fold}*HeldOut_permEdgeScores*{seedspan}.csv

  - Each file is a tidy table containing (typical) columns such as:

    - CV_fold, permutation, bce_loss, perm_roc_auc and is intended for
      downstream aggregation/plotting across seeds and folds.

------------------------------------------------------------------------

## GiG Local (pathway-subgraph rewiring, CPU only)

- **On Mac:**

``` bash
python scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --label_col targ_stat \
  --device cpu \
  --gene_set_csv data/demo/GeneSet_Demo_PathwaysNEW.csv \
  --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

- **On Windows:**

``` bash
python scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py^
 --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
 --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
 --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
 --study_name MIDAS_GIN_demo^
 --label_col targ_stat^
 --device cpu^
 --gene_set_csv data/demo/GeneSet_Demo_PathwaysNEW.csv^
 --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

- **Outputs:**

  - Per-gene CSV files in:
    - <output_dir>/interpretability/permEdgeLocalImp/
  - Each row corresponds to one permutation seed for one pathway for
    that GOI, with columns like:
    - gene, CV_fold, perm_type (“edge_only”), pathway_id, pathway_name
      (if present), permutation, orig_pred, perm_pred

------------------------------------------------------------------------

# Tips and troubleshooting

- **Optuna DB not found:** run the training script first (same
  ‘–study_name’ & ‘–output_dir’), or pass ‘–optuna_db’ explicitly.
- **Graph/feature node mismatch:** ensure graph endpoints match the
  union of train+test indices exactly.
- **Apple Silicon wheels:** stick to Python 3.9 and ‘torch==2.8.0’.
  Avoid mixing system/site Python.
- **xswap compile errors:** confirm toolchains (macOS CLT or Linux
  build-essential).
- **Optuna DB Locking**: If running on a GPU cluster, the script automatically
  increases the database timeout to 60 seconds to prevent sqlite3.OperationalError. 
  Avoid running multiple scripts writing to the same DB file simultaneously.

------------------------------------------------------------------------

# Citation / attribution

- GIN implementation relies on **PyTorch Geometric**.
- Pathway rewiring uses **‘xswap’** (Hetionet group).

