MIDAS-GIN (CPU & GPU, Py39): Training and Permutation-Importance
================================================================

MIDAS  
2026-01-23

Graph neural network training and permutation-importance demos for MIDAS-style data.

This repository contains CPU and single-GPU workflows to:

- train a GIN (Graph Isomorphism Network) classifier on MIDAS-style gene-level data
- quantify feature and edge importance via permutation-based methods, including GiG and pathway‑local rewiring

The code is written for **Python 3.9** and has been tested with:

- **CPU environments** created via conda on macOS, Linux, and Windows
- **CPU environment** on the UCL CS cluster
- **GPU environment** on the UCL Myriad cluster (using Myriad’s CUDA / PyTorch modules)

PyTorch / PyG are always installed *after* creating the base environment (conda env or virtualenv).

For pathway‑rewiring permutations, the code uses
**[xswap](https://github.com/hetio/xswap)**.

---

## Paper

This code allows for the reproduction of the results published in:

> https://www.researchsquare.com/article/rs-5499857/v1

The demo data frames provided here follow a similar structure to the data used in 
the paper. See the paper’s data‑availability statement for full details.

---

## Repository layout (within the MIDAS project)

At the top level of the MIDAS project you have:

```text
MIDAS_Project/
├─ GNN/
│  ├─ README_GNN.md
│  ├─ env/
│  │  ├─ environment_midas-gin_demo_cpu_py39.yml
│  │  ├─ environment_midas-gin_demo_cpu_py39_windows.yml
│  │  └─ environment_midas-gin_demo_cpu_cs.yml
│  ├─ data/
│  │  └─ demo/
│  │      ├─ demo_MIDAS_InputMatrix_train.csv
│  │      ├─ demo_MIDAS_InputMatrix_test.csv
│  │      ├─ demo_MIDAS_InputMatrix_Graph.pkl
│  │      └─ GeneSet_Demo_PathwaysNEW.csv
│  ├─ scripts/
│  │  ├─ train/
│  │  │  └─ DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py
│  │  └─ interpretability/
│  │      ├─ GIN_PermImpGlobal_MIDAS_Demo_CPU.py
│  │      ├─ GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py
│  │      └─ GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py
│  └─ results/
│     ├─ train/
│     │   ├─ logs/
│     │   │   └─ config_*.json
│     │   ├─ optuna_runs/
│     │   │   └─ *.optuna.db
│     │   └─ models/
│     │       └─ *.pt
│.    │.  └─ predictions/
│     │       └─ *.csv
│     └─ interpretability/
│         ├─ permFeatImp/
│         ├─ permEdgeGlobalImp/
│         └─ permEdgeLocalImp/
├─ FeatureProcessing/
└─ <code and scripts for pre-processing>
└─ Figs/
   └─ <code and scripts to generate figures>
```

Everything in this README refers to paths *relative to* the `GNN/` directory.

---

## Environments overview

There are four main tested ways to run the GNN code:

1. **Local CPU (Linux / macOS)** via `env/environment_midas-gin_demo_cpu_py39.yml`
2. **Local CPU (Windows)** via `env/environment_midas-gin_demo_cpu_py39_windows.yml`
3. **CPU on the UCL CS cluster** via `env/environment_midas-gin_demo_cpu_cs.yml`
4. **GPU on the UCL Myriad cluster** using Myriad’s modules plus a small Python virtualenv


The sections below describe each environment in more detail.

---

## 1. Local CPU (Linux / macOS)

From the `GNN/` directory:

```bash
conda env create -f env/environment_midas-gin_demo_cpu_py39.yml
conda activate midas-gin-cpu
```

The CPU env pins (or is intended to pin) versions along the lines of:

- `python=3.9`
- `numpy=1.26`
- `pandas=2.3`
- `scipy=1.10`
- `scikit-learn=1.4`
- `imbalanced-learn=0.12`
- `optuna=4.4`

This combination matches the versions used for the MIDAS‑GIN runs and is friendly to most
modern Linux / macOS systems.

After activating the env, install CPU‑only PyTorch + PyG, for example:

```bash
pip install --upgrade pip

# Example: PyTorch 2.8.0 CPU (adjust everything if other version are of interest)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric 2.6.1 + CPU extensions
pip install torch-geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# xswap (for pathway rewiring)

pip install git+https://github.com/hetio/xswap.git@v0.0.2

```

> If you see build errors for PyG extensions, check that:
> - the `-f` URL matches your installed `torch` version; and
> - you ran the commands inside the activated conda env.  
> `pyg_lib` is not installed; `torch_sparse` is used instead.

Quick sanity check:

```bash
python -c "import torch, torch_geometric, xswap; print(torch.__version__)"
```

---

## 2. Local CPU (Windows)

Open an **Anaconda Prompt**, `cd` into the `GNN` directory and run:

```bat
conda env create -f env/environment_midas-gin_demo_cpu_py39_windows.yml
conda activate midas-gin-cpu
```

The Windows env is aligned with, for example:

- `python=3.9`
- `scikit-learn==1.2.2`
- `optuna==3.0.3`
- `sqlalchemy==1.4.44`

which are known‑good versions for Windows replication.

Then install PyTorch CPU + PyG with the appropriate Windows wheels for your PyTorch version
(see the official PyTorch and PyG docs for up‑to‑date commands), and finally:

```bat
python -c "import torch, torch_geometric, xswap; print(torch.__version__)"
```

---

## 3. CPU on the UCL CS cluster

This path is designed to reproduce the original paper’s results on the UCL CS cluster.

1. Log in and clone/copy the MIDAS project so that you have a `GNN/` folder.
2. Load *no extra modules* that might override conda’s C/C++ libraries.
3. Create the env from the CS YAML:

```bash
cd /path/to/MIDAS_Project/GNN
conda env create -f env/environment_midas-gin_demo_cpu_cs.yml
conda activate midas-gin-cpu-cs
```
4. Install PyTorch (CPU) and PyG inside this env:

```bash
pip install --upgrade pip

# PyTorch 2.6.0 CPU + torchvision
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric 2.6.1 + CPU extensions for torch 2.6.0
pip install torch-geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
  

# xswap (for pathway rewiring)

pip install git+https://github.com/hetio/xswap.git@v0.0.2

```


5. On some CS nodes the system `libstdc++.so.6` is older than what SciPy / scikit‑learn expect.
   If you see **SciPy / libstdc++ / GLIBC** errors such as:

   ```text
   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found ...
   ```

   then, *after* activating the env, run:

   ```bash
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
   python -c "import scipy.sparse; import imblearn"
   ```

   This forces the runtime to prefer the C++ libraries shipped with the conda env.

6. On the CS cluster, use `--device_choice cpu` when running the training script (see
   **Running the training demo** below).


---

## 4. GPU on the UCL Myriad cluster (recommended GPU path)

On **Myriad**, you do *not* build CUDA via conda. Instead you use the cluster’s
modules for the toolchain + CUDA/cuDNN, and then install **PyTorch + PyG inside
a Python virtualenv** with `pip` (you do *not* rely on the `pytorch` module).

The modules you need are:

- CUDA 11.3.1
- cuDNN 8.2 for CUDA 11.3
- GCC libs + Python 3.9 (see commands below)

The GPU environment is intended for **training only**; interpretability
scripts that rely on GiG rewiring can be run in a CPU environment that has `xswap`
installed.

A typical workflow is:

### 4.1. Load modules and create a venv

In a Myriad login shell:

```bash
module purge

# Core toolchain + Python + CUDA + cuDNN
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3

```

Then create and activate a virtualenv *inside your project*:

```bash

cd $HOME/MIDAS_Project/GNN

python -m venv venv-midas-gin-myriad

source venv-midas-gin-myriad/bin/activate

pip install --upgrade pip
```

### 4.2. Install Python dependencies in the venv

Inside the activated venv:

```bash
# Core scientific stack (compatible with the Myriad module toolchain)
pip install \
  numpy==1.26.4 \
  pandas==2.2.2 \
  scipy==1.13.1 \
  scikit-learn==1.4.2 \
  imbalanced-learn==0.12.0 \
  optuna==4.4.0 \
  sqlalchemy==2.0.36 \
  joblib \
  tqdm \
  pytz \
  python-dateutil \
  six
```

### 4.3. Install PyTorch with CUDA 11.3 wheels

```bash
pip install \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

```

### 4.4. Install PyTorch Geometric (PyG) and extensions

With `torch==1.12.1+cu113` already present, install
the PyG core package and matching extension wheels compiled for that torch/CUDA
combination:


```bash

# PyTorch Geometric core

pip install torch-geometric==2.3.1

# PyG extensions compiled for torch 1.12.0 + cu113

# Install the correct PyG extension wheels for torch 1.12.1 + cu113 (pyg-lib not installed)

pip install --no-index --find-links https://data.pyg.org/whl/torch-1.12.1+cu113.html \
  torch-scatter==2.1.0+pt112cu113 \
  torch-sparse==0.6.16+pt112cu113 \
  torch-cluster==1.6.0+pt112cu113 \
  torch-spline-conv==1.2.1+pt112cu113
```

Check imports and GPU visibility:

```bash
python - << 'EOF'
import sys, numpy, pandas, scipy, torch
print("sys.path:")
print("\n".join(sys.path))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)

from torch_geometric.loader import NeighborLoader
print("torch_geometric + NeighborLoader import OK")
EOF
```

If `CUDA available: True`, you are ready to run with `--device_choice gpu`.


---

## Data expectations

### Train / Test CSVs

- Located by default in `data/demo/`.
- **Index**: first column is a unique gene identifier (string).
- **Columns**: numeric features plus a binary label column (default `targ_stat`).
- You can override the label column name via `--label_col`.

### Graph pickle

- Path: `data/demo/demo_MIDAS_InputMatrix_Graph.pkl`
- A pickled `pandas.DataFrame` with two columns representing edges
  (e.g. `source_name`, `target_name`).
- The training script:
  - filters edges to match the node indices in the union of train+test CSV indices,
  - checks for dangling nodes,
  - builds an undirected, de‑duplicated `edge_index` tensor.

### Gene set CSV (for GiG / Local)

- Path: `data/demo/GeneSet_Demo_PathwaysNEW.csv`
- Contains at least:
  - a gene identifier column,
  - pathway identifiers (e.g. `PathwayID`),
  - optional descriptive columns.
- Used by the GiG‑based interpretability scripts for pathway‑aware rewiring.

---

## Training: Optuna + GIN cross‑validation

The main training script is:

```text
scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py
```

It:

- standardises/splits the data into train/test with a fixed seed,
- runs `RepeatedStratifiedKFold` CV (with repeats),
- performs Optuna hyperparameter optimisation,
- trains a GIN model on each fold,
- saves:
  - the Optuna SQLite study,
  - per‑fold best checkpoints,
  - merged predictions (train / test / held‑out),
  - logs and a JSON config snapshot.

By default, it writes into:

```text
results/train/
  ├─ optuna_runs/
  ├─ models/
  ├─ predictions/
  └─ logs/
```

### Default `output_dir`

Inside the script, the default `output_dir` is:

```text
<project_root>/results/train
```

where `project_root` is the `GNN/` directory (the folder that contains `scripts/`, `data/`, `env/`, `results/`).

You can change this with `--output_dir`, but if you omit it, **all training artefacts go under `results/train`**.

> Ensure the path is writable on your system/cluster. When running multiple jobs in parallel, use a unique --study_name (or separate --output_dir) to avoid collisions/locking on the Optuna SQLite database.

### Example: CPU run (Linux / macOS)

From the `GNN/` directory:

```bash
conda activate midas-gin-cpu

python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --study_name MIDAS_GIN_demo --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 --device_choice cpu
```

This will create (by default):

```text
results/train/
  optuna_runs/MIDAS_GIN_demo.optuna.db
  models/Best_GINBiDir_train_foldID_*.pth
  predictions/predictions_MIDAS_GIN_demo.csv
  logs/MIDAS_GIN_demo.log
  logs/config_MIDAS_GIN_demo.json
```

### Example: Windows CPU run

```bat
conda activate midas-gin-cpu

python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py ^
  --train_csv data\demo\demo_MIDAS_InputMatrix_train.csv ^
  --test_csv data\demo\demo_MIDAS_InputMatrix_test.csv  ^
  --graph_pkl data\demo\demo_MIDAS_InputMatrix_Graph.pkl ^
  --study_name MIDAS_GIN_demo ^
  --n_trials 1 --epochs 2 --cv_splits 5 --cv_repeats 1 ^
  --device_choice cpu
```

### Example: CS cluster CPU run

```bash
conda activate midas-gin-cpu-cs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /path/to/MIDAS_Project/GNN

python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --study_name MIDAS_GIN_demo --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 --device_choice cpu
```

### Example: Myriad GPU run

For quick interactive testing on a GPU node:

```bash
module purge
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3


cd $HOME/MIDAS_Project/GNN

source venv-midas-gin-myriad/bin/activate

python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --study_name MIDAS_GIN_demo_myriad_test --n_trials 1 --epochs 1 --cv_splits 2 --cv_repeats 1 --device_choice gpu
```

For a full run the submission guideline on Myriad have to be followed taking into account the code above.

---

## Optuna study location

By default (no `--optuna_db` given), the training script stores the Optuna study as:

```text
results/train/optuna_runs/<study_name>.optuna.db
```

You can override this with `--output_dir` (changes the root) or explicitly with `--optuna_db`
(via a SQLite URI or a path), but for most workflows just keep the default.

The **same Optuna DB** is reused by all interpretability scripts.

---

## Interpretability scripts

The interpretability scripts live in `scripts/interpretability/`:

- `GIN_PermImpGlobal_MIDAS_Demo_CPU.py`
- `GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py`
- `GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py`

They:

- **do not train** models;
- **rebuild the CV splits** used in training (with the same seed and CV settings);
- **load per‑fold checkpoints** from `results/train/models/`;
- for each fold/feature/seed:
  - perturb features or edges,
  - recompute predictions,
  - log performance drops as permutation scores.
  
> All interpretability scripts are intended to run in the **same conda environment**
as the training script. For example:
- locally: `midas-gin-cpu`
- on the UCL CS cluster: `midas-gin-cpu-cs`

You do not need a separate env; the scripts reuse the trained models and Optuna
study from `results/train/`.


### Default `output_dir` for interpretability

Each interpretability script has its own `output_dir`, which by default is:

```text
<project_root>/results/interpretability
```

Subdirectories are used for different kinds of outputs, e.g.:

```text
results/interpretability/
  permFeatImp/
  permEdgeGlobalImp/
  permEdgeLocalImp/
  logs/
```

The **Optuna DB is always read from**:

```text
results/train/optuna_runs/<study_name>.optuna.db
```

unless you pass `--optuna_db` explicitly.

---

## Global feature permutation importance

Script:

```text
scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py
```

Example CPU run (on UCL CS cluster):

```bash
conda activate midas-gin-cpu-cs # adjust if necessary

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /path/to/MIDAS_Project/GNN # adjust if necessary

python scripts/interpretability/GIN_PermImpGlobal_MIDAS_Demo_CPU.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --study_name MIDAS_GIN_demo --label_col targ_stat --device cpu --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

- `--seeds 0:49` means seeds 0 to 49 inclusive (50 permutations).

- The script will:

  - load the Optuna study from `results/train/optuna_runs/MIDAS_GIN_demo.optuna.db`,
  - load fold checkpoints from `results/train/models/`,
  - write permutation‑importance CSVs under:

    ```text
    results/interpretability/permFeatImp/
    ```

---

## GiG global rewiring (edge‑level importance)

Script:

```text
scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py
```

Example (on UCL CS cluster):

```bash
conda activate midas-gin-cpu-cs # adjust if necessary

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /path/to/MIDAS_Project/GNN # adjust if necessary

python scripts/interpretability/GIN_PermImpGiG_Global_MIDAS_Demo_CPU.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --study_name MIDAS_GIN_demo --label_col targ_stat --device cpu --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

Outputs:

```text
results/interpretability/permEdgeGlobalImp/
```

Each CSV encodes performance changes under global, degree‑preserving rewiring of edges consistent with pathway structure.

---

## GiG local / pathway‑local rewiring

Script:

```text
scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py
```

Example (on UCL CS cluster):

```bash
conda activate midas-gin-cpu-cs # adjust if necessary

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /path/to/MIDAS_Project/GNN # adjust if necessary


python scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv --test_csv data/demo/demo_MIDAS_InputMatrix_test.csv --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl --gene_set_csv data/demo/GeneSet_Demo_PathwaysNEW.csv --study_name MIDAS_GIN_demo --label_col targ_stat --device cpu --seeds 0:49 --cv_splits 5 --cv_repeats 1
```

Outputs:

```text
results/interpretability/permEdgeLocalImp/
```

This script restricts rewiring to pathway‑specific subgraphs and aggregates local edge importance.

---

## Reproducibility notes

- Use the **same**:
  - `study_name`,
  - `cv_splits`, `cv_repeats`,
  - `label_col`,
  - and data files
- across training and interpretability runs if you want strict consistency.
- CV folds are reproducible as long as:
  - you keep the same random seed,
  - you do not reorder the rows,
  - and you use the same scikit‑learn version.
- The training script logs a `config_<study_name>.json` file under `results/train/logs/`
  with all runtime arguments (including seeds and CV settings).

---

## Troubleshooting

- **Optuna DB not found**  
  Make sure you have run the training script for the same `study_name`
  and that `results/train/optuna_runs/<study_name>.optuna.db` exists. You
  can override the path with `--optuna_db`.

- **CUDA or GPU issues (local or Myriad)**  
  Test with:

  ```python
  import torch
  print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
  ```

  Only use `--device_choice gpu` when `cuda.is_available()` is `True`. On shared clusters,
  request a single GPU per job.

- **SciPy / libstdc++ errors on the CS cluster**  
  If you see errors such as:

  ```text
  ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found ...
  ```

  ensure you:

  1. Activate `midas-gin-cpu-cs`:

     ```bash
     conda activate midas-gin-cpu-cs
     ```

  2. Export the env’s `lib` directory first in `LD_LIBRARY_PATH`:

     ```bash
     export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
     ```

  Run your job **after** these steps (or put them in your batch script).

- **PyG GLIBC warnings on older Linux clusters**  
  You may see warnings like:

  ```text
  GLIBC_2.32 not found (required by .../torch_scatter/_version_cpu.so)
  ```

  In that case, PyG disables those compiled extensions and falls back to
  alternative implementations. Training and interpretability scripts should
  still run; these warnings can usually be ignored unless you encounter a
  hard error.

---

## Citation and attribution

- GIN implementation is built on **PyTorch Geometric**.
- Pathway‑based edge rewiring uses the **`xswap`** library.
- If you use this code in academic work, please cite the associated MIDAS‑GIN paper
  (https://www.researchsquare.com/article/rs-5499857/v1).
