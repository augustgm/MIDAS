#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold
=================================================
CPU/GPU pipeline to train and evaluate a GIN classifier on MIDAS-style
gene target data using node features (CSV) and a graph (edge list PKL).
Hyperparameters are tuned with Optuna; training/eval use PyG NeighborLoader.

On CPU, CV folds are evaluated in parallel with joblib. On GPU, folds are
evaluated sequentially in a single process to avoid CUDA + multiprocessing
issues (so behaviour is stable on clusters).


Paper
=====

This code allows for the reproduction of the results published in:
    
    https://www.researchsquare.com/article/rs-5499857/v1
    

The demo data frames provided here follow a similar structure to the data used in 
the paper.

See the paper's data availability.


Summary
-------
- Reads train/test CSVs (first column = gene ID; includes a binary label).
- Loads a pickled pandas DataFrame edge list; infers source/target columns.
- Builds an undirected, de-duplicated 'edge_index' with strict sanity checks.
- RepeatedStratifiedKFold on training nodes only with class undersampling.
- Optuna TPE + MedianPruner for tuning; best params reused for per-fold fits.
- Saves per-fold models and consolidated predictions (train/test/held-out).
- Supports CPU or a single GPU via the '--device' flag.


Files created
-------------
output_dir/
  ├─ models/Best_GINBiDir_train_foldID_<k><study_name>.pth
  ├─ predictions/predictions_<study_name>.csv
  ├─ optuna_runs/<safe(study_name)>.optuna.db        (unless --in_memory_optuna)
  └─ logs/<safe(study_name)>.log

Inputs
------
- Train CSV: first column = unique gene ID; contains binary label column
  (default 'targ_stat' or overridable via '--label_col').
- Test  CSV: same structure as train (index + features + label).
- Graph PKL: pickled pandas DataFrame with two endpoint columns
  (e.g. 'source_name'/'target_name'); will be symmetrised and deduplicated.
  
Assumptions & requirements
--------------------------
- Labels are strictly binary {0, 1}. The script exits if not.
- Node sets between features and graph must match exactly (no danglers).
- Optuna storage defaults to a local SQLite file unless '--in_memory_optuna'.

Command-line parameters
-----------------------
All parameters below are available as CLI flags (see Examples).

Parameters
----------
--train_csv : pathlib.Path, required
    Path to the training CSV. First column is gene ID (index). Must include
    the binary label column (see '--label_col').
--test_csv : pathlib.Path, required
    Path to the test/held-out CSV. Same schema as '--train_csv'.
--graph_pkl : pathlib.Path, required
    Path to a pickled pandas DataFrame representing the graph edge list.
    Endpoint column names are inferred from strict aliases; see notes above.

--output_dir : pathlib.Path, optional (default: directory of this script)
    Root output directory. Subfolders for 'models/', 'predictions/',
    'optuna_runs/', and 'logs/' are created as needed.

--study_name : str, optional (default: "GINBiDir_Optuna_MIDAS_demo")
    Human-readable study identifier used to name artifacts (DB/predictions/models/logs).
--optuna_db : str, optional (default: None)
    SQLAlchemy storage URI for Optuna (e.g., 'sqlite:///path/to.db').
    If omitted, a local SQLite DB is created under 'output_dir/optuna_runs/'.

--n_trials : int, optional (default: 100)
    Number of Optuna trials to run.
--epochs : int, optional (default: 100)
    Training epochs per trial (and for the final per-fold fits).
--batch_size : int, optional (default: 256)
    NeighborLoader batch size for training. (Evaluation uses full masks per set.)
--seed : int, optional (default: 42)
    Global seed for Python, NumPy, and PyTorch.
--cv_splits : int, optional (default: 10)
    Number of stratified folds for cross-validation (on training nodes only).
--cv_repeats : int, optional (default: 10)
    Number of repetitions for RepeatedStratifiedKFold.
--label_col : str, optional (default: 'targ_stat')
    Name of the binary target column in the CSVs (case-insensitive match).


--device_choice : {'cpu', 'gpu'}, optional (default: 'cpu')
    Device selection. In order to use GPU, the appropriate environment has to be created.



Exit codes
----------
2   Missing input file.
3   Optuna storage error (e.g., SQLite DB locked).
4   Graph/feature validation failed (mismatched nodes, dangling nodes, etc.).
5   Labels not strictly binary {0,1}.
7   A CV fold produced an empty validation mask.
130 Keyboard interrupt.
1   Uncaught error.

Examples (on Mac)
-----------------
Basic CPU run (small test):

    python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py \
        --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
        --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv  \
        --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
        --study_name MIDAS_GIN_demo \
        --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 \
        --device_choice cpu

Basic GPU run (single GPU, sequential folds):

    python scripts/train/DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py \
        --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
        --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv  \
        --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
        --study_name MIDAS_GIN_demo \
        --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 \
        --device_choice gpu
        
Examples (on Windows):
--------
Basic CPU run (small test):
    python scripts/train/DevelopInferMIDAS-GIN_Demo_CPU_logsfold.py^
     --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
     --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
     --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
     --study_name MIDAS_GIN_demo^
     --n_trials 10 --epochs 5 --cv_splits 5 --cv_repeats 1 --device_choice cpu


Notes
-----
- If '--device_choice gpu' is requested but CUDA is not available, the script logs
  a warning and falls back to CPU.
- Optuna storage defaults to a SQLite file in 'output_dir/optuna_runs'
  unless '--in_memory_optuna' is set.
"""



# =============================================================================
# Import libraries
# =============================================================================


from __future__ import annotations

# ── Standard library imports ──────────────────────────────────────────────────

import argparse
import hashlib
import json
import logging
import pickle
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from collections import defaultdict
from typing import Hashable, Optional, Tuple, Literal
from joblib import Parallel, delayed
import re
from urllib.parse import urlparse
import os
import importlib, types

# ── Third-party imports ───────────────────────────────────────────────────────

import numpy as np
import optuna
from optuna.storages import RDBStorage
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# --- Validate heavy dependencies early with clear error messages. ------------

try:
    import torch
    import torch.nn as nn
    from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
except Exception as e:
    raise RuntimeError("PyTorch is required. Try: conda install pytorch -c pytorch") from e

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import GINConv
except Exception as e:
    raise RuntimeError("PyTorch Geometric is required. Try: conda install pyg -c pyg -c pytorch") from e


# --- Project root. -----------------------------------------------------------


# Project root is the directory that contains `scripts/`, `data/`, etc.
# This assumes this file lives in: <project_root>/scripts/train/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "train"





# =============================================================================
# Helper functions
# =============================================================================

def safe_study_filename(name: str) -> str:
    """Make a filesystem-safe filename from a study name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s or "optuna_study"

def sqlite_uri_from_path(p: Path) -> str:
    """Build a sqlite SQLAlchemy URI from a local file path."""
    return "sqlite:///" + str(p.resolve())

def sqlite_file_from_uri(uri: str) -> Optional[Path]:
    """Return a local sqlite file Path if 'uri' is a file-based sqlite URI; else None."""
    parsed = urlparse(uri)
    if parsed.scheme != "sqlite":
        return None
    if parsed.path in (":memory:", "", None):
        return None
    return Path(parsed.path.lstrip("/")).expanduser()

def require_binary_01_labels(y, label_name="labels") -> None:
    """Ensure y contains exactly two classes {0,1}; otherwise log and exit."""
    vals = np.unique(pd.Series(y).dropna())
    ok = set(vals.tolist()) == {0.0, 1.0} or set(vals.tolist()) == {0, 1}
    if not ok:
        logging.warning(
            f"Expected binary {{0,1}} {label_name}, but found values: {vals.tolist()}. "
            "This script uses a single-logit head (BCEWithLogitsLoss). Exiting."
        )
        sys.exit(5)

# =============================================================================
# Configuration, seeding, logging
# =============================================================================

@dataclass
class Config:
    """Configuration for data paths, training, and hyperparameter search.
    
    - Ensures 'output_dir' exists.
    - If 'optuna_db' is None, builds a SQLite URI at:
      '<output_dir>/optuna_runs/<safe(study_name)>.optuna.db' and creates the
      parent directory.
    - If 'optuna_db' is a SQLite file URI (e.g., 'sqlite:///path/to.db'), the
      parent directory is created if missing.
    """
    train_csv: Path
    test_csv: Path
    graph_pkl: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    
    study_name: str = "GINBiDir_Optuna_MIDAS_demo"
    optuna_db: Optional[str] = None  # derived from output_dir + study if None

    n_trials: int = 100
    epochs: int = 100
    batch_size: int = 256

    seed: int = 42  # seed for Python, NumPy, PyTorch

   

    cv_splits: int = 10
    cv_repeats: int = 10

    label_col: str = "targ_stat"

    device_choice: Literal["cpu", "gpu"] = "cpu"  # 'cpu' or 'gpu'

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.optuna_db is None:
            runs_dir = self.output_dir / "optuna_runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            db_file = runs_dir / f"{safe_study_filename(self.study_name)}.optuna.db"
            self.optuna_db = sqlite_uri_from_path(db_file)
        else:
            db_path = sqlite_file_from_uri(self.optuna_db)
            if db_path is not None:
                db_path.parent.mkdir(parents=True, exist_ok=True)




def save_config(cfg: Config, path: Path) -> None:
    """Save the current configuration to JSON for reproducibility."""
    data = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(cfg).items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)



def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless on CPU-only runs

def setup_logging(verbosity: str = "INFO", log_file: Path | None = None) -> None:
    """Initialise the root logger with a simple, timestamped format."""
    level = getattr(logging, verbosity.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    
    # clear existing handlers 
    
    for h in list(root.handlers):
        root.removeHandler(h)
    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)
    # optional file (per-process to avoid contention during Parallel)
    if log_file is not None:
        per_proc = log_file.with_name(f"{log_file.stem}_{os.getpid()}{log_file.suffix}")
        per_proc.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(per_proc, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
        

def get_device(device_choice: str) -> torch.device:
    """
    Resolves the device to use: CUDA if available/requested, else CPU.
    """
    device_choice = device_choice.lower()

    if device_choice == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("Using GPU (CUDA)")
        else:
            device = torch.device("cpu")
            logging.warning("GPU requested, but CUDA is not available. Falling back to CPU.")
    else: 
        device = torch.device("cpu")
        logging.info("Using CPU")

    return device

def sha256sum(path: Path) -> str:
    """Compute the SHA-256 checksum of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# =============================================================================
# Data utilities
# =============================================================================

def _compat_pickle_load(path: Path):
    """Load a pickle that may reference 'numpy._core' (NumPy 2.x pickles)."""
    try:
        import numpy._core  # noqa: F401
    except Exception:
        core_proxy = types.ModuleType("numpy._core")
        core_proxy.multiarray   = importlib.import_module("numpy.core.multiarray")
        core_proxy.numerictypes = importlib.import_module("numpy.core.numerictypes")
        core_proxy.umath        = importlib.import_module("numpy.core.umath")
        sys.modules.setdefault("numpy._core", core_proxy)
        sys.modules.setdefault("numpy._core.multiarray",   core_proxy.multiarray)
        sys.modules.setdefault("numpy._core.numerictypes", core_proxy.numerictypes)
        sys.modules.setdefault("numpy._core.umath",        core_proxy.umath)
        try:
            core_proxy._multiarray_umath = importlib.import_module("numpy.core._multiarray_umath")
            sys.modules.setdefault("numpy._core._multiarray_umath", core_proxy._multiarray_umath)
        except ModuleNotFoundError:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)

def read_midas_csv(path: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read a MIDAS-style feature table into (X, y) with strict validation.
    - Row index: gene identifier (first CSV column).
    - Columns: numeric features + one binary label column (0/1).
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()

    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate gene IDs in {path.name}: e.g. {dup[:8]} (gene IDs must be unique)."
        )

    # Find label column (case-insensitive fallback)
    if label_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == label_col.lower()]
        if matches:
            label_col = matches[0]
        else:
            candidates = [c for c in df.columns if c.lower() in {"target", "label", "targ_stat"}]
            raise KeyError(
                f"Label column '{label_col}' not found in {path.name}. "
                f"Available: {list(df.columns)[:8]}... Try one of: {candidates or 'targ_stat'}"
            )

    y = pd.to_numeric(df.pop(label_col), errors="coerce")
    vals = pd.Series(y).dropna().unique().tolist()
    if not set(vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Label column '{label_col}' in {path.name} must be binary 0/1; found values: {sorted(vals)}"
        )
    y = y.astype(float)

    X = df.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X.to_numpy()).all():
        bad_cols = X.columns[~np.isfinite(X).all()].tolist()
        raise ValueError(
            f"Non-numeric or non-finite values found in feature columns in {path.name}: "
            f"{bad_cols[:8]} (showing up to 8). Clean your CSV."
        )

    if X.shape[1] == 0:
        raise ValueError(f"No feature columns left in {path.name} after parsing.")
    return X, y

def infer_edge_columns(edges: pd.DataFrame) -> Tuple[Hashable, Hashable]:
    """Infer (source, target) column names from an edge-list DataFrame (strict)."""
    if not isinstance(edges, pd.DataFrame):
        raise TypeError("edges must be a pandas DataFrame.")
    if edges.shape[1] < 2:
        raise ValueError("Edge list must have at least two columns for source/target.")
    if isinstance(edges.columns, pd.MultiIndex):
        raise ValueError("MultiIndex columns are not supported; please flatten/rename columns.")

    def canon(name: str) -> str:
        s = str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        return "".join(ch for ch in s if ch.isalnum())

    canon_to_orig = defaultdict(list)
    for orig in list(edges.columns):
        canon_to_orig[canon(orig)].append(orig)

    ambiguous = [vals for vals in canon_to_orig.values() if len(vals) > 1]
    if ambiguous:
        flat = [v for vals in ambiguous for v in vals]
        raise ValueError(
            "Ambiguous column names differing only by case/spacing/underscores: "
            f"{flat}. Please rename to distinct, unambiguous names."
        )

    source_aliases = ["source_name", "source", "src", "u", "from"]
    target_aliases = ["target_name", "target", "dst", "v", "to"]

    s_col = next((canon_to_orig[canon(a)][0] for a in source_aliases if canon(a) in canon_to_orig), None)
    t_col = next((canon_to_orig[canon(a)][0] for a in target_aliases if canon(a) in canon_to_orig), None)

    if s_col is None or t_col is None or s_col == t_col:
        raise ValueError(
            "Could not infer distinct source/target columns from known aliases.\n"
            f"Available columns: {list(edges.columns)}\n"
            "Expected a pair like ('source','target'), ('src','dst'), ('u','v'), "
            "('source_name','target_name')."
        )
    return s_col, t_col

def load_graph_to_edge_index(graph_pkl: Path, node_index: pd.Index) -> torch.Tensor:
    """
    Load a pickled pandas edge list and return a strict PyG 'edge_index' tensor.
    - Node sets must match between graph and features (no graph-only or feature-only nodes).
    - No dangling nodes (degree == 0) after filtering to 'node_index' (raises).
    - Warn if >1 connected component among non-dangling nodes.
    """
    max_examples = 10
    obj = _compat_pickle_load(graph_pkl)
    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Expected a pickled pandas DataFrame for the graph PKL.")
    edges = obj.copy()

    s_col, t_col = infer_edge_columns(edges)
    edges[s_col] = edges[s_col].astype(str).str.strip()
    edges[t_col] = edges[t_col].astype(str).str.strip()

    graph_nodes_raw = pd.Index(pd.unique(pd.concat([edges[s_col], edges[t_col]], ignore_index=True)).astype(str))
    feature_nodes = pd.Index(node_index.astype(str))

    in_graph_only = graph_nodes_raw.difference(feature_nodes)
    in_features_only = feature_nodes.difference(graph_nodes_raw)
    if len(in_graph_only) or len(in_features_only):
        msg = (
            "Feature/graph node mismatch detected. "
            f"{len(in_graph_only)} node(s) only in graph, "
            f"{len(in_features_only)} node(s) only in features. "
            f"Examples (graph-only): {list(map(str, in_graph_only[:max_examples]))}; "
            f"(features-only): {list(map(str, in_features_only[:max_examples]))}."
        )
        logging.error(msg)
        raise ValueError(msg)

    edges_f = edges[edges[s_col].isin(feature_nodes) & edges[t_col].isin(feature_nodes)]
    logging.info(f"Graph: {len(edges)} edges raw; {len(edges_f)} after filtering to feature index.")

    node_to_idx = {g: i for i, g in enumerate(feature_nodes)}
    if len(edges_f):
        src = edges_f[s_col].map(node_to_idx).to_numpy(dtype=np.int64)
        dst = edges_f[t_col].map(node_to_idx).to_numpy(dtype=np.int64)
        edge_index_np = np.hstack([
            np.stack([src, dst], 0),
            np.stack([dst, src], 0),
        ])
    else:
        edge_index_np = np.empty((2, 0), dtype=np.int64)

    N = len(feature_nodes)
    if edge_index_np.size:
        enc = edge_index_np[0] * N + edge_index_np[1]
        keep = np.unique(enc, return_index=True)[1]
        keep.sort(kind="mergesort")
        edge_index_np = edge_index_np[:, keep]

    if edge_index_np.size:
        deg = np.bincount(edge_index_np.ravel(), minlength=N)
    else:
        deg = np.zeros(N, dtype=np.int64)
    dangling_mask = (deg == 0)
    if np.any(dangling_mask):
        dangling_ids = feature_nodes[dangling_mask]
        msg = (
            f"Dangling nodes detected (degree==0) after filtering: {dangling_ids.size}. "
            f"E.g. {list(map(str, dangling_ids[:max_examples]))}"
        )
        logging.error(msg)
        raise ValueError(msg)

    if edge_index_np.size:
        parent = np.arange(N, dtype=np.int64)
        rank = np.zeros(N, dtype=np.int8)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        u, v = edge_index_np[0], edge_index_np[1]
        uu, vv = np.minimum(u, v), np.maximum(u, v)
        undirected = np.unique(uu * N + vv)
        for enc in undirected:
            a = int(enc // N)
            b = int(enc % N)
            union(a, b)

        roots = np.array([find(i) for i in range(N)], dtype=np.int64)
        n_comp = np.unique(roots).size
        if n_comp > 1:
            logging.warning(f"Graph is not connected: {n_comp} component(s) among non-dangling nodes.")

    return torch.tensor(edge_index_np, dtype=torch.long)

def load_midas_data(train_csv: Path, test_csv: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index]:
    """Load train/test CSVs and return concatenated features/labels + index splits."""
    X_tr, y_tr = read_midas_csv(train_csv, label_col=label_col)
    X_te, y_te = read_midas_csv(test_csv,  label_col=label_col)
    X_all = pd.concat([X_tr, X_te], axis=0)
    y_all = pd.concat([y_tr, y_te], axis=0).reindex(X_all.index)
    return X_all, y_all, X_tr.index, X_te.index

# =============================================================================
# Model
# =============================================================================
class GNN(nn.Module):
    """
    Graph Isomorphism Network (GIN) with small pre/post MLP stacks
    """
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_hidden: int,
        num_layers: int,
        num_layers_pre: int,
        num_layers_post: int,
        dropout: float,
        rate: float,
        eps: float,
        train_eps: bool,
    ) -> None:
        super().__init__()
        if n_classes != 1:
            raise ValueError("This model is configured for BCEWithLogitsLoss; set n_classes=1.")

        def mlp_stack(n_layers: int, in_dim: int, hid_dim: int) -> nn.ModuleList:
            layers = [Sequential(Linear(in_dim, hid_dim), BatchNorm1d(hid_dim), ReLU())]
            for _ in range(n_layers - 1):
                layers.append(Sequential(Linear(hid_dim, hid_dim), BatchNorm1d(hid_dim), ReLU()))
            return nn.ModuleList(layers)

        self.pre  = mlp_stack(max(1, num_layers_pre),  n_features, n_hidden)
        self.post = mlp_stack(max(1, num_layers_post), n_hidden,    n_hidden)

        self._conv_mult = [2 if i < (num_layers - 1) else 1 for i in range(num_layers)]
        self.convs = nn.ModuleList([
            GINConv(
                Linear(
                    n_hidden * (self._conv_mult[i - 1] if i > 0 else 1),
                    n_hidden * self._conv_mult[i]
                ),
                eps=eps, train_eps=train_eps
            )
            for i in range(num_layers)
        ])

        self.cnorms = nn.ModuleList([BatchNorm1d(n_hidden * m) for m in self._conv_mult])
        self.crelus = nn.ModuleList([ReLU() for _ in range(num_layers)])

        self.pre_drops  = nn.ModuleList([nn.Dropout(dropout * (rate ** i)) for i in range(len(self.pre))])
        self.conv_drops = nn.ModuleList([nn.Dropout(dropout * (rate ** i)) for i in range(len(self.convs))])
        self.post_drops = nn.ModuleList([nn.Dropout(dropout * (rate ** i)) for i in range(len(self.post))])

        self.head = Linear(n_hidden, n_classes)

    @staticmethod
    def _apply_stack(stack: nn.ModuleList, drops: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        assert len(stack) == len(drops), "Stack and dropout lists must be the same length."
        for layer, drop in zip(stack, drops):
            x = drop(layer(x))
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._apply_stack(self.pre, self.pre_drops, x)
        for conv, bn, relu, drop in zip(self.convs, self.cnorms, self.crelus, self.conv_drops):
            x = conv(x, edge_index)
            x = drop(relu(bn(x)))
        x = self._apply_stack(self.post, self.post_drops, x)
        return self.head(x)  # [N, 1] logits for BCEWithLogitsLoss

# =============================================================================
# Train / Inference utils
# =============================================================================

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: NeighborLoader,
    device: torch.device,
) -> float:
    """Train the model for one epoch using BCEWithLogitsLoss on training nodes only."""
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    total_loss = 0.0
    total_seen = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x, batch.edge_index).squeeze(-1)

        m = batch.train_mask
        if m is None:
            raise RuntimeError("Batch is missing 'train_mask'.")
        if not torch.any(m):
            continue

        masked_logits  = logits[m]
        masked_targets = batch.y[m].to(logits.dtype)

        loss = criterion(masked_logits, masked_targets)
        loss.backward()
        optimizer.step()

        bs = masked_logits.numel()
        total_loss += float(loss.detach().cpu()) * bs
        total_seen += int(bs)

    return (total_loss / total_seen) if total_seen > 0 else 0.0

def predict_proba(
    model: nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    mask_name: str = "test_mask",
    node_names: Optional[pd.Index] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | pd.Index]:
    """Predict p(y=1) for nodes selected by 'mask_name' in each mini-batch."""
    model.eval()
    ys, yhats, idxs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, mask_name):
                raise AttributeError(f"Batch has no attribute '{mask_name}'.")
            mask = getattr(batch, mask_name).bool()
            if not torch.any(mask):
                continue

            logits = model(batch.x, batch.edge_index).squeeze(-1)
            probs  = torch.sigmoid(logits)

            ys.append(batch.y[mask].detach().cpu().view(-1))
            yhats.append(probs[mask].detach().cpu().view(-1))
            idxs.append(batch.n_id[mask].detach().cpu().view(-1))

    if not yhats:
        empty = np.array([], dtype=np.float32)
        empty_idx = np.array([], dtype=np.int64)
        return empty, empty, (node_names[empty_idx] if node_names is not None else empty_idx)

    y    = torch.cat(ys).numpy()
    yhat = torch.cat(yhats).numpy()
    nidx = torch.cat(idxs).numpy()
    names = node_names[nidx] if node_names is not None else nidx
    return y, yhat, names

# =============================================================================
# Shared fold runner
# =============================================================================
def run_fold(
    *,
    xx: pd.DataFrame,
    yy: pd.Series,
    nodes_train: pd.Index,
    nodes_heldout: Optional[pd.Index],  # global held-out nodes ( 'nodes_test')
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Network: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    n_features: int,
    n_classes: int,
    n_hidden: int,
    num_layers: int,
    num_layers_pre: int,
    num_layers_post: int,
    dropout: float,
    rate: float,
    eps: float,
    train_eps: bool,
    lr: float,
    weight_decay: float,
    batch_size: int,
    neighbors: list[int],
    epochs: int,
    device: torch.device,
    fold_id: Optional[int] = None,
    save_model_path: Optional[Path] = None,  # if provided, save state_dict here
    seed: int = 0,                            # per-worker seed for loaders
    return_sets: tuple[str, ...] = ("test",), # any of {"train","test","heldout"}
    return_model: bool = False,               # keep False for Parallel
    log_root: Optional[Path] = None,
) -> dict:
    
    """
    Train on a CV fold and optionally return predictions for selected sets.
    Returns a dict with keys possibly including: 'auc_test', 'df_train', 'df_test', 'df_heldout'.
    """
    
    # --- Per-worker logging bootstrap ---
    try:
        logs_dir = (log_root or (Path.cwd() / "logs" / "logs_fold")).resolve()
        
        # setup_logging will mkdir(parents=True) for the file's parent, but this
        # ensures the directory exists even before we compute the file path.
        
        logs_dir.mkdir(parents=True, exist_ok=True)

        # One file per fold/process; setup_logging will append _<pid> automatically
        
        log_file = logs_dir / f"logs_fold{fold_id}.log"
        
        setup_logging(verbosity="INFO", log_file=log_file)

        logging.info("Worker logging ready | fold=%s | pid=%s", fold_id, os.getpid())
    except Exception as e:
        print(f"[run_fold] logging init failed for fold={fold_id}: {e}", file=sys.stderr)
        
  
    # --- Restrict to nodes_train universe and derive per-fold train/val splits
    
    X_truniv = xx.loc[xx.index.isin(nodes_train), :]
    y_truniv = yy.loc[yy.index.isin(nodes_train)]
    X_train, y_train = X_truniv.iloc[train_idx, :], y_truniv.iloc[train_idx]
    X_val = X_truniv.iloc[test_idx, :]

    # --- RandomUnderSampler with identity preservation
    
    rus = RandomUnderSampler(random_state=42)
    Xtr_df = X_train.reset_index().rename(columns={"index": "_row_id_"})
    ytr_s  = y_train.reset_index(drop=True)
    X_res_df, y_res_s = rus.fit_resample(Xtr_df, ytr_s)
    kept_ids = set(X_res_df["_row_id_"].astype(str))

    # --- Build PyG Data with masks
    Xgraph = torch.tensor(xx.values, dtype=torch.float)
    ygraph = torch.tensor(yy.values, dtype=torch.float)
    data = Data(x=Xgraph, edge_index=Network, edge_attr=edge_weights, y=ygraph)

    xx_index = pd.Index(xx.index)
    kept_ids_idx = pd.Index(kept_ids)
    data.train_mask = torch.as_tensor(xx_index.isin(kept_ids_idx), dtype=torch.bool)
    data.test_mask  = torch.as_tensor(xx_index.isin(pd.Index(X_val.index)), dtype=torch.bool)
    data.val_mask   = torch.as_tensor(
        xx_index.isin(pd.Index(nodes_heldout) if nodes_heldout is not None else pd.Index([])),
        dtype=torch.bool
    )

    if data.test_mask.sum().item() == 0:
        logging.error(f"test_mask in fold {fold_id} is all False — no validation nodes to score; Exiting")
        sys.exit(7)

    # --- Loaders
    torch.manual_seed(seed)
    train_loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=int(batch_size),
        input_nodes=data.train_mask,
    )

    # --- Model / Optimiser
    model = GNN(
        n_features, n_classes, n_hidden, int(num_layers),
        int(num_layers_pre), int(num_layers_post),
        float(dropout), float(rate), float(eps), bool(train_eps)
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    # --- Train
    for _ in range(int(epochs)):
        train_epoch(model, optimiser, train_loader, device)

    # --- Save model if requested
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_model_path))

    result: dict = {}

    # --- Predict helpers
    def _predict(mask_name: str, set_label: str):
        bs = int(getattr(data, mask_name).sum().item()) or 1
        torch.manual_seed(seed)
        loader = NeighborLoader(data, num_neighbors=neighbors, batch_size=bs, input_nodes=getattr(data, mask_name))
        y, yhat, names = predict_proba(model, loader, device, mask_name=mask_name, node_names=xx.index)
        df = pd.DataFrame({"Gene": np.asarray(names), "Prob": yhat, "Target": y})
        if fold_id is not None:
            df["Fold_ID"] = fold_id
        df["Set"] = set_label
        return df

    # Test/validation fold
    if "test" in return_sets:
        df_test = _predict("test_mask", "Test fold (pred with fold model)")
        result["df_test"] = df_test
        if df_test.shape[0] > 0:
            result["auc_test"] = float(roc_auc_score(df_test["Target"], df_test["Prob"]))
        else:
            result["auc_test"] = float("nan")

    # Train predictions (over the training mask)
    if "train" in return_sets:
        df_train = _predict("train_mask", "Train fold (pred with fold model)")
        result["df_train"] = df_train

    # Held-out predictions (global test nodes supplied as nodes_heldout)
    if "heldout" in return_sets and nodes_heldout is not None and len(nodes_heldout) > 0:
        df_held = _predict("val_mask", "Held out (pred with fold model)")
        result["df_heldout"] = df_held

    if return_model:
        result["model"] = model  # WARNING: not picklable under joblib; keep False in Parallel
    return result

# =============================================================================
# Optuna objective + wrappers
# =============================================================================
def objective(
    trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
    nodes_train: pd.Index,
    nodes_test: pd.Index,
    Network: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_epochs: int,
    device: torch.device,
    log_root: Optional[Path],
) -> float:
    """Optuna objective that samples hyperparameters and averages CV AUC."""
    params = {
        "lr":              trial.suggest_float("lr", 1e-4, 5e-2, log=True),
        "n_hidden":        trial.suggest_int("n_hidden", 32, 256, step=32),
        "dropout":         trial.suggest_float("dropout", 0.0, 0.7),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "num_layers":      trial.suggest_int("num_layers", 1, 5),
        "eps":             trial.suggest_float("eps", 0.0, 0.1),
        "train_eps":       trial.suggest_categorical("train_eps", [False, True]),
        "num_layers_pre":  trial.suggest_int("num_layers_pre", 1, 3),
        "num_layers_post": trial.suggest_int("num_layers_post", 1, 3),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        "n_b":             trial.suggest_int("n_b", 5, 200),
        "rate":            trial.suggest_float("rate", 0.05, 1.0),
    }

    # Restrict CV to training nodes only and materialize splits
    X_tr = X.loc[X.index.isin(nodes_train), :]
    y_tr = y.loc[y.index.isin(nodes_train)]
    splits = list(cv.split(X_tr, y_tr))

    neighbors = [int(params["n_b"])] * int(params["num_layers"])
    
   

    if device.type == "cuda":
        # --- GPU: run folds sequentially in this process (avoid CUDA + multiprocessing)
        scores = []
        for k, (tr_idx, va_idx) in enumerate(splits):
            res = run_fold(
                xx=X, yy=y,
                nodes_train=nodes_train,
                nodes_heldout=None,
                train_idx=tr_idx,
                test_idx=va_idx,
                Network=Network,
                edge_weights=edge_weights,
                n_features=X.shape[1], n_classes=1,
                n_hidden=params["n_hidden"],
                num_layers=params["num_layers"],
                num_layers_pre=params["num_layers_pre"],
                num_layers_post=params["num_layers_post"],
                dropout=params["dropout"],
                rate=params["rate"],
                eps=params["eps"],
                train_eps=params["train_eps"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                neighbors=neighbors,
                epochs=num_epochs,
                device=device,
                fold_id=k,
                save_model_path=None,
                seed=0 + k,
                return_sets=("test",),
                return_model=False,
                log_root=log_root,
            )
            scores.append(res)
    else:
        # --- CPU: parallelise across folds
        scores = Parallel(n_jobs=-1, backend="loky")(
            delayed(run_fold)(
                xx=X, yy=y,
                nodes_train=nodes_train,
                nodes_heldout=None,
                train_idx=tr_idx,
                test_idx=va_idx,
                Network=Network,
                edge_weights=edge_weights,
                n_features=X.shape[1], n_classes=1,
                n_hidden=params["n_hidden"],
                num_layers=params["num_layers"],
                num_layers_pre=params["num_layers_pre"],
                num_layers_post=params["num_layers_post"],
                dropout=params["dropout"],
                rate=params["rate"],
                eps=params["eps"],
                train_eps=params["train_eps"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                neighbors=neighbors,
                epochs=num_epochs,
                device=device,
                fold_id=k,
                save_model_path=None,
                seed=0 + k,
                return_sets=("test",),
                return_model=False,
                log_root=log_root,
            )
            for k, (tr_idx, va_idx) in enumerate(splits)
        )



    aucs = [float(s["auc_test"]) for s in scores]
    return float(np.nanmean(aucs))

def run_optimisation(
    n_trials: int,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
    nodes_train: pd.Index,
    nodes_test: pd.Index,
    Network: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    num_epochs: int,
    study_name: str,
    storage: Optional[str],
    device: torch.device,
    log_root: Optional[Path],
) -> optuna.study.Study:
    """Create and run an Optuna study, returning the fitted Study object."""
    sampler = optuna.samplers.TPESampler(seed=0)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda t: objective(
            t,
            X=X,
            y=y,
            cv=cv,
            nodes_train=nodes_train,
            nodes_test=nodes_test,
            Network=Network,
            edge_weights=edge_weights,
            num_epochs=num_epochs,
            device=device,
            log_root=log_root,
        ),
        n_trials=n_trials,
    )
    return study

# =============================================================================
# CLI
# =============================================================================
def parse_args() -> Config:
    """Parse command-line options and construct a 'Config'."""
    parser = argparse.ArgumentParser(description="MIDAS-GIN Training and Inference Script with Optuna.")
    parser.add_argument("--train_csv", type=Path, required=True, help="Path to train CSV (first column = gene ID).")
    parser.add_argument("--test_csv",  type=Path, required=True, help="Path to test CSV (first column = gene ID).")
    parser.add_argument("--graph_pkl", type=Path, required=True, help="Path to graph PKL (edge list DataFrame).")

    parser.add_argument("--output_dir", type=Path, default=Config.output_dir, help="Directory for outputs/artefacts.")
    parser.add_argument("--study_name", type=str, default=Config.study_name, help="Optuna study name.")
    parser.add_argument("--optuna_db", type=str, default=None, help="Optuna storage URI. Default: <output_dir>/<study_name>.optuna.db")
    parser.add_argument("--in_memory_optuna", action="store_true", help="Run Optuna in-memory (no DB file).")

    parser.add_argument("--n_trials", type=int, default=Config.n_trials, help="Optuna trials.")
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="Epochs per trial/final fit.")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size, help="Base batch size (ignored for eval).")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed.")
    parser.add_argument("--cv_splits", type=int, default=Config.cv_splits, help="CV splits.")
    parser.add_argument("--cv_repeats", type=int, default=Config.cv_repeats, help="CV repeats.")
    parser.add_argument("--label_col", type=str, default=Config.label_col, help="Label column name (default: targ_stat).")

    # CPU and GPU, supported
    parser.add_argument("--device_choice", type=str, choices=["cpu", "gpu"], default="cpu",
                        help="Device selection ('cpu' or 'gpu').")
    
    args = parser.parse_args()
    if args.in_memory_optuna:
        args.optuna_db = None

    cfg = Config(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        graph_pkl=args.graph_pkl,
        output_dir=args.output_dir,
        study_name=args.study_name,
        optuna_db=args.optuna_db,
        n_trials=args.n_trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        label_col=args.label_col,
        device_choice=args.device_choice,
    )
    return cfg

# =============================================================================
# MAIN
# =============================================================================
def main(cfg: Config) -> None:
    """End-to-end training, Optuna tuning, and evaluation (CPU or single GPU)."""
 
    # Limit thread oversubscription when using joblib + torch
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # --- Setup logging ---
    
    logs_dir = cfg.output_dir / "logs"
    log_path = logs_dir / f"{safe_study_filename(cfg.study_name)}.log"
    setup_logging(verbosity="INFO", log_file=log_path)
    
    # --- Save config snapshot ---
    
    config_path = logs_dir / f"config_{safe_study_filename(cfg.study_name)}.json"
    save_config(cfg, config_path)
    logging.info("Saved run config to %s", config_path)

    # --- Seeds & device ---
    
    set_seed(cfg.seed)
    device = get_device(cfg.device_choice)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Log configuration ----
    
    logging.info(f"Config:\n{json.dumps(asdict(cfg), indent=2, default=str)}")

    # ---- Input file checksums (provenance) ----
    
    chksums = {
        "sha256_train_csv": sha256sum(cfg.train_csv),
        "sha256_test_csv":  sha256sum(cfg.test_csv),
        "sha256_graph_pkl": sha256sum(cfg.graph_pkl),
    }
    logging.info(f"Input file checksums: {json.dumps(chksums, indent=2)}")

    # ---- Load data ----
    
    X_all, y_all, nodes_train, nodes_test = load_midas_data(cfg.train_csv, cfg.test_csv, label_col=cfg.label_col)
    logging.info(f"Features: {X_all.shape}, Labels: {y_all.shape}")

    # --- Check labels
    
    require_binary_01_labels(y_all, label_name=cfg.label_col)

    # ---- Build graph ----
    
    try:
        edge_index = load_graph_to_edge_index(cfg.graph_pkl, X_all.index)
    except ValueError:
        logging.exception("Graph/feature validation failed — cannot proceed.")
        sys.exit(4)
    logging.info(f"edge_index shape: {tuple(edge_index.shape)}")
    edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float)

    # ---- Cross-validation strategy ----
    
    cv = RepeatedStratifiedKFold(n_splits=cfg.cv_splits, n_repeats=cfg.cv_repeats, random_state=cfg.seed)

    # ---- Optuna study (persisted by default unless --in_memory_optuna) ----
    
    logs_dir_folds = (cfg.output_dir / "logs" / "logs_train").resolve() # fold logs
    logs_dir_folds.mkdir(parents=True, exist_ok=True)


    
    
    if cfg.optuna_db:# --- MODIFIED STORAGE INIT FOR LOCK PREVENTION ---
        
        # Increase timeout to 60s to prevent 'database is locked' errors on GPU
        
        storage = RDBStorage(
            url=cfg.optuna_db,
            engine_kwargs={"connect_args": {"timeout": 60}}
        )
    else:
        storage = None
    
    
    
    study = run_optimisation(
        n_trials=cfg.n_trials,
        X=X_all, y=y_all, cv=cv,
        nodes_train=nodes_train, nodes_test=nodes_test,
        Network=edge_index, edge_weights=edge_weights,
        num_epochs=cfg.epochs,
        study_name=cfg.study_name,
        storage=storage,
        device=device,
        log_root=logs_dir_folds,
    )

    # Log best results from study
    
    logging.info(f"Best AUC: {study.best_value:.4f}")
    logging.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # ---- Final fit on full training set using best params; evaluate held-out ----
    
    X_tr = X_all.loc[X_all.index.isin(nodes_train), :]
    y_tr = y_all.loc[y_all.index.isin(nodes_train)]
    splits = list(cv.split(X_tr, y_tr))

    p = study.best_params
    neighbors = [int(p["n_b"])] * int(p["num_layers"])

    models_dir = cfg.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build args for each fold
    
    fold_args = [
        dict(
            xx=X_all,
            yy=y_all,
            nodes_train=nodes_train,
            nodes_heldout=nodes_test,
            train_idx=tr_idx,
            test_idx=va_idx,
            Network=edge_index,
            edge_weights=edge_weights,
            n_features=X_all.shape[1],
            n_classes=1,#(for single logits head output)
            n_hidden=p["n_hidden"],
            num_layers=p["num_layers"],
            num_layers_pre=p["num_layers_pre"],
            num_layers_post=p["num_layers_post"],
            dropout=p["dropout"],
            rate=p["rate"],
            eps=p["eps"],
            train_eps=p["train_eps"],
            lr=p["lr"],
            weight_decay=p["weight_decay"],
            batch_size=p["batch_size"],
            neighbors=neighbors,
            epochs=cfg.epochs,
            device=device,
            fold_id=k,
            save_model_path=models_dir / f"Best_GINBiDir_train_foldID_{k}{safe_study_filename(cfg.study_name)}.pth",
            seed=0 + k,
            return_sets=("train", "test", "heldout"),
            return_model=False,
            log_root=logs_dir_folds,
        )
        for k, (tr_idx, va_idx) in enumerate(splits)
    ]
    
    if device.type == "cuda":
        # GPU: run folds sequentially in this process
        fold_results = [run_fold(**kwargs) for kwargs in fold_args]
    else:
        # CPU: parallelise over folds
        fold_results = Parallel(n_jobs=-1, backend="loky")(delayed(run_fold)(**kwargs) for kwargs in fold_args)
        
    # --- Save predictions
    
    predictions_dir = cfg.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = predictions_dir / f"predictions_{safe_study_filename(cfg.study_name)}.csv"

    frames = []
    for res in fold_results:
        if "df_train" in res:   frames.append(res["df_train"])
        if "df_test" in res:    frames.append(res["df_test"])
        if "df_heldout" in res: frames.append(res["df_heldout"])
    out_pred = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=["Gene","Prob","Target","Fold_ID","Set"])
    out_pred.to_csv(pred_path, index=False)
    logging.info(f"Saved predictions to: {pred_path}")

    logging.info("Training and prediction complete.")

if __name__ == "__main__":
    cfg = parse_args()
    try:
        main(cfg)
    except optuna.TrialPruned:
        raise
    except KeyboardInterrupt:
        logging.warning("Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
    except FileNotFoundError as e:
        logging.exception(f"Missing file: {getattr(e, 'filename', str(e))}")
        sys.exit(2)
    except Exception as e:
        if isinstance(e, sqlite3.OperationalError) or (getattr(e, "orig", None) is not None and isinstance(e.orig, sqlite3.OperationalError)):
            logging.exception("Optuna storage error (e.g., database locked). Use a different DB file, fewer parallel workers, or --in_memory_optuna / a server DB.")
            sys.exit(3)
        logging.exception("Uncaught error in main")
        sys.exit(1)
