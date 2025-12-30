#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GIN_PermImpGiG_local_MIDAS_Demo_CPU.py
===========================================

Pathway-restricted, degree-preserving permutation importance for a trained GIN model.

Summary
-------

For each CV fold model from a prior training pipeline, it:
    
- Reconstructs the per-fold train/validation split and the held-out set.
- For each gene of interest (GOI), gathers the pathway(s) it belongs to from a
   user-supplied gene–pathway mapping (CSV,parameter --gene_set_csv).
- For each (GOI, pathway):
     - Identifies the subgraph induced by all genes in that pathway,
     - Uses 'xswap' to rewire only the edges inside that pathway while
       preserving in/out degree within the subgraph,
     - Keeps all other edges ('static edges') unchanged,
     - Runs inference and records the GOI’s permuted probability alongside its
       baseline probability.
- Writes one CSV per GOI containing long-form rows across permutations and pathways.

What this script does NOT do
----------------------------
- It does not train or fine-tune models.
- It does not create the default Optuna DB directory if '--optuna_db' is omitted.
- If you rely on the default SQLite path, the file must already exist.

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
    Root output directory. Subfolders for 'predictions/',
     and 'logs/' are created as needed.

--study_name : str, optional (default: "GINBiDir_Optuna_MIDAS_demo")
    Human-readable study identifier used to name artifacts (DB/predictions/models/logs).
    

--optuna_db str
    SQLAlchemy storage URI for the existing Optuna study (e.g.,
    sqlite:////path/to/optuna_runs/<study>.optuna.db). This script assumes the
    study already exists. If omitted, a canonical path under
    <output_dir>/optuna_runs/<safe(study_name)>.optuna.db is assumed and must
    already exist (created by the training pipeline).

--seed : int, optional (default: 42)
    Global seed for Python, NumPy, and PyTorch.

--cv_splits : int, (default: 10)
    Number of stratified folds for cross-validation (has to be consistent with training parameters).

--cv_repeats : int, optional (default: 10)
    Number of repetitions for RepeatedStratifiedKFold (has to be consistent with training parameters).

--label_col : str, optional (default: 'targ_stat')
    Name of the binary target column in the CSVs (case-insensitive match).


--device : {'cpu', 'gpu'}, optional (default: 'cpu')
    Device selection. This script is CPU-only; choosing 'gpu' exits with code 90.

--seeds str
    Rewiring seeds, either a range "a:b" (inclusive) or a comma list "s1,s2,…".
    Example: "0:49" (default).
    
---gene_set_csv str
    CSV with columns: gene_expl,gene,PathwayID

Outputs
-------

Per-gene CSV files in:
  <output_dir>/interpretability/permEdgeLocalImp/
Each row corresponds to one permutation seed for one pathway for that GOI, with columns like:
  gene, CV_fold, perm_type ("edge_only"), pathway_id, pathway_name (if present),
  permutation, orig_pred, perm_pred

Assumptions and prerequisites
---------------------------
- The Optuna study and per-fold model exist (this script does not
  create the study or train models).  
- Train/test CSVs share the same feature schema; gene IDs are unique.  
- The pickled graph contains only nodes present in features after alignment.  
- CPU-only execution (NeighborLoader + GIN) is expected.


Example (On Mac):
-------
python scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py \
  --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv \
  --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv \
  --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl \
  --study_name MIDAS_GIN_demo \
  --label_col targ_stat \
  --device cpu \
  --gene_set_csv data/demo/GeneSet_Demo_PathwaysNEW.csv \
  --seeds 0:49 --cv_splits 5 --cv_repeats 1


Example (On Windows):
-------
python scripts/interpretability/GIN_PermImpGiG_Local_MIDAS_Demo_CPU.py^
 --train_csv data/demo/demo_MIDAS_InputMatrix_train.csv^
 --test_csv  data/demo/demo_MIDAS_InputMatrix_test.csv^
 --graph_pkl data/demo/demo_MIDAS_InputMatrix_Graph.pkl^
 --study_name MIDAS_GIN_demo^
 --label_col targ_stat^
 --device cpu^
 --gene_set_csv data/demo/GeneSet_Demo_PathwaysNEW.csv^
 --seeds 0:49 --cv_splits 5 --cv_repeats 1
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
from typing import Dict, Hashable, Iterable, List, Optional, Tuple
from joblib import Parallel, delayed
import re
from urllib.parse import urlparse
import os
import importlib, types

# ── Third-party imports ───────────────────────────────────────────────────────

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import xswap 

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
# This assumes this file lives in: <project_root>/scripts/interpretability/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the TRAINING script writes its outputs by default
TRAIN_OUTPUT_ROOT = PROJECT_ROOT / "results" / "train"

# Where THIS script should write its interpretability outputs by default
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "interpretability"


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


    seed: int = 42  # seed for Python, NumPy, PyTorch


    cv_splits: int = 10
    cv_repeats: int = 10

    label_col: str = "targ_stat"

    device_choice: str = "cpu"  # 'cpu' (gpu is disallowed in this script)
    
    # NEW: needed by permutation-importance
    
    gene_set_csv: Optional[Path] = None       # CSV with columns: gene,PathwayID
    seeds: str = "0:49"                       # range "start:end" (inclusive) or comma-list "0,1,2"
   
    def __post_init__(self):
        
        # Ensure output dir for interpretability exists
        
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        # If no explicit optuna_db was given, assume it lives under
        # the TRAINING output root: results/train/optuna_runs/<study>.optuna.db
        if self.optuna_db is None:
            train_runs_dir = TRAIN_OUTPUT_ROOT / "optuna_runs"
            db_file = train_runs_dir / f"{safe_study_filename(self.study_name)}.optuna.db"
            self.optuna_db = sqlite_uri_from_path(db_file)
        else:
            # If user passed a URI/path, normalise it
            db_path = sqlite_file_from_uri(self.optuna_db)
            if db_path is not None:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                


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

def get_device(device_choice: str = "cpu") -> torch.device:
    """Resolve the device to use, enforcing CPU-only policy."""
    if device_choice.lower() == "gpu":
        print("This script is CPU-only. Please run with --device cpu (or omit the flag). Exiting.")
        sys.exit(90)
    if torch.cuda.is_available():
        logging.info("CUDA detected, but this is a CPU-only script. Forcing CPU.")
    logging.info("Using CPU")
    return torch.device("cpu")

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
# Inference helpers
# =============================================================================


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


def get_predictions(
    *,
    data: Data,
    model: nn.Module,
    device: torch.device,
    neighbors: List[int],
    mask_name: str,
    fold_id: int,
    set_label: str,
    node_names: pd.Index,
) -> pd.DataFrame:
    """
    Run model inference over ALL mini-batches selected by 'mask_name' and
    return a tidy DataFrame with columns: Gene, Prob, Target, Fold_ID, Set.
    """
    if not hasattr(data, mask_name):
        raise AttributeError(f"'data' has no attribute '{mask_name}'")

    sel_mask = getattr(data, mask_name).bool()
    bs = int(sel_mask.sum().item()) or 1  # fetch selected nodes in one pass

    loader = NeighborLoader(
        data,
        num_neighbors=neighbors,
        batch_size=bs,
        input_nodes=sel_mask,
    )

    y, yhat, names = predict_proba(
        model=model,
        loader=loader,
        device=device,
        mask_name=mask_name,
        node_names=node_names,
    )

    df = pd.DataFrame(
        {
            "Gene": np.asarray(names),
            "Prob": yhat.astype(float),
            "Target": y.astype(float),
            "Fold_ID": int(fold_id),
            "Set": set_label,
        }
    )
    logging.debug(
        f"[fold {fold_id}] {set_label}: produced {len(df)} predictions (mask={mask_name})."
    )
    return df


# =============================================================================
# Permutation runner (pathway rewiring)
# =============================================================================


def calc_perm_edge_imp(
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_id: int,
    xx: pd.DataFrame,
    yy: pd.Series,
    nodes_train: pd.Index,
    nodes_test: pd.Index,
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
    Network: torch.Tensor,
    neighbors: List[int],
    device: torch.device,
    # New: explicit inputs (no hidden globals)
    model_ckpt_path: Optional[Path] = None,
    save_model_path: Optional[Path] = None,   # kept for backward-compat; used if model_ckpt_path is None
    edge_list_all: Optional[pd.DataFrame] = None,
    gene_set: Optional[pd.DataFrame] = None,  # must contain columns: ['gene','PathwayID']
    genes_of_interest: Optional[Iterable[str]] = None,
    seeds: Iterable[int] = (0,),              # seeds for xswap permutations
    out_dir: Optional[Path] = None,           # where to write CSVs
    log_root: Optional[Path] = None,
) -> str:
    
    """
    Degree-preserving, pathway-restricted edge permutation importance for a fold model.

    This function:
      - Rebuilds per-fold masks (undersampled train, validation/test, held-out).
      - Loads the saved fold models.
      - Computes baseline predictions for test + held-out masks.
      - For each gene and its pathways, rewires ONLY edges within that pathway
         (degree-preserving via 'xswap'') and measures delta in that gene's predicted probability.
      - Writes one CSV per gene with long-form records across permutations.

    """
    
    # --- Per-worker logging bootstrap ---
    try:
        logs_dir = (log_root or (Path.cwd() / "logs" / "perm_local")).resolve()
        # setup_logging will mkdir(parents=True) for the file's parent, but this
        # ensures the directory exists even before we compute the file path.
        logs_dir.mkdir(parents=True, exist_ok=True)

        # One file per fold/process; setup_logging will append _<pid> automatically
        log_file = logs_dir / f"perm_local_fold{fold_id}.log"
        setup_logging(verbosity="INFO", log_file=log_file)

        logging.info("Worker logging ready | fold=%s | pid=%s", fold_id, os.getpid())
    except Exception as e:
        print(f"[calc_perm_edge_imp] logging init failed for fold={fold_id}: {e}", file=sys.stderr)
    # -------------------------------
    # Resolve model checkpoint path
    # -------------------------------
    
    ckpt_path = model_ckpt_path or save_model_path
    if ckpt_path is None:
        raise ValueError("Please provide 'model_ckpt_path' (or 'save_model_path') for the fold checkpoint.")
    ckpt_path = Path(ckpt_path)

    # -------------------------------
    # Build fold splits and masks
    # -------------------------------
    
    X_truniv = xx.loc[xx.index.isin(nodes_train), :]
    y_truniv = yy.loc[yy.index.isin(nodes_train)]
    X_train, y_train = X_truniv.iloc[train_idx, :], y_truniv.iloc[train_idx]
    X_val = X_truniv.iloc[test_idx, :]

    # Undersample positives/negatives on the TRAIN partition only, keep identities
    rus = RandomUnderSampler(random_state=42)
    Xtr_df = X_train.reset_index().rename(columns={"index": "_row_id_"})
    ytr_s  = y_train.reset_index(drop=True)
    X_res_df, _ = rus.fit_resample(Xtr_df, ytr_s)
    kept_train_ids = set(X_res_df["_row_id_"].astype(str))

    # Build PyG Data for the *original* graph (baseline)
    
    Xgraph = torch.tensor(xx.values, dtype=torch.float)
    ygraph = torch.tensor(yy.values, dtype=torch.float)

    data_base = Data(
        x=Xgraph,
        edge_index=Network,  
        y=ygraph,
    )

    idx_all = pd.Index(xx.index)
    data_base.train_mask = torch.as_tensor(idx_all.isin(pd.Index(kept_train_ids)), dtype=torch.bool)
    data_base.test_mask  = torch.as_tensor(idx_all.isin(pd.Index(X_val.index)), dtype=torch.bool)
    data_base.val_mask   = torch.as_tensor(idx_all.isin(pd.Index(nodes_test)), dtype=torch.bool)

    if int(data_base.test_mask.sum()) == 0:
        logging.error(f"[fold {fold_id}] No validation/test nodes in this fold. Aborting.")
        return "Done"

    # -------------------------------
    # Load fold model
    # -------------------------------
    
    model = GNN(
        n_features=n_features,
        n_classes=n_classes,
        n_hidden=int(n_hidden),
        num_layers=int(num_layers),
        num_layers_pre=int(num_layers_pre),
        num_layers_post=int(num_layers_post),
        dropout=float(dropout),
        rate=float(rate),
        eps=float(eps),
        train_eps=bool(train_eps),
    ).to(device)

    logging.info(f"[fold {fold_id}] Loading checkpoint: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -------------------------------
    # Baseline predictions (original graph)
    # -------------------------------
    
    df_base_test = get_predictions(
        data=data_base,
        model=model,
        device=device,
        neighbors=neighbors,
        mask_name="test_mask",
        fold_id=fold_id,
        set_label="Test fold (baseline)",
        node_names=xx.index,
    )

    df_base_held = get_predictions(
        data=data_base,
        model=model,
        device=device,
        neighbors=neighbors,
        mask_name="val_mask",
        fold_id=fold_id,
        set_label="Held out (baseline)",
        node_names=xx.index,
    )

    base_map = (
        pd.concat([df_base_test[["Gene", "Prob"]], df_base_held[["Gene", "Prob"]]], axis=0)
        .drop_duplicates(subset=["Gene"])
        .set_index("Gene")["Prob"]
    )

    # -------------------------------
    # Genes of interest and edge list
    # -------------------------------
    
    # Infer default genes of interest: those present in test or held-out
    
    if genes_of_interest is None:
        goi_candidates = set(df_base_test["Gene"]).union(set(df_base_held["Gene"]))
        genes_of_interest = sorted(goi_candidates)

    # Build an edge list with gene names if not provided:
        
    if edge_list_all is None:
        ei = Network.cpu().numpy()
        edge_list_all = pd.DataFrame(
            {
                "source_name": idx_all[np.asarray(ei[0], dtype=np.int64)],
                "target_name": idx_all[np.asarray(ei[1], dtype=np.int64)],
            }
        )

    # Validate gene_set:
        
    if gene_set is None or not {"gene", "PathwayID"}.issubset(gene_set.columns):
        raise ValueError(
            "Please provide 'gene_set' with columns ['gene','PathwayID']."
        )

    # Output dir
    
    
    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = DEFAULT_OUTPUT_DIR / "permEdgeLocalImp"

    out_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute the full network (directed edges) as a DataFrame of indices
    
    full_df = pd.DataFrame(Network.cpu().numpy().T, columns=["src", "dst"])

    # Map gene -> integer index for quick lookup
    
    name_to_idx = {name: i for i, name in enumerate(idx_all)}

    # -------------------------------
    # Permutation loop
    # -------------------------------
    for gi, curr_gene in enumerate(genes_of_interest, start=1):
        
        if curr_gene not in base_map.index:
            logging.info(f"[fold {fold_id}] Gene '{curr_gene}' not scored in baseline; skipping.")
            continue
        
        
        # Extract pathways pertaining to the current gene
        
        rel_paths = gene_set.loc[gene_set["gene_expl"] == curr_gene]
        
        unique_paths = rel_paths["PathwayID"].unique().tolist()
        
        if not unique_paths:
            logging.info(f"[fold {fold_id}] Gene '{curr_gene}' has no pathways in gene_set; skipping.")
            continue

        logging.info(f"[fold {fold_id}] Gene {gi}/{len(genes_of_interest)}: {curr_gene} — {len(unique_paths)} pathway(s).")

        perm_imps = []

        for path_counter, curr_path in enumerate(unique_paths, start=1):
            
            # All member genes for this pathway
            
            rel_path_genes = rel_paths.loc[rel_paths["PathwayID"] == curr_path, "gene"].to_list()
            
            
            # Keep only genes present in the graph/index
            rel_path_genes = pd.Index(rel_path_genes).intersection(pd.Index(idx_all)).tolist()


            # Edges inside the pathway (by name), then convert to indices
            
            sub_edges_named = edge_list_all[
                edge_list_all["source_name"].isin(rel_path_genes)
                & edge_list_all["target_name"].isin(rel_path_genes)
            ]

            if sub_edges_named.empty:
                logging.debug(f"[fold {fold_id}] {curr_gene} | pathway={curr_path}: no edges; recording NA.")
                perm_imps.append(
                    {
                        "gene": curr_gene,
                        "CV_fold": fold_id,
                        "perm_type": "edge_only",
                        "pathway_id": curr_path,
                        "pathway_name": rel_paths.loc[
                            rel_paths["PathwayID"] == curr_path, "PathwayID"
                        ].unique()[0],
                        "permutation": np.nan,
                        "orig_pred": float(base_map.get(curr_gene, np.nan)),
                        "perm_pred": np.nan,
                    }
                )
                continue

            src_idx = sub_edges_named["source_name"].map(name_to_idx).to_numpy(dtype=np.int64)
            dst_idx = sub_edges_named["target_name"].map(name_to_idx).to_numpy(dtype=np.int64)
            sub_df = pd.DataFrame({"src": src_idx, "dst": dst_idx})

            # Include both directions for deletion and re-addition
            
            sub_bidir = pd.concat(
                [sub_df, sub_df.rename(columns={"src": "dst", "dst": "src"})],
                ignore_index=True,
            ).drop_duplicates(ignore_index=True)

            # Static edges = full network minus the subnetwork we will permute
            merged = full_df.merge(sub_bidir, on=["src", "dst"], how="left", indicator=True)
            static_edges = merged.loc[merged["_merge"] == "left_only", ["src", "dst"]].reset_index(drop=True)

            # Run permutations
            for seed in seeds:
                
                # Degree-preserving permutation within the subnetwork
                
                # Xswap expects an edge list of tuples
                sub_edge_list = list(map(tuple, sub_bidir[["src", "dst"]].to_numpy()))
                perm_edges, _perm_stats = xswap.permute_edge_list(
                    edge_list=sub_edge_list,
                    seed=int(seed),
                    allow_self_loops=False,
                    allow_antiparallel=False,
                    multiplier=100,
                )

                perm_df = pd.DataFrame(perm_edges, columns=["src", "dst"])
                
                # Keep bidirectional symmetry in the permuted subgraph as well
                perm_bidir = pd.concat(
                    [perm_df, perm_df.rename(columns={"src": "dst", "dst": "src"})],
                    ignore_index=True,
                )

                # Combine with static edges and build new edge_index
                new_edges = pd.concat([static_edges, perm_bidir], ignore_index=True)
                perm_edge_index = torch.tensor(new_edges.to_numpy().T, dtype=torch.long)

                # Data object with same features/masks, but permuted edges
                data_perm = Data(
                    x=data_base.x,
                    edge_index=perm_edge_index,
                    y=data_base.y,
                    train_mask=data_base.train_mask,
                    test_mask=data_base.test_mask,
                    val_mask=data_base.val_mask,
                )

                # Predictions on permuted graph
                df_test_p = get_predictions(
                    data=data_perm,
                    model=model,
                    device=device,
                    neighbors=neighbors,
                    mask_name="test_mask",
                    fold_id=fold_id,
                    set_label="Test (xswap)",
                    node_names=xx.index,
                )

                df_held_p = get_predictions(
                    data=data_perm,
                    model=model,
                    device=device,
                    neighbors=neighbors,
                    mask_name="val_mask",
                    fold_id=fold_id,
                    set_label="Held-out (xswap)",
                    node_names=xx.index,
                )

                df_perm_all = pd.concat([df_test_p, df_held_p], axis=0, ignore_index=True)

                if curr_gene in df_perm_all["Gene"].values:
                    perm_prob = float(df_perm_all.loc[df_perm_all["Gene"] == curr_gene, "Prob"].iloc[0])
                else:
                    perm_prob = np.nan

                perm_imps.append(
                    {
                        "gene": curr_gene,
                        "CV_fold": fold_id,
                        "perm_type": "edge_only",
                        "pathway_id": curr_path,
                        "pathway_name": rel_paths.loc[
                            rel_paths["PathwayID"] == curr_path, "PathwayID"
                        ].unique()[0],
                        "permutation": int(seed),
                        "orig_pred": float(base_map.get(curr_gene, np.nan)),
                        "perm_pred": perm_prob,
                    }
                )

            logging.info(
                f"[fold {fold_id}] Gene {curr_gene}: finished pathway {path_counter}/{len(unique_paths)} "
                f"(PathwayID={curr_path}) with {len(list(seeds))} permutation(s)."
            )

        # Write one CSV per gene
        if perm_imps:
            df_out = pd.DataFrame(perm_imps)
            seed_span = f"{min(seeds)}-{max(seeds)}" if len(list(seeds)) > 1 else f"{list(seeds)[0]}"
            out_path = out_dir / f"GINBiDir_KNN{neighbors[0]}_{curr_gene}_foldID{fold_id}_permEdgeScores_{seed_span}.csv"
            df_out.to_csv(out_path, index=False)
            logging.info(f"[fold {fold_id}] Wrote permutation importance for '{curr_gene}' → {out_path}")

    return "Done"



# =============================================================================
# CLI
# =============================================================================
def parse_args() -> Config:
    
    """Parse command-line options and construct a 'Config'."""
    parser = argparse.ArgumentParser(
        description="Permutation edge-importance (CPU-only) using trained GIN fold models."
    )
    parser.add_argument("--train_csv", type=Path, required=True, help="Path to train CSV (first column = gene ID).")
    parser.add_argument("--test_csv",  type=Path, required=True, help="Path to test CSV (first column = gene ID).")
    parser.add_argument("--graph_pkl", type=Path, required=True, help="Path to graph PKL (edge list DataFrame).")

    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for outputs/artefacts.")
    parser.add_argument("--study_name", type=str, default=Config.study_name, help="Optuna study name.")
    parser.add_argument("--optuna_db", type=str, default=None, help="Optuna storage URI. Default: <output_dir>/<study_name>.optuna.db")

    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed.")
    parser.add_argument("--cv_splits", type=int, default=Config.cv_splits, help="CV splits.")
    parser.add_argument("--cv_repeats", type=int, default=Config.cv_repeats, help="CV repeats.")
    parser.add_argument("--label_col", type=str, default=Config.label_col, help="Label column name (default: targ_stat).")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                        help="Device selection. 'gpu' is not supported in this CPU-only script.")

    # NEW: permutation-importance specifics
    
    parser.add_argument("--gene_set_csv", type=Path, required=True,
                        help="CSV with columns: gene,PathwayID")
   
    parser.add_argument("--seeds", type=str, default="0:49",
                        help="Seed range 'start:end' (inclusive) or comma-list '0,1,2'.")
    
    args = parser.parse_args()
   

    return Config(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        graph_pkl=args.graph_pkl,
        output_dir=args.output_dir,
        study_name=args.study_name,
        optuna_db=args.optuna_db,
        seed=args.seed,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        label_col=args.label_col,
        device_choice=args.device,
        gene_set_csv=args.gene_set_csv,
        seeds=args.seeds
    )



# =============================================================================
# MAIN
# =============================================================================
# --- Helper functions

def load_best_params_from_optuna(study_name: str, storage: str) -> Dict[str, object]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    logging.info(f"Loaded Optuna study '{study_name}'. Best value: {study.best_value:.4f}")
    logging.info("Best params:\n" + json.dumps(study.best_params, indent=2))
    p = study.best_params
    # ensure required fields exist
    required = ["n_hidden","num_layers","num_layers_pre","num_layers_post","dropout","rate","eps","train_eps","n_b","batch_size","lr","weight_decay"]
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"Study best_params missing keys: {missing}")
    return p


def parse_seeds_arg(s: str) -> List[int]:
    s = str(s).strip()
    if ":" in s:
        a, b = s.split(":", 1)
        a, b = int(a), int(b)
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    # comma/space-separated list
    return [int(x) for x in re.split(r"[,\s]+", s) if x != ""]


def main(cfg: Config) -> None:
    
    # --- Limit thread oversubscription when using joblib + torch
    
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # --- Setup logging ---
    
    logs_dir = cfg.output_dir / "logs"
    log_path = logs_dir / f"{safe_study_filename(cfg.study_name)}.log"
    setup_logging(verbosity="INFO", log_file=log_path)

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
    require_binary_01_labels(y_all, label_name=cfg.label_col)

    # ---- Build graph ----
    
    try:
        edge_index = load_graph_to_edge_index(cfg.graph_pkl, X_all.index)
    except ValueError:
        logging.exception("Graph/feature validation failed — cannot proceed.")
        sys.exit(4)
    logging.info(f"edge_index shape: {tuple(edge_index.shape)}")

    # ---- CV (same scheme as training so folds align) ----
    
    cv = RepeatedStratifiedKFold(n_splits=cfg.cv_splits, n_repeats=cfg.cv_repeats, random_state=cfg.seed)

    # --- Load best hyperparameters from Optuna (same study/storage) ---
    
    try:
        params = load_best_params_from_optuna(cfg.study_name, cfg.optuna_db)
    except Exception:
        logging.exception("Failed to load Optuna best_params. Provide a valid --optuna_db/--study_name.")
        sys.exit(12)

    n_features = X_all.shape[1]
    n_classes = 1
    neighbors = [int(params["n_b"])] * int(params["num_layers"]) # Number of neighbors to sample per GNN layer in PyG's NeighborLoader. Effective neighbor list is '[n_b] * num_layers' for each sampled hop.

    # ---- Load gene set + GOI + seeds ----
    
    gene_set_df = pd.read_csv(cfg.gene_set_csv)
    
    required_cols = {"gene_expl","gene", "PathwayID"}
    
    if not required_cols.issubset(gene_set_df.columns):
        
        raise KeyError(f"--gene_set_csv must have columns {sorted(required_cols)}; found {sorted(gene_set_df.columns)}")
        
    genes_of_interest = gene_set_df["gene_expl"].unique().tolist()

    seeds_list = parse_seeds_arg(cfg.seeds)
    
    logging.info(f"Permutation seeds: {seeds_list[0]}..{seeds_list[-1]} (n={len(seeds_list)})" if len(seeds_list) > 1 else f"Permutation seed: {seeds_list[0]}")

    # ---- Models dir ----
    
    models_dir = TRAIN_OUTPUT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Using models_dir: {models_dir}")

    # ---- Build fold arguments (FIXED: pass only what calc_perm_edge_imp expects) ----
    
    X_tr = X_all.loc[X_all.index.isin(nodes_train), :]
    y_tr = y_all.loc[y_all.index.isin(nodes_train)]
    splits = list(cv.split(X_tr, y_tr))
    
    logs_dir = (cfg.output_dir / "logs" / "perm_local").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)


    fold_args = [
        dict(
            train_idx=tr_idx,
            test_idx=va_idx,
            fold_id=k,
            xx=X_all,
            yy=y_all,
            nodes_train=nodes_train,
            nodes_test=nodes_test,
            n_features=n_features,
            n_classes=n_classes,
            n_hidden=params["n_hidden"],
            num_layers=params["num_layers"],
            num_layers_pre=params["num_layers_pre"],
            num_layers_post=params["num_layers_post"],
            dropout=params["dropout"],
            rate=params["rate"],
            eps=params["eps"],
            train_eps=params["train_eps"],
            Network=edge_index,
            neighbors=neighbors,
            device=device,
            model_ckpt_path=models_dir / f"Best_GINBiDir_train_foldID_{k}{safe_study_filename(cfg.study_name)}.pth",
            edge_list_all=None,                 # optional; will be reconstructed if None
            gene_set=gene_set_df,
            genes_of_interest=genes_of_interest,
            seeds=seeds_list,
            out_dir=cfg.output_dir / "permEdgeLocalImp",
            log_root=logs_dir,
        )
        for k, (tr_idx, va_idx) in enumerate(splits)
    ]

    logging.info(f"Launching permutation-importance across {len(fold_args)} CV folds …")
    
    results=Parallel(n_jobs=-1, backend="loky")(delayed(calc_perm_edge_imp)(**kwargs) for kwargs in fold_args)
    
    logging.info("All permutation-importance jobs finished.")


if __name__ == "__main__":
    cfg = parse_args()
    try:
        main(cfg)
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
    finally:
        logging.shutdown()