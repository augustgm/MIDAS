# MIDAS Project: Feature Processing, GNN Modelling, and Figure Generation

This repository contains the full MIDAS analysis stack, organised into three main components:

- **`FeatureProcessing/`**: scripts for building and aggregating gene-level features from multiple data sources (single-cell interactomes, GWAS, HLA peptidomics, bulk sequencing).
- **`GNN/`**: graph neural network (GIN) training, hyperparameter optimisation, and permutation-based interpretability on MIDAS-style input matrices.
- **`Figs/`**: code used to generate the figures for the associated manuscript, consuming outputs from `FeatureProcessing/` and `GNN/`.

This makes it clear **where each part of the pipeline lives** and how to reproduce the key steps end-to-end.


---

## Paper

This code allows for the reproduction of the results published in:
    
https://www.researchsquare.com/article/rs-5499857/v1
    

The demo data frames provided here follow a similar structure to the data used in 
the paper.

> **See the paper's data availability and the methods section.**



---

## 1. Repository structure

At the top level:

```text
.
├─ README.md                # This file (top-level overview)
├─ FeatureProcessing/       # MIDAS feature engineering scripts (R + Python)
├─ GNN/                     # GIN training + interpretability (Python)
└─ Figs/                    # Figure generation code (R)
```

### 1.1 `GNN/` subrepo

The `GNN/` folder is a self-contained subproject for:

- training a GIN classifier on MIDAS-style features,
- running Optuna-based hyperparameter optimisation,
- computing permutation importance for features and edges (including GiG and local/pathway-based variants).

It has its **own README** with full details.

> **For environment setup, training commands, and interpretability workflows, see `GNN/README_GNN.md`.**

---

## 2. Feature processing (`FeatureProcessing/`)

The `FeatureProcessing/` folder contains scripts that prepare or aggregate input features used by the GNN models and downstream analyses. These scripts are **not tied to the demo data** in `GNN/data/demo/` but are provided to make the upstream feature engineering reproducible.

```text
FeatureProcessing/
├─ SCINETinteractomes.R
├─ TopologicalSpecificity.R
├─ GWAScatalog_gSNPassoc.R
├─ EDGE_HLApep.R
└─ BulkSequencingData.py
```

### 2.1 R package versions used in the paper (Reporting Summary)

The feature pre-processing scripts in `FeatureProcessing/` were run with **R ≥ 4.0** (the paper used **R 4.1.3**).  
For exact reproducibility, we follow the **Reporting Summary associated with the paper** for the R package versions used.


**Core utilities & plotting**

- `ggplot2` v3.4.1  
- `ggpubr` v0.6.0  
- `tidyverse` v2.0.0  
- `tidyr` v1.3.0  
- `dplyr` v1.1.0  
- `reshape2` v1.4.1  
- `readxl` v1.4.3  
- `data.table` v1.15.4  
- `magrittr` v2.0.3  

**Statistics / matrices / normalisation**

- `matrixStats` v1.3.0  
- `Matrix.utils` v0.9.8  
- `nortest` v1.0.4  
- `preprocessCore` v1.56.0  
- `limma` v3.50.3  
- `apeglm` v1.16.0  
- `DESeq2` v1.34.0  
- `pROC` v1.18.5  
- `fmsb` v0.7.6  

**Single-cell / SCINET pipeline**

- `Seurat` v4.3.0  
- `SeuratObject` v4.1.3  
- `SingleCellExperiment` v1.16.0  
- `locfit` v1.5-9.6  
- `loomR` v0.2.1.9  
- `ACTIONet` v2.1.9  
- `SCINET` v1.0  

**Graph / network analysis & visualisation**

- `igraph` v1.4.0  
- `ggraph` v2.1.0  
- `visNetwork` v2.1.2  

**Enrichment**

- `WebGestaltR` v0.4.6  

> Note: Some packages (e.g. `DESeq2`, `limma`, `apeglm`, `SingleCellExperiment`, `preprocessCore`) are typically installed via **Bioconductor**.  
> Others (e.g. `ACTIONet`, `SCINET`, `loomR`) are often installed from **GitHub/source** and may require additional system libraries.  
> Please refer to each package’s installation documentation if needed.


### Example installation (convenience)

The commands below are a convenience install and are **not guaranteed** to resolve to the exact versions listed above (package repositories change over time). For exact version pinning, consider using an `renv.lock` generated on the same platform/toolchain as the paper.

```
install.packages(c(
  "ggplot2","ggpubr","tidyverse","tidyr","dplyr","reshape2","readxl",
  "data.table","magrittr","matrixStats","Matrix.utils","nortest",
  "pROC","fmsb","igraph","ggraph","visNetwork","WebGestaltR","locfit"
))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c("preprocessCore","limma","apeglm","DESeq2","SingleCellExperiment"))
```


> Many scripts expect large input files (e.g. single-cell objects, GWAS catalogue CSVs)
> that are not tracked in Git for size/licensing reasons. See each script header for
> required filenames and paths.


### 2.2 `SCINETinteractomes.R`

**Purpose:** construct **cell type-specific interactomes** from single-cell RNA-seq atlases using [ACTIONet](https://github.com/shmohammadi86/ACTIONet/tree/R-release)/[SCINET](https://github.com/shmohammadi86/SCINET/tree/master).

- Reads an integrated single-cell object for a given cell "supertype" (e.g. `DCs`, `MFs`, `Bcells`, `Tcells`).
- Converts to a `SingleCellExperiment`.
- Runs dimensionality reduction with ACTIONet.
- Optionally performs batch correction (Harmony) across studies.
- Computes gene specificity per cell type.
- Builds cell type-specific networks via SCINET and saves them as `.rds` objects under:

```text
FeatureProcessing/scinet_interactomes/<SUPER_TYPE>/<SUPER_TYPE>_celltypeSpecificNetworks_<assay>_<slot>_<batchCorrFlag>.rds
```

You typically invoke this script from the command line, e.g.:

```bash
Rscript SCINETinteractomes.R DCs
Rscript SCINETinteractomes.R Bcells
# etc.
```

where the argument selects which supertype object to load and process.

### 2.3 `TopologicalSpecificity.R`

**Purpose:** extract **topological specificity scores** for each gene in each cell type-specific interactome.

- Iterates over the `.rds` interactome files created by `SCINETinteractomes.R`.
- For each cell-type-specific network:
  - extracts node attributes (e.g. `name`, `specificity`),
  - writes a CSV with columns:
    - `gene` (node name),
    - `topo_spec` (topological specificity).

Outputs are written by default to:

```text
FeatureProcessing/scinet_interactomes/<CELLTYPE>_topospec_<assay>_<slot>_<batchCorrFlag>.csv
```

These CSVs can then be consumed elsewhere (e.g. by `BulkSequencingData.py` in the GNN feature pipeline).

### 2.4 `GWAScatalog_gSNPassoc.R`

**Purpose:** derive gene-level features from the **[GWAS Catalogue](https://www.ebi.ac.uk/gwas/)**, focusing on immune, autoimmune, rheumatic, and related phenotypes.

High-level steps:

1. Load the full GWAS catalogue (e.g. `gwas_catalog_v1.0-associations_*.csv`).
2. Define sets of disease traits:
   - autoimmune/rheumatic/allergic conditions,
   - immune-related quantitative phenotypes (cytokines, blood counts),
   - immune modulatory responses (e.g. drug response).
3. Restrict to manually curated phenotype sets using small helper CSVs
   (e.g. `autoImmune_allergic_rheumatic_phenotypes.csv`,
   `immune_phenotypes_gwas_catalog_v2.csv`, `immune_modulatory_response_gwas_catalog.csv`).
4. Filter to genome-wide significant SNPs (e.g. `P < 5e-8`) and non-intergenic entries.
5. Compute **per-gene SNP counts** for different phenotype groups and export CSVs, e.g.:

```text
autoimmRheumAllergy_snpCounts_perGWAScatalogEntrezGene.csv
bloodCounts_snpCounts_perGWAScatalogEntrezGene.csv
cytokineLevel_snpCounts_perGWAScatalogEntrezGene.csv
```

These outputs provide GWAS-derived features that can be mapped onto the gene-level GNN nodes.

### 2.5 `EDGE_HLApep.R`

**Purpose:** explore the association between **HLA peptidomics and gene expression**, producing gene-level correlation features.

Main source: https://www.nature.com/articles/nbt.4313#Abs2 

Main steps:

- Load:
  - mass-spectrometry-based peptide presentation data,
  - sample-level metadata (including HLA types and peptide counts),
  - RNA expression data (mapped to HGNC symbols / Ensembl IDs).
- Summarise transcript-level counts to gene level.
- Merge with sample metadata (including number of presented peptides for different thresholds).
- For each gene, compute correlations between:
  - gene expression and
  - the number of detected peptides (e.g. `peptides_q01`, `peptides_q05`, `peptides_q10`).

Outputs a CSV, for example:

```text
corr_numPresPeptides_summedTranscriptGeneExp_noImputation.csv
```

which can be used as a feature layer capturing HLA presentation correlation.

### 2.6 `BulkSequencingData.py`

**Purpose:** aggregate bulk sequencing data (mutations, expression, etc.) into **GNN-ready node features**, optionally stratified by response status.

Key points:

- Uses `pandas`, `numpy`, `pickle`, and `os`; can be run in the same Python environment as the `GNN/` code (e.g. `midas-gin-cpu`).
- Contains functions that:
  - read bulk mutation/expression data from CSV files,
  - map raw gene IDs to HUGO-approved symbols via a mapping file,
  - aggregate features by clinical response groups (e.g. R vs NR),
  - optionally merge with topological specificity CSVs or other feature layers.

You will typically:

1. Edit file paths at the top of your own driver script or notebook.
2. Import functions from `BulkSequencingData.py`.
3. Save the resulting **gene × feature** matrices to CSV or pickle.
4. Point the `GNN/` training script at those matrices via `--train_csv` / `--test_csv`.

> Note: inputs for `BulkSequencingData.py` are not provided here, but the script shows the expected column structure and file naming conventions.

---

## 3. Figure generation (`Figs/`)

The `Figs/` folder contains scripts used to generate the figures for the associated manuscript. 

A typical workflow for reproducing figures is described below.

---

## 4. Recommended workflow (end-to-end)

A high-level sequence for full reproducibility is:

1. **Feature engineering on real data**
   - Use scripts in `FeatureProcessing/` to process:
     - scRNA-seq atlases (SCINET + topological specificity),
     - GWAS catalogue,
     - HLA peptidomics,
     - bulk sequencing data.
   - Export gene-level feature matrices and store them in a convenient location.

2. **GNN training and interpretability**
   - Set up the environment(s) as described in `GNN/README.md`.
   - Copy or symlink your feature matrices into `GNN/data/` (or pass absolute paths).
   - Run `DevelopInferMIDAS-GIN_Demo_CPUandGPU_logsfold.py` to:
     - perform Optuna search,
     - train GIN models,
     - save models, Optuna DB, predictions.
   - Run the interpretability scripts to compute permutation feature/edge importance.

3. **Figure generation**
   - Use the scripts/notebooks in `Figs/` to reproduce manuscript figures
     from the outputs of steps 1–2.

---

## 5. Reproducibility notes

- All training and interpretability scripts log their configuration (including seeds, paths, CV settings) to JSON in `GNN/results/train/logs/`.
- Cross-validation folds are reproducible as long as:
  - the same random seed,
  - the same data row order,
  - and the same scikit-learn version is used.
- Feature-processing scripts depend on external large files (single-cell objects, GWAS catalogues, etc.) that must be obtained separately; the scripts document the expected filenames.

---

If you are primarily interested in **GNN training and interpretability**, you can focus on the `GNN/` folder.
If you want to regenerate the **full feature stack** feeding into the model, start with `FeatureProcessing/`.
