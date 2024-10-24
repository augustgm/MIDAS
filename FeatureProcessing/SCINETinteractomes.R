#################################################
# Run SCINET on scRNAseq atlas cell types #
#################################################
library(Seurat)
library(scater)
library(loomR) 
library(SingleCellExperiment)
library(locfit)
library(ACTIONet)
library(SCINET) 
library(igraph) 


## define assay_name slot for all ACTIONet/SCINET functions
args = commandArgs(TRUE)

cell_super = args[1]   # DCs, MFs, Bcells, Tcells
des_slot = "counts" 
default_assay = "SCT"
batch_corr = F  # whether to batch correct using Harmony (True) or not (False)
assay_name = "counts"

path_stem = "./scinet_interactomes/"


## define input paths
if (cell_super == "DCs") {
  obj = readRDS("combined_DC_integrated_16dataset_v2.rds")
  celltype_annot_col = "clusters_0.4"
}

if (cell_super == "MFs") {
  obj = readRDS(file = "integrated_s17_macro_220606.rds") 
  celltype_annot_col = "clust24name"
}

if (cell_super == "Bcells") {
  obj = readRDS(file = "Tumourlowres_subsetted_integrated.rds")
  celltype_annot_col = "clust_names"
}

if (cell_super == "Tcells") {
  obj = readRDS(file = "combined_TCELL_integrated_16dataset.rds")
  celltype_annot_col = "clusters"
}


## Read in files
DefaultAssay(obj) = default_assay

## convert to single cell experiment object
ace <- SingleCellExperiment(assays = list(counts = GetAssayData(object = obj, slot = des_slot)),
                            colData = obj@meta.data)  

## Assign a priori identified cell type annotations to each cell
celltypes = as.character(ace@colData[, celltype_annot_col])
rm(obj)

## run dimensional reduction using ACTIONet
ace = reduce.ace(ace = ace, assay_name = assay_name)

## batch correct if desired
if (batch_corr) {
  ace = reduce.and.batch.correct.ace.Harmony(ace, batch_attr = ace$study, assay_name = assay_name, max_iter = 50)
  print("corrected for study using Harmony")  
  file_substr = "batchCorr"
} else {
  print("not performing batch correction")
  file_substr = "noBatchCorr"
}


## Run ACTIONet 
ace = run.ACTIONet(ace = ace, assay_name = assay_name)  

## Compute specificity of genes for each cell type/annotation (from ACTIONet package)
# returns an ACE object with specificity scores of each cluster added to rowMaps(ace) as a matrix with name defined by output_slot
ace = compute.cluster.feature.specificity(ace, celltypes, "celltype_specificity_scores", assay_name = assay_name)
print("computed cluster feature specificity")

## Construct networks using the annotated cell types (from ACTIONet package)
Celltype.specific.networks = run.SCINET.clusters(ace, specificity.slot.name = "celltype_specificity_scores_feature_specificity")
saveRDS(Celltype.specific.networks, 
        file = paste0(path_stem, cell_super, "/", cell_super, "_celltypeSpecificNetworks_", 
                      default_assay, "_", des_slot, "_", file_substr, ".rds"))

print("finished and saved final output object to rds file")





