############
# Extract topological specificity for cell type-specific interactomes
##############
library(igraph)
library(dplyr)

## define parameters
assay = "SCT"
slot = "counts"
batch_corr = "noBatchCorr"
supertypes = c("DCs", "Bcells", "MFs", "Tcells")

## set path
path_stem = "./scinet_interactomes/"

## iterate through files
for (i in 1:length(supertypes)) {
  curr_super = supertypes[i]  
  curr_networks = readRDS(paste0(path_stem, curr_super, "/", curr_super, "_celltypeSpecificNetworks_",
                                 assay, "_", slot, "_", batch_corr, ".rds"))  
  subtype_names = names(curr_networks)
  
  lapply(X = seq_along(curr_networks), FUN = function(index) {  
    subtype_net = curr_networks[[index]]
    curr_cell = names(curr_networks)[index]
    curr_cell = gsub(pattern = "/", replacement = "_", x = curr_cell)
    
    ## extract topological specificity
    nodes = igraph::V(subtype_net)
    topo_spec_df = data.frame(gene = nodes$name, topo_spec = nodes$specificity)
    
    ## save to file
    topo_spec_file_path = paste0(path_stem,
                                 curr_cell, "_topospec_", assay, "_", slot, "_", batch_corr, ".csv")
    print(topo_spec_file_path)
    write.csv(x = topo_spec_df, row.names = F, file = topo_spec_file_path)
  })
}
