#########################################
# Plot node and edge pathway permutation feature importance #
########################################
library(tidyverse)
library(ggplot2)
library(ggraph)
library(visNetwork)
library(igraph)
`%ni%` = Negate(`%in%`)

## define variables
des_targs = c("PTPN22", "OSM", "OSMR")

## read in manually curated pathways
all_pathways = read.csv(header = T, stringsAsFactors = F, file = "CTD_genes_pathways_reactome.csv")
colnames(all_pathways)[1:2] = c("gene", "pathway_name")
all_pathways$pathway_id_clean = unlist(lapply(strsplit(all_pathways$PathwayID, split = ":"), FUN = function(x) {x[2]}))

## read in data
perm_node = read.csv(header = T, stringsAsFactors = F, file = "PathwayPermImp_NodeFeatures_PTPN22-OSM-OSMR.csv  ")
perm_edge = read.csv(header = T, stringsAsFactors = F, file = "PathwayPermImp_Edges_PTPN22-OSM-OSMR.csv")

## read in and create hierarchy from reactome
pathways = read.table(header = F, stringsAsFactors = F, sep = "\t", file = "ReactomePathwayHierarchy.txt",
                      col.names = c("parent", "child"))

## Tidy names
perm_edge$pathway_id_clean = unlist(lapply(strsplit(perm_edge$pathway_id, split = ":"), FUN = function(x) {x[2]}))
perm_node$pathway_id_clean = unlist(lapply(strsplit(perm_node$pathway_id, split = ":"), FUN = function(x) {x[2]}))

## Read in pathway names (manually tidied)
pathway_clean = read.csv(header = T, stringsAsFactors = F, encoding = "UTF-8", file = paste0(path_stem, "AbbrevPathwayNamesClean_OSM_OSMR_PTPN22.csv"))

## 1 pathway tree per target
plot_pathway_perm_tree = function(curr_targ) {
  curr_df = perm_edge %>% filter(gene_expl %in% curr_targ)
  tmp_edges = pathways %>% filter((parent %in% curr_df$pathway_id_clean) | (child %in% curr_df$pathway_id_clean))
  tree_path = igraph::graph_from_edgelist(as.matrix(tmp_edges), directed = T)

  # Extract nodes from current cc
  tmp_nodes = V(tree_path)
  tmp_nodes_df = data.frame(node_id = as.integer(tmp_nodes), pathway_id_clean = names(tmp_nodes))
  
  # Tested pathways and the bridging pathways
  tmp_attr = perm_edge %>% filter(gene_expl == curr_targ) %>%
    select(pathway_id_clean, pathway_name, gene_expl, mean_abs_perm_imp_norm_sqrtedge, med_abs_perm_imp_norm_sqrtedge) %>%
    merge(y = tmp_nodes_df, by = "pathway_id_clean", all.y = T)
  
  # can have NA for pathway_name.y here, if the pathway was not manually curated in the entry
  tmp_attr = merge(x = tmp_attr, y = dplyr::distinct(pathway_clean[, c("pathway_id_clean", "pathway_name_clean")]), 
                   by = "pathway_id_clean", all.x = T)
  
  if (length(curr_targ) > 1) {
    ## as this is the same axis, group OSM and OSMR together by mean (with 2 data points, mean = median)
    tmp_attr = tmp_attr %>% group_by(pathway_id_clean, pathway_name_clean, node_id) %>%
      summarise(mean_abs_perm_imp_norm_sqrtedge = mean(mean_abs_perm_imp_norm_sqrtedge),
                med_abs_perm_imp_norm_sqrtedge = mean(med_abs_perm_imp_norm_sqrtedge)) %>%
      as.data.frame()
    curr_targ = "OSM-OSMR"
  }
  
  # Order rows of tmp_attr to match node ordering in graph
  oracle = all(tmp_attr$pathway_id_clean == names(tmp_nodes))
  if (!oracle) {
    orderings = match(names(tmp_nodes), tmp_attr$pathway_id_clean)
    tmp_attr = tmp_attr[orderings, ] 
  }
  
  V(tree_path)$imp = tmp_attr$med_abs_perm_imp_norm_sqrtedge
  V(tree_path)$name_clean = tmp_attr$pathway_name_clean
  
  p1 = ggraph(tree_path, layout = "tree", circular = F) +
    geom_edge_diagonal(strength = 1) +
    geom_node_text(mapping = aes(label = name_clean, angle = 0, color = imp), size = 4) +
    #scale_color_viridis(alpha = 1) +
    scale_color_gradient(low = "orange", high = "red") +
    ggtitle(curr_targ) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5))
  p1
return(p1)
}


plot_pathway_perm_tree(curr_targ = "PTPN22")
plot_pathway_perm_tree(curr_targ = c("OSM", "OSMR"))
