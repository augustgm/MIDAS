#########################################
# Plot node and edge pathway permutation feature importance #
########################################
library(tidyverse)
library(ggplot2)
library(ggraph)
library(visNetwork)
library(igraph)
`%ni%` = Negate(`%in%`)



rm(list = ls())

## define variables
des_targs = c("PTPN22", "OSM", "OSMR") 

## read in manually curated pathways
all_pathways = read.csv(header = T, stringsAsFactors = F, 
                        file = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/CTD_genes_pathways_reactome.csv")
colnames(all_pathways)[1:2] = c("gene", "pathway_name")
all_pathways$pathway_id_clean = unlist(lapply(strsplit(all_pathways$PathwayID, split = ":"), FUN = function(x) {x[2]}))

## read in data
perm_edge_a = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig4_SourceData.xlsx",
                                 sheet = "Fig4B_PTPN22") %>% as.data.frame()

perm_edge_b = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig4_SourceData.xlsx",
                                 sheet = "Fig4C_OSM-OSMR") %>% as.data.frame()

perm_edge = rbind(perm_edge_a, perm_edge_b)
rm(perm_edge_a, perm_edge_b)

perm_edge = perm_edge %>% 
  group_by(gene_expl, pathway_id, pathway_name) %>%
  summarise(mean_abs_perm_imp_norm_sqrtedge = mean(abs(norm_sqrtedge_perm_imp), na.rm = T),
            med_abs_perm_imp_norm_sqrtedge = median(abs(norm_sqrtedge_perm_imp), na.rm = T)) %>%
  as.data.frame()




## read in and create hierarchy from reactome
pathways = read.table(header = F, stringsAsFactors = F, sep = "\t",
                      file = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/ReactomePathwayHierarchy.txt",
                      col.names = c("parent", "child"))

## Tidy names
perm_edge$pathway_id_clean = unlist(lapply(strsplit(perm_edge$pathway_id, split = ":"), FUN = function(x) {x[2]}))

## Read in pathway names (manually tidied)
pathway_clean = read.csv(header = T, stringsAsFactors = F, encoding = "UTF-8", 
                         file = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/AbbrevPathwayNamesClean_OSM_OSMR_PTPN22.csv")

## 1 pathway tree per target
plot_pathway_perm_tree = function(curr_targ) {
  curr_df = perm_edge %>% filter(gene_expl %in% curr_targ)
  tmp_edges = pathways %>% filter((parent %in% curr_df$pathway_id_clean) | (child %in% curr_df$pathway_id_clean))
  tree_path = igraph::graph_from_edgelist(as.matrix(tmp_edges), directed = T)

  # Extract nodes from current cc
  tmp_nodes = V(tree_path)
  tmp_nodes_df = data.frame(node_id = as.integer(tmp_nodes), pathway_id_clean = names(tmp_nodes))
  
  # Tested pathways and the bridging pathways
  tmp_attr = perm_edge %>% filter(gene_expl %in% curr_targ) %>%
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
