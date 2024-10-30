##############################
# Plot permutation importance for GiG edges on training set
###############################
library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(pROC)

#### Read in CV training performance across all folds ####
get_roc = function(pred_df, target_column, pred_column, curr_fold, graph_type) {
  roc_obj = pROC::roc(response = pred_df[pred_df$Fold_ID == curr_fold, target_column], 
                      predictor = pred_df[pred_df$Fold_ID == curr_fold, pred_column],
                      auc = T, ci = T, plot = F)
  output = data.frame(graph_backbone = graph_type, 
                      fold = curr_fold, 
                      auc = roc_obj$auc, 
                      lowerCI = roc_obj$ci[1], 
                      upperCI = roc_obj$ci[3])
  return(output)
}

## read in targets
targ_df = read.csv(header = T, stringsAsFactors = F, file = "CRI2017-2019_IOtargets_exclAgbased_CellTherapy_OncolyticVirus.csv")

## Read in data
curr_res = read.csv(header = T, stringsAsFactors = F, 
file = "Predictions_Best_GINBiDir_WithCVModels_GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_GiG_KNN50.csv") %>%
  filter(Set == "Train (fold)") %>%
  mutate(graph_type = "GiG",
         KNN = 50) %>%
  as.data.frame()

## iterate across all CV folds to determine training performance
roc_res = data.frame()

for (i in 0:99) {
  roc_tr = get_roc(pred_df = curr_res, target_column = "Target", pred_column = "Prob", 
                   curr_fold = i, graph_type = "GiG")
  roc_tr$Set = "Train"
  
  ## concatenate
  roc_res = rbind(roc_res, roc_tr)
}
rm(i, roc_tr, file_list)


#### Read in permutation performance ####
## read in model performance
mod_perf = read.csv(header = T, stringsAsFactors = F, file = "GIN_model_performance.csv")
mod_perf = mod_perf %>% filter(KNN == "KNN=50")

## define path stem 
edge_path = "GINinterpret/permEdgeImp/"

## read in edge permutation data
file_vec = list.files(edge_path)
file_vec = grep(x = file_vec, pattern = "csv", value = T)
perm_all = data.frame()
for (i in 1:length(file_vec)) {
  curr_df = read.csv(header = T, stringsAsFactors = F, file = paste0(edge_path, file_vec[i]))
  curr_df = curr_df %>% mutate(perf_drop = perm_roc_auc - mod_perf[mod_perf$Set == "Train", "auc"]) %>% as.data.frame()
  perm_all = rbind(perm_all, curr_df)
}
rm(i, curr_df, file_vec)

colnames(roc_res)[1:2] = c("feature", "CV_fold")
colnames(perm_all)[5] = "auc"
plot_df = rbind(roc_res %>% select(CV_fold, feature, auc) %>% mutate(set = "Orig") %>% as.data.frame(),
                perm_all %>% select(CV_fold, feature, auc) %>% mutate(set = "Perm") %>% as.data.frame())

## Density plot instead
p1 = ggplot(data = plot_df, mapping = aes(x = auc, y = set, fill = set)) +
  ggridges::geom_density_ridges() +
  scale_fill_manual(values = c("Orig" = "#ED7A5D", "Perm" = "#EFB75E")) +
  xlab("Train ROC-AUC") + ylab("Network topology") +
  theme_bw() +
  theme(text = element_text(size = 18), legend.position = "bottom")

p1 

