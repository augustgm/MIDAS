################################################
# Analyse node permutation feature importance    
################################################
library(tidyverse)
library(ggplot2)
library(ggpubr)

## read in model performance
mod_perf = read.csv(header = T, stringsAsFactors = F, file = "GIN_model_performance.csv")
mod_perf = mod_perf %>% filter(KNN == "KNN=50")

## read data type and biological categories annotations
feat_categs = read.csv(header = T, stringsAsFactors = F, file = "GINinterpret/CategoriseFeatures.csv")

## define path stem 
path_stem = "GINinterpret/permImp/"

## read in data
file_vec = list.files(path_stem)
file_vec = grep(x = file_vec, pattern = "csv", value = T)
file_vec = grep(x = file_vec, pattern = "_permScores", value = T)
perm_all = data.frame()
for (i in 1:length(file_vec)) {
  curr_df = read.csv(header = T, stringsAsFactors = F, file = paste0(path_stem, file_vec[i]))
  perm_all = rbind(perm_all, curr_df)
}
rm(i, curr_df, file_vec)


## merge with feat categs
perm_all = merge(x = feat_categs, y = perm_all, by = "feature", all = T)

### Define functions ###
compute_perm_feat_imp = function(perm_res, true_perf=mod_perf[mod_perf$Set == "Train", "auc"]) {
  return(length(which(perm_res >= true_perf)) / length(perm_res))
}


compute_mean_effect_size = function(perm_res, true_perf=mod_perf[mod_perf$Set == "Train", "auc"]) {
  return(true_perf - mean(perm_res))
}


barplot_feat_imp_nogig = function(perm_res_df, xlab_str) {  
  ## change column names for below
  colnames(perm_res_df)[1] = "feature"
  
  ## format p-values
  perm_res_df = perm_res_df %>% mutate(sig_stat = ifelse(test = (padj < 0.05), yes = "FDR<0.05", no = "n.s"))
  
  ## order factors for better aesthetics
  perm_res_df[perm_res_df$feature == "DCs dysfunctional", "feature"] = "DCs\ndysfunctional"
  perm_res_df[perm_res_df$feature == "Immunosuppressive mac", "feature"] = "Suppressive\nmac"
  perm_res_df[perm_res_df$feature == "T proliferating", "feature"] = "T\nproliferating"

  perm_res_df$feature = factor(perm_res_df$feature,
                               levels = perm_res_df[order(perm_res_df$mean_eff_size, decreasing = T), "feature"])
  
  ## plot as column plot
  perm_res_df$sig_stat = "other"
  perm_res_df[perm_res_df$feature %in% levels(perm_res_df$feature)[1:5], "sig_stat"] = "top5"
  p1 = ggplot(data = perm_res_df, mapping = aes(x = feature, y = mean_eff_size, fill = sig_stat)) +
    geom_col() +
    ylab("Permutation feature importance") + xlab(xlab_str) + labs(fill = "FDR") +
    #scale_fill_manual(values = c("FDR<0.05" = "darkred", "n.s" = "grey")) +
    #scale_fill_manual(values = c("FDR<0.05" = "#55868c", "n.s" = "#b8d5d4")) +
    scale_fill_manual(values = c("top5" = "#eec2a4", "other" = "#685968")) +
    theme_bw() +
    theme(text = element_text(size = 16), # axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 60, hjust = 1, vjust = 1))
  
  return(p1) 
}


#### Analyse by biological category ####
perm_biocat_imp = perm_all %>% group_by(bio_category) %>%
  summarise(mean_eff_size = compute_mean_effect_size(perm_res = perm_roc_auc),
            perm_pval = compute_perm_feat_imp(perm_res = perm_roc_auc)) %>%
  mutate(padj = p.adjust(perm_pval, method = "fdr")) %>%
  as.data.frame()


p4 = barplot_feat_imp_nogig(perm_res_df = perm_biocat_imp, xlab_str = "Biological category")
p4 = p4 + theme(legend.position = "none")
p5 = p4 + coord_polar(start = 0) + 
  # Annotate custom scale inside plot
  annotate(x = -0.8, y = 0.01, label = "", geom = "text", color = "gray12" ) +
  annotate(x = 0, y = 0.01, label = "0.01", geom = "text", color = "gray12" ) +
  annotate(x = 0, y = 0.02, label = "0.02", geom = "text", color = "gray12" ) +
  annotate(x = 0, y = 0.03, label = "0.03", geom = "text", color = "gray12" ) +
  
  # Scale y axis so bars don't start in the center
  scale_y_continuous(
    limits = c(-0.01, 0.04),
    expand = c(0, 0),
    breaks = c(0, 1000, 2000, 3000)) +
  
  theme(
    axis.text.x = element_text(angle = 0),
    # Remove axis ticks and text
    panel.border = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "none",
    #plot.margin = margin(10, 10, 10, 10)
  )

p5
