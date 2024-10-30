#####################
# Compare predictions on time-sliced targets to randomly sample size-matched gene sets
#######################
library(dplyr)
library(ggplot2)
library(ggpubr)


## read in time-sliced targets
new_targs = read.csv(header = T, stringsAsFactors = F, file = "NewIOtargets_CTgov.csv")
new_targs = new_targs %>% filter(Include != "N") %>% as.data.frame()
new_targs = new_targs %>% tidyr::separate_rows(Target, sep = ";") %>% as.data.frame()

## exclude any overlapping targets
true_pos = read.csv(header = T, stringsAsFactors = F, file = "CRI2017-2019_IOtargets_exclAgbased_CellTherapy_OncolyticVirus.csv")

new_targs = new_targs %>% filter(!(Target %in% true_pos$Hugo)) %>% as.data.frame()
new_targs = new_targs %>% filter(Target != "") %>% as.data.frame()
new_targs = new_targs %>% filter(!(Target %in% c("IL12B", "IL15RA", "IL2"))) %>% as.data.frame()

## check rankings in predicted list 
predictions = read.csv(header = T, stringsAsFactors = F, file = "GiG_NeighbourLoaderKNN50_cvTestHout_predProba.csv")

## annotate new targets
predictions$new_targs = "no"
predictions[predictions$Gene %in% new_targs$Target, "new_targs"] = "yes"
predictions = predictions[order(predictions$Gene), ]  


null_dist = data.frame()
for (i in 1:1e3) {
  set.seed(i)
  rand_genes = sample(x = predictions$Gene, replace = F, size = nrow(predictions[predictions$new_targs == "yes", ]))
  plot_df = predictions[predictions$Gene %in% c(rand_genes, new_targs$Target), ]
  plot_df$rand_vs_targs = "rand"
  plot_df[plot_df$Gene %in% new_targs$Target, "rand_vs_targs"] = "new targets"
  
  tmp = data.frame(model = model_type, subtype = ifelse(test = (model_type == "graph"), yes = gnn_type,  no = paste0("noAutoim_", no_autoim)),
                   rand_mean_prob = mean(plot_df[plot_df$rand_vs_targs == "rand", "mean_prob"]), seed = i)
  
  null_dist = rbind(null_dist, tmp)

}
pval = nrow(null_dist[null_dist$rand_mean_prob >=  mean(predictions[predictions$new_targs == "yes", "mean_prob"]), ]) / nrow(null_dist)

plot_df_pred = data.frame(proba = predictions[predictions$new_targs == "yes", "mean_prob"], source = "time-sliced")
plot_df_rand = data.frame(proba = null_dist$rand_mean_prob, source = "random")
plot_df = rbind(plot_df_pred, plot_df_rand)

ggplot(data = plot_df, 
       mapping = aes(x = source, y = proba, fill = source)) +
  #geom_violin(linewidth = 1.3) +
  geom_boxplot(notch = T) +  # width = 0.1
  ggpubr::stat_compare_means(method = "wilcox.test", size = 4.5, label.x = 1.2) +
  ylab("Predicted IO target probability") +
  scale_fill_manual(values = c("random" = "#EFB75E", "time-sliced" = "#ED7A5D")) +
  theme_bw() +
  theme(text = element_text(size = 18))
