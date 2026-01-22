#####################
# Compare predictions on time-sliced targets to randomly sample size-matched gene sets
#######################
library(dplyr)
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggpubr)


rm(list = ls())

## read in data - using this allows to do the whole sampling to get the answer - for just plotting from source data see belwo
predictions = read.csv(header = T, stringsAsFactors = F,
                       file = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/Fig2B_allTimeSliced_forPlot.csv")

## randomly sample
null_dist = data.frame()
for (i in 1:1e3) {
  set.seed(i)
  rand_genes = sample(x = predictions$id, replace = F, size = nrow(predictions[predictions$new_targs == "yes", ]))
  
  tmp = data.frame(model = "graph", subtype = "GINBiDir",
                   rand_mean_prob = mean(predictions[predictions$id %in% rand_genes, "mean_prob"], seed = i))
  
  null_dist = rbind(null_dist, tmp)

}
pval = nrow(null_dist[null_dist$rand_mean_prob >=  mean(predictions[predictions$new_targs == "yes", "mean_prob"]), ]) / nrow(null_dist)

plot_df_pred = data.frame(proba = predictions[predictions$new_targs == "yes", "mean_prob"], source = "time-sliced")
plot_df_rand = data.frame(proba = null_dist$rand_mean_prob, source = "random")
plot_df = rbind(plot_df_pred, plot_df_rand)


# this is equivalent to the read_excel one below.
#plot_df = read.csv(header = T, stringsAsFactors = F,
#                   file = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig2B_allTimeSlicedVsRandom_inclNull_forPlot.csv")

##### from source data
plot_df = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig2_SourceData.xlsx",
                             sheet = "Fig2B") %>% as.data.frame()



ggplot(data = plot_df, 
       mapping = aes(x = source, y = proba, fill = source)) +
  #geom_violin(linewidth = 1.3) +
  geom_boxplot(notch = T) +  # width = 0.1
  ggpubr::stat_compare_means(method = "wilcox.test", size = 4.5, label.x = 1.2) +
  ylab("Predicted IO target probability") +
  scale_fill_manual(values = c("random" = "#EFB75E", "time-sliced" = "#ED7A5D")) +
  theme_bw() +
  theme(text = element_text(size = 18))
