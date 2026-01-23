##############################
# Plot permutation importance for GiG edges on training set
###############################
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggpubr)

# Read in data
plot_df = readxl::read_excel(path = "Fig3_SourceData.xlsx", 
                             sheet = "Fig3C") %>% as.data.frame()

# Density plot
p1 = ggplot(data = plot_df, mapping = aes(x = auc, y = set, fill = set)) +
  ggridges::geom_density_ridges() +
  scale_fill_manual(values = c("Orig" = "#ED7A5D", "Perm" = "#EFB75E")) +
  xlab("Train ROC-AUC") + ylab("Network topology") +
  theme_bw() +
  theme(text = element_text(size = 18), legend.position = "bottom")

p1 

