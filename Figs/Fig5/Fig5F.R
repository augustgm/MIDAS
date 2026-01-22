#########
# Fig 5F
#########
library(tidyverse)
library(readxl)
library(ggplot2)
library(reshape2)

rm(list = ls())

## read in data
ccl4_data = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig5_SourceData.xlsx",
                               sheet = "Fig5F") %>%
  as.data.frame()


## tidy data
colnames(ccl4_data)[1] = "Patient_ID"
ccl4_data = ccl4_data[1:8, ]

plot_df = reshape2::melt(ccl4_data, id.vars = c("Patient_ID"), variable.name = "condition")
plot_df$condition = as.character(plot_df$condition)
plot_df[plot_df$condition == "DMSO", "condition"] = "Control"
plot_df[plot_df$condition == "anti-OSM", "condition"] = "aOSM"

plot_df$condition = factor(plot_df$condition, levels = c("Control", "aOSM"))


## plot
wilcox.test(formula = value ~ condition, data = plot_df, paired = T)

colours = c("#AECAE4", "#CFD5EA", "#AF93BA", "#F1D687", "#D69E78", "#E68C7C", "#AECEB5", "#D6D6CE")

p1 = ggplot(data = plot_df, mapping = aes(x = condition, y = value)) +
  geom_boxplot() +
  geom_jitter(height = 0, width = 0.25, mapping = aes(color = Patient_ID), size = 4) +
  labs(x = "Condition", y = "CCL4 (pg/mL)") +
  scale_color_manual(values = colours) +
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "none") 

p1

