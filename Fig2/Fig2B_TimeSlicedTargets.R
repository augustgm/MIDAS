#####################
# Compare predictions on time-sliced targets to randomly sample size-matched gene sets
#######################
library(dplyr)
library(ggplot2)
library(ggpubr)


## read in data
plot_df = read.csv(header = T, stringsAsFactors = F, file = "Fig2B_timeSlicedTargets_forPlot.csv")

## plot
ggplot(data = plot_df, 
       mapping = aes(x = source, y = proba, fill = source)) +
  #geom_violin(linewidth = 1.3) +
  geom_boxplot(notch = T) +  # width = 0.1
  ggpubr::stat_compare_means(method = "wilcox.test", size = 4.5, label.x = 1.2) +
  ylab("Predicted IO target probability") +
  scale_fill_manual(values = c("random" = "#EFB75E", "time-sliced" = "#ED7A5D")) +
  theme_bw() +
  theme(text = element_text(size = 18))
