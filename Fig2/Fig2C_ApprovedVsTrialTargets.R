##################################################
# Plotting predictions for targets that are approved and those that are undergoing clinical trial development
##################################################
library(tidyverse)
library(ggplot2)
library(ggpubr)

# read in data
plot_df = read.csv(header = T, stringsAsFactors = F, file = "Fig2C_TargetsByPhase_forPlot.csv")
plot_df$appr_vs_clin_vs_non = factor(plot_df$appr_vs_clin_vs_non, levels = c("NA", "Trials", "Approved")[3:1])

# plot 
ggplot(data = plot_df, mapping = aes(x = appr_vs_clin_vs_non, y = mean_proba, fill = appr_vs_clin_vs_non)) + 
  geom_col() +
  geom_errorbar(mapping = aes(ymin = lower_bound, ymax = upper_bound), linewidth = 1) +
  #geom_text(mapping = aes(label = paste0("n = ", count), y = upper_bound + 0.02), size = 4.5) +
  xlab("Clinical development phase") + ylab ("Mean P(IO target)") + labs(fill = "Target stage") +
  scale_fill_manual(values = c("NA" = "#548E9E", #"Phase 1" = "#AEDDCE", "Phase 2" = "#BCA1BC",
                               "Trials" = "#F29B86", "Approved" = "#F3C986")) +
  theme_bw() +
  theme(text = element_text(size = 18))
