#############
# Plot model predictions for targets
#############
library(readxl)
library(tidyverse)
library(ggplot2)

# Read in data
preds = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig4_SourceData.xlsx",
                           sheet = "Fig4A_AllPredictions") %>% as.data.frame()

plot_df = readxl::read_excel(path = "C:/Users/Marce/OneDrive/Documents/Rscripts-MBPhD/MIDAS/PaperFigures/2025-11-ReviewUpdate/Fig4_SourceData.xlsx",
                             sheet = "Fig4A_Targets") %>% as.data.frame()


#preds = read.csv(header = T, stringsAsFactors = F, file = "Fig4A_candidateIOtargets_AllPreds_forPlot.csv")
#plot_df = read.csv(header = T, stringsAsFactors = F, file = "Fig4A_candidateIOtargets_Targets_forPlot.csv")

unique(preds$label)
preds[preds$label == "clinically\r\r\nunexplored", "label"] = "clinically\nunexplored"
preds[preds$label == "Known IO\r\r\ntargets", "label"] = "Known IO\ntargets"


# Plot
p1 = ggplot(data = preds, mapping = aes(x = mean_prob, y = label)) +
  ggridges::geom_density_ridges(mapping = aes(fill = label),
                                quantile_lines = T, quantile_fun = function(mean_prob, ...) {median(mean_prob)},
                                scale = 5, alpha = 0.9, bandwidth = 0.04) +
  geom_point(data = plot_df, mapping = aes(x = mean_prob, y = label, color = target_stat), shape = 8, size = 5) +
  #geom_text(data = plot_df, mapping = aes(x = mean_prob, y = label, label = Gene)) +
  ggrepel::geom_label_repel(data = plot_df, mapping = aes(x = mean_prob, y = label, label = Gene)) +
  scale_fill_manual(values = c("clinically\nunexplored" = "#414b69", "Known IO\ntargets" = "#eec2a4")) +
  scale_color_manual(values = c("IO\ncandidate" = "#F4B23D")) +
  labs(x = "P(IO target)", fill = "Clinical status", color = "") +
  xlim(0, 1) +
  theme_bw() +
  theme(text = element_text(size = 18), axis.title.y = element_blank())

p1


