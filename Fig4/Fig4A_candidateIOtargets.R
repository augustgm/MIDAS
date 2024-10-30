#############
# Plot model predictions for targets
#############
library(tidyverse)
library(ggplot2)

## define targets to plot
des_targs = c("OSM", "OSMR", "PTPN22") 

## read in predictions
preds = read.csv(header = T, stringsAsFactors = F, file = "GiG_NeighbourLoaderKNN50_cvTestHout_predProba.csv")

## filter
preds = preds %>% mutate(des_stat = ifelse(test = Gene %in% des_targs, yes = "desired", no = "non-target")) %>% as.data.frame()
preds$rank = 1:nrow(preds)

## plot simple bar graph showing predicted probability
preds_targ = preds %>% filter(Gene %in% des_targs)
preds_targ$Gene = factor(preds_targ$Gene, levels = preds_targ[order(preds$mean_prob, decreasing = F), "Gene"])

sum_stats = preds %>% group_by(Target) %>%
  summarise(mean_val = mean(mean_prob), 
            med_val = median(mean_prob), 
            sd_val = sd(mean_prob),
            n_val = n(),
            sem = sd(mean_prob) / sqrt(n())) %>%
  mutate(lower_sem = mean_val - sem, 
         upper_sem = mean_val + sem) %>%
  as.data.frame()

colnames(sum_stats)[1:2] = c("Gene", "mean_prob")
plot_df = rbind(preds_targ[, c("Gene", "mean_prob")], sum_stats[, c("Gene", "mean_prob")])
plot_df$Gene = factor(plot_df$Gene, levels = plot_df[order(plot_df$mean_prob, decreasing = F), "Gene"])

## plot some ridge plots
preds$label = "clinically\nunexplored"
preds[preds$Target == 1, "label"] = "Known IO\ntargets"

plot_df = plot_df %>% filter(Gene %in% des_targs) %>% 
  mutate(label = "clinically\nunexplored",
         target_stat = "IO\ncandidate") %>%
  as.data.frame()

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
