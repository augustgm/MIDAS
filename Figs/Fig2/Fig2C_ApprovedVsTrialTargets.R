##################################################
# Plotting predictions for targets that are approved and those that are undergoing clinical trial development
##################################################
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggpubr)
library(pROC)

## Read in predictions
preds = readxl::read_excel(path = "Fig2_SourceData.xlsx",
                           sheet = "Fig2C") %>%
  as.data.frame()

assess_target_phase_benchmark = function(pred_agg_test) {

  ## stat stest
  ks_res = kruskal.test(formula = mean_prob ~ appr_vs_clin_vs_non, data = pred_agg_test)  
  print(ks_res)
  print(ks_res$p.value)
  print("")
  print(wilcox.test(formula = mean_prob ~ appr_vs_clin_vs_non, data = pred_agg_test %>% filter(appr_vs_clin_vs_non != "NA")))
      
  ## aggregate for plot 
  plot_df = pred_agg_test %>%
    group_by(appr_vs_clin_vs_non) %>%
    summarise(mean_proba = mean(mean_prob),
              med_proba = median(mean_prob),
              sd_proba = sd(mean_prob),
              count = n()) %>%
    mutate(sem = sd_proba / sqrt(count)) %>% 
    mutate(upper_bound = mean_proba + sem,
           lower_bound = mean_proba - sem) %>%
    as.data.frame()
    
  ## plot
  plot_df$appr_vs_clin_vs_non = factor(plot_df$appr_vs_clin_vs_non, levels = c("NA", "Trials", "Approved")[3:1])
  
  p1 = ggplot(data = plot_df, mapping = aes(x = appr_vs_clin_vs_non, y = mean_proba, fill = appr_vs_clin_vs_non)) +
    geom_col() +
    geom_errorbar(mapping = aes(ymin = lower_bound, ymax = upper_bound), linewidth = 1) +
    xlab("Clinical development phase") + ylab ("Mean P(IO target)") + labs(fill = "Target stage") +
    scale_fill_manual(values = c("NA" = "#414D6A", "Trials" = "#F29B86", "Approved" = "#F3C986")) +
    theme_bw() +
    theme(text = element_text(size = 18))
    
  return(p1)
}

res_k50 = assess_target_phase_benchmark(pred_agg_test = preds)
res_k50


