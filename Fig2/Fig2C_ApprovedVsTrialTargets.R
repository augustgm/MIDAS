##################################################
# Plotting predictions for targets that are approved and those that are undergoing clinical trial development
##################################################
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(pROC)

## Read in predictions
curr_res = read.csv(header = T, stringsAsFactors = F, file = "Predictions_Best_GINBiDir_WithCVModels_GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_GiG_KNN50.csv")
curr_res$graph_type = "GiG"
curr_res$KNN = "KNN=100"

## aggregate target prediction
pred_agg = curr_res %>% group_by(Gene, Set, KNN) %>%
  summarise(mean_prob = mean(Prob),
            Target = unique(Target)) %>%
  as.data.frame()
  
pred_agg = pred_agg[order(pred_agg$mean_prob, decreasing = T), ]

## Read in targets ###
phase_df = read.csv(header = T, stringsAsFactors = F, file = "CRI2020_IOtargetsClinicalPhase_clean_v1.csv")

## combine phase annotations
phase_df = phase_df %>% group_by(HUGO_symbol) %>% summarise(max_phase = max(max_phase)) %>% as.data.frame()

assess_target_phase_benchmark = function(pred_agg_test, targs_phase_annot) {
  pred_agg_test = pred_agg_test[pred_agg_test$KNN == "KNN=50", ]  
  
  ### annotate target phase
  pred_agg_test$phase = "NA"
  pred_agg_test[pred_agg_test$Gene %in% targs_phase_annot[targs_phase_annot$max_phase == 4, "HUGO_symbol"], "phase"] = "Approved"
  pred_agg_test[pred_agg_test$Gene %in% targs_phase_annot[targs_phase_annot$max_phase == 3, "HUGO_symbol"], "phase"] = "Phase 1"
  pred_agg_test[pred_agg_test$Gene %in% targs_phase_annot[targs_phase_annot$max_phase == 2, "HUGO_symbol"], "phase"] = "Phase 2"
  pred_agg_test[pred_agg_test$Gene %in% targs_phase_annot[targs_phase_annot$max_phase == 1, "HUGO_symbol"], "phase"] = "Phase 3"
  print(table(pred_agg_test$phase))
  pred_agg_test$phase = factor(pred_agg_test$phase, levels = c("NA", "Phase 1", "Phase 2", "Phase 3", "Approved"))
    
  ## plot approved vs clinical development vs non-target
  pred_agg_test$appr_vs_clin_vs_non = "NA"
  pred_agg_test[pred_agg_test$phase == "Approved", "appr_vs_clin_vs_non"] = "Approved"
  pred_agg_test[pred_agg_test$phase %in% c("Phase 3", "Phase 2", "Phase 1"), "appr_vs_clin_vs_non"] = "Trials"
    
  nortest::ad.test(pred_agg_test[pred_agg_test$appr_vs_clin_vs_non == "Approved", "mean_prob"])
  nortest::ad.test(pred_agg_test[pred_agg_test$appr_vs_clin_vs_non == "Trials", "mean_prob"])
    
  print(kruskal.test(formula = mean_prob ~ appr_vs_clin_vs_non, data = pred_agg_test))
  print(wilcox.test(formula = mean_prob ~ appr_vs_clin_vs_non, data = pred_agg_test %>% filter(appr_vs_clin_vs_non != "NA")))
      
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
    #geom_text(mapping = aes(label = paste0("n = ", count), y = upper_bound + 0.02), size = 4.5) +
    xlab("Clinical development phase") + ylab ("Mean P(IO target)") + labs(fill = "Target stage") +
    scale_fill_manual(values = c("NA" = "#548E9E", #"Phase 1" = "#AEDDCE", "Phase 2" = "#BCA1BC",
                                 "Trials" = "#F29B86", "Approved" = "#F3C986")) +
    theme_bw() +
    theme(text = element_text(size = 18))
    
  return(list(p1, pred_agg_test))
}

res_k50 = assess_target_phase_benchmark(pred_agg_test = rbind(test_pred, hout_pred), targs_phase_annot = phase_df)
res_k50[[1]]
