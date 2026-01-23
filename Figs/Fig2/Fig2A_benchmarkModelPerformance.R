##############################################################
# Benchmark held-out MIDAS performances against alternatives #
##############################################################
library(pROC)
library(ggplot2)
library(ggpubr)
library(readxl)
library(tidyverse)


rm(list = ls())

## define path stem 
path_stem = ""

## read data
preds_all = readxl::read_excel(path = "Fig2_SourceData.xlsx",
                               sheet = "Fig2A") %>%
  as.data.frame()


## graph model: GIN KNN=50
head(preds_all)
roc_gr = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "MIDAS_GIN"), auc = T, ci = T)

## ensemble model: stacker scale meta probs RUS=1.0 on graph held-out set
roc_ens = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "MIDAS_meta_learner"), auc = T, ci = T)

## OpenTargets: dir, indir, med(dir, indir)
roc_otdir = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "OT_dir"), auc = T, ci = T)
roc_otindir = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "OT_indir"), auc = T, ci = T)
roc_ot_med = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "OT_median"), auc = T, ci = T)


## TargetDB: tractability, MPO default
roc_targdb_tract = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "TargetDB_tractability"), auc = T, ci = T)
roc_targdb_mpo = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "TargetDB_MPO"), auc = T, ci = T)


## CRISPR co-cultures: min(pos), min(neg)
roc_crisprco_pos = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "CRISPRco_pos"), auc = T, ci = T)
roc_crisprco_neg = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "CRISPRco_neg"), auc = T, ci = T)


## DepMap: min
roc_depmap_min = roc(formula = targ_stat ~ pred, data = preds_all %>% filter(model == "DepMap"), auc = T, ci = T)


### get number of genes for each method
grep("roc_", ls(), value = T)

head(roc_crisprco_neg$original.predictor)
head(roc_crisprco_neg$original.response)
table(roc_crisprco_neg$original.response)

length(roc_crisprco_neg$cases)
length(roc_crisprco_neg$controls)

length(roc_crisprco_neg$original.predictor) == length(roc_crisprco_neg$cases) + length(roc_crisprco_neg$controls)
sum(table(roc_crisprco_neg$original.response)) == length(roc_crisprco_neg$original.predictor)


## get numbers
crispr_neg = length(roc_crisprco_neg$original.predictor)
crispr_pos = length(roc_crisprco_pos$original.predictor)

depmap = length(roc_depmap_min$original.predictor)

ens = length(roc_ens$original.predictor)
gr = length(roc_gr$original.predictor)

ot_med = length(roc_ot_med$original.predictor)
otdir = length(roc_otdir$original.predictor)
otindir = length(roc_otindir$original.predictor)

targdb_mpo = length(roc_targdb_mpo$original.predictor)
targdb_tract = length(roc_targdb_tract$original.predictor)



### compare roc results
roc_objs = list(roc_ens, roc_otdir, roc_otindir, roc_ot_med, roc_targdb_tract, roc_targdb_mpo, roc_crisprco_pos, roc_crisprco_neg, roc_depmap_min)
benchmarks = data.frame(alt_methods = c("meta learner", "OpenTargets\ndirect", "OpenTargets\nindirect", "OpenTargets\nmedian", "targetDB\ntractability", "targetDB MPO", "CRISPRco pos", "CRISPRco neg", "DepMap"),
                        roc_auc = NA, lowerCI = NA, upperCI = NA, htest_method = NA, pval = NA)

for (i in 1:9) {
  set.seed(0)
  curr_test = pROC::roc.test(roc_gr, roc_objs[[i]])
  benchmarks[i, "roc_auc"] = roc_objs[[i]]$auc
  benchmarks[i, "lowerCI"] = roc_objs[[i]]$ci[1]
  benchmarks[i, "upperCI"] = roc_objs[[i]]$ci[3]
  benchmarks[i, "htest_method"] = curr_test$method
  benchmarks[i, "pval"] = curr_test$p.value
}

benchmarks$padj = p.adjust(p = benchmarks$pval, method = "fdr")

## plot - put a bar to sort them by method type: dry lab, wet lab
head(benchmarks)

benchmarks$Significance = NA
benchmarks[benchmarks$padj < 0.05, "Significance"] = "FDR<0.05"
benchmarks[benchmarks$padj >= 0.05, "Significance"] = "ns"
benchmarks$padj_str <- paste0("FDR=", sprintf("%.3f", round(benchmarks$padj, digits = 3)))
benchmarks[benchmarks$padj_str == "FDR=0.000", "padj_str"] <- "<0.001"

benchmarks = rbind(data.frame(alt_methods = "GIN",  # need to add the other columns here
                              roc_auc = roc_gr$auc, 
                              lowerCI = roc_gr$ci[1], 
                              upperCI = roc_gr$ci[3], 
                              htest_method = NA, pval = "", padj = NA, Significance = "", padj_str = ""),
                   benchmarks)

benchmarks$type = factor(c(rep("Dry Lab", 7), rep("Wet Lab", 3)), levels = c("Wet Lab", "Dry Lab"))

benchmarks$alt_methods = factor(benchmarks$alt_methods, 
                                levels = benchmarks[order(benchmarks$type, benchmarks$roc_auc, decreasing = F), "alt_methods"])

ggplot(data = benchmarks, 
       mapping = aes(y = alt_methods, x = roc_auc, group = alt_methods, color = alt_methods)) +
  geom_point() +
  geom_errorbarh(mapping = aes(xmin = lowerCI, xmax = upperCI)) +
  geom_text(mapping = aes(y = alt_methods, x = 1, label = signif(padj, digits = 3), angle = 0),
            size = 4.5) +
  scale_color_manual(values = c("MIDAS GIN" = "#EC7557", "meta learner" = "#EC7557", "OpenTargets\ndirect" = "#EFB75E", 
                                "OpenTargets\nindirect" = "#EFB75E", "OpenTargets\nmedian" = "#EFB75E", 
                                "targetDB\ntractability" = "#A17BA0", "targetDB MPO" = "#A17BA0", 
                                "CRISPRco pos" = "#1B687E", "CRISPRco neg" = "#1B687E", "DepMap" = "#93D2BD")) +
  ylab("Target discovery methods") + xlab("Held-out ROC-AUC") +
  theme_bw() +
  theme(legend.position = "none", text = element_text(size = 18))


