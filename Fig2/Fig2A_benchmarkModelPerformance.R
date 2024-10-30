##############################################################
# Benchmark held-out MIDAS performances against alternatives #
##############################################################
library(pROC)
library(ggplot2)
library(ggpubr)


rm(list = ls())

## define path stem 
path_stem = ""


## graph model: GIN KNN=50
roc_gr = readRDS(paste0(path_stem, "GINBiDir_KNN50_houtROCAUC.rds"))  

## ensemble model: stacker scale meta probs RUS=1.0 on graph held-out set
roc_ens = readRDS(paste0(path_stem, "ensembleStackerRUS1.0consistentWithGraph_ginhoutROCAUC.rds"))  


## OpenTargets: dir, indir, med(dir, indir)
roc_otdir = readRDS(paste0(path_stem, "OpenTargets_dirCanOverallScore_houtROCAUC_rocobj.rds"))
roc_otindir = readRDS(paste0(path_stem, "OpenTargets_indirCanOverallScore_houtROCAUC_rocobj.rds"))
roc_ot_med = readRDS(paste0(path_stem, "OpenTargets_medDirIndirCanOverallScore_houtROCAUC_rocobj.rds"))


## TargetDB: tractability, MPO default
roc_targdb_tract = readRDS(paste0(path_stem, "targetDB_tractProba_houtROCAUC_rocobj.rds"))
roc_targdb_mpo = readRDS(paste0(path_stem, "targetDB_defaultMPO_houtROCAUC_rocobj.rds"))


## CRISPR co-cultures: min(pos), min(neg)
roc_crisprco_pos = readRDS(paste0(path_stem, "inputCRISPRcoCulturesPosMed_houtROCAUC_rocobj.rds"))
roc_crisprco_neg = readRDS(paste0(path_stem, "inputCRISPRcoCulturesNegMed_houtROCAUC_rocobj.rds"))

## DepMap: min
roc_depmap_min = readRDS(paste0(path_stem, "DepMapCRISPR_houtROCAUC_rocobj.rds"))


### compare roc results
roc_objs = list(roc_ens, roc_otdir, roc_otindir, roc_ot_med, roc_targdb_tract, roc_targdb_mpo, roc_crisprco_pos, roc_crisprco_neg, roc_depmap_min)

benchmarks = data.frame(alt_methods = c("ensemble", "OpenTargets_dir", "OpenTargets_indir", "OpenTargets_med", "targetDB_tract", "targetDB_mpo", "CRISPRco_pos", "CRISPRco_neg", "DepMap"),
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
  ylab("Target discovery methods") + xlab("Held-out ROC-AUC") +
  theme_bw() +
  theme(legend.position = "none", text = element_text(size = 18))

