##############################################################
# Benchmark held-out MIDAS performances against alternatives #
##############################################################
library(ggplot2)
library(ggpubr)

benchmarks = read.csv(header = T, stringsAsFactors = F, file = "Benchmarks_ROCAUCs.csv")

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

