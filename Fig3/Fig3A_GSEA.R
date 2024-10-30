#######################
# Compute GSEA on model predictions and plot top 20
#######################
library(tidyverse)
library(ggplot2)
library(WebGestaltR)

res_k50 = read.csv(header = T, stringsAsFactors = F, file = "GiG_NeighbourLoaderKNN50_cvTestHout_predProba.csv")

## Compute GSEA
gsea = WebGestaltR(enrichMethod = "GSEA", organism = "hsapiens", enrichDatabase = "pathway_Reactome", 
                   interestGene = res_k50[, c("Gene", "mean_prob")], interestGeneType = "genesymbol",
                   isOutput = F, nThreads = 5, fdrMethod = "BH", fdrThr = 0.05, topThr = 10)

gsea = gsea[order(abs(gsea$normalizedEnrichmentScore), decreasing = T), ]

## plot 
colnames(gsea)[colnames(gsea) == "size"] = "set_size"
gsea$col_stat = "pos"
gsea[gsea$normalizedEnrichmentScore < 0, "col_stat"] = "neg"

plot_df = gsea[1:20, ]
plot_df$description = factor(plot_df$description,
                             levels = plot_df[order(plot_df$normalizedEnrichmentScore, decreasing = F), "description"])

ggplot(data = plot_df, 
       mapping = aes(x = normalizedEnrichmentScore, y = description, fill = col_stat)) +
  geom_col() +
  xlab("Normalised enrichment score") +
  scale_fill_manual(values = c("pos" = "#c08081", "neg" = "#685968")) +
  theme_bw() +
  theme(text = element_text(size = 18), legend.position = "none",
        axis.title.y = element_blank()) 


