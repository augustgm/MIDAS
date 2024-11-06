#######################
# Compute GSEA on model predictions and plot top 20
#######################
library(tidyverse)
library(ggplot2)
library(WebGestaltR)


# Read in data
gsea = read.csv(header = T, stringsAsFactors = F, file = "GiG_KNN=50_GSEA_webgestaltR_Reactome_supplementary.csv")
colnames(gsea)[colnames(gsea) == "NES"] = "normalizedEnrichmentScore"
gsea = gsea[order(abs(gsea$normalizedEnrichmentScore), decreasing = T), ]

## plot 
colnames(gsea)[colnames(gsea) == "size"] = "set_size"
gsea$col_stat = "pos"
gsea[gsea$normalizedEnrichmentScore < 0, "col_stat"] = "neg"

plot_df = gsea[1:20, ]
plot_df$Description = factor(plot_df$Description,
                             levels = plot_df[order(plot_df$normalizedEnrichmentScore, decreasing = F), "Description"])

ggplot(data = plot_df, 
       mapping = aes(x = normalizedEnrichmentScore, y = Description, fill = col_stat)) +
  geom_col() +
  xlab("Normalised enrichment score") +
  scale_fill_manual(values = c("pos" = "#c08081", "neg" = "#685968")) +
  theme_bw() +
  theme(text = element_text(size = 18), legend.position = "none",
        axis.title.y = element_blank()) 


