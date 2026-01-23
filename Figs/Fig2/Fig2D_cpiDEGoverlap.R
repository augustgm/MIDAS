#########################################
# Number of cPI3000 DEGs in MIDAS top predictions
#########################################
library(tidyverse)
library(readxl)
library(ggplot2)

rm(list = ls())


## define variables 
pan_str = "Pan\n(n=658)"
lung_str = "Lung\n(n=380)"
renal_str = "Renal\n(n=136)" 
bladder_str = "Bladder\n(n=142)"


## read in data for random overlaps with DEGs
plot_df = readxl::read_excel(path = "Fig2_SourceData.xlsx",
                             sheet = "Fig2D_randomOverlapDEGs") %>%
  as.data.frame()

unique(plot_df$variable)
plot_df[plot_df$variable == "Pan\r\r\n(n=658)", "variable"] = pan_str
plot_df[plot_df$variable == "Lung\r\r\n(n=380)", "variable"] = lung_str
plot_df[plot_df$variable == "Renal\r\r\n(n=136)", "variable"] = renal_str
plot_df[plot_df$variable == "Bladder\r\r\n(n=142)", "variable"] = bladder_str


## read in data for MIDAS overlaps with DEGs
midas_overlap = readxl::read_excel(path = "Fig2_SourceData.xlsx",
                                   sheet = "Fig2D_MIDAS200_overlapDEGs") %>%
  as.data.frame()

unique(midas_overlap$variable)
midas_overlap[midas_overlap$variable == "Pan\r\r\n(n=658)", "variable"] = pan_str
midas_overlap[midas_overlap$variable == "Lung\r\r\n(n=380)", "variable"] = lung_str
midas_overlap[midas_overlap$variable == "Renal\r\r\n(n=136)", "variable"] = renal_str
midas_overlap[midas_overlap$variable == "Bladder\r\r\n(n=142)", "variable"] = bladder_str


## empirical p-values
empirical_pval = function(obs_val, null_dist) {
  return(length(which(null_dist >= obs_val)) / length(null_dist))
}


pvals = data.frame(Pan = empirical_pval(obs_val = midas_overlap[midas_overlap$variable == pan_str, "value"], null_dist = plot_df[plot_df$variable == pan_str, "value"]),
                   Lung = empirical_pval(obs_val = midas_overlap[midas_overlap$variable == lung_str, "value"], null_dist = plot_df[plot_df$variable == lung_str, "value"]), 
                   Renal = empirical_pval(obs_val = midas_overlap[midas_overlap$variable == renal_str, "value"], null_dist = plot_df[plot_df$variable == renal_str, "value"]),
                   Bladder = empirical_pval(obs_val = midas_overlap[midas_overlap$variable == bladder_str, "value"], null_dist = plot_df[plot_df$variable == bladder_str, "value"]))

pvals = reshape2::melt(pvals)
pvals$variable = as.character(pvals$variable)
pvals[pvals$variable == "Renal", "variable"] = renal_str
pvals[pvals$variable == "Bladder", "variable"] = bladder_str
pvals[pvals$variable == "Lung", "variable"] = lung_str  
pvals[pvals$variable == "Pan", "variable"] = pan_str 


## prepare for plotting
plot_df$variable = factor(plot_df$variable, 
                          levels = c(pan_str, lung_str, bladder_str, renal_str))

pvals$variable = factor(pvals$variable, levels = levels(plot_df$variable))

ggplot(data = plot_df, mapping = aes(x = variable, y = value, fill = variable)) +
  geom_col(data = midas_overlap,
             mapping = aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(fill = "white") +
  geom_text(data = pvals, mapping = aes(x = variable, y = 54, size = 10, 
                                        label = paste0("empirical,\np = ", value))) +
  ylab("Number of DEGs") + xlab("Cancer type") +
  scale_fill_manual(values = c("Pan\n(n=658)" = "#EDC2A4", 
                               "Lung\n(n=380)" = "#F68B6B",  
                               "Renal\n(n=136)" = "#F3C986",
                               "Bladder\n(n=142)" = "#F1BB7F")) + 
  scale_colour_manual(values = c("Pan\n(n=658)" = "#EDC2A4",   
                                 "Lung\n(n=380)" = "#F68B6B",   
                                 "Renal\n(n=136)" = "#F3C986",
                                 "Bladder\n(n=142)" = "#F1BB7F")) + 
  theme_bw() +  
  theme(text = element_text(size = 20), legend.position = "none")

