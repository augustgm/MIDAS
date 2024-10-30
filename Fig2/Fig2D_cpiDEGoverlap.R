#########################################
# Number of cPI3000 DEGs in MIDAS top predictions
#########################################
library(tidyverse)
library(DESeq2)
library(ggplot2)

## define top x genes
top_x = 200
graph_mod = T

## read in DEG analysis
dds_pan = readRDS("CPI3000newStudies_DESeq2_DEGs_allCohorts_designStudySexResponse.rds")
dds_lung = readRDS("CPI3000newStudies_DESeq2_DEGs_LUNGCohorts_designStudySexResponse.rds")
dds_renal = readRDS("CPI3000newStudies_DESeq2_DEGs_RENALCohorts_designStudySexResponse.rds")
dds_bladder = readRDS("CPI3000newStudies_DESeq2_DEGs_BLADDERCohorts_designSexResponse.rds")

## extract DEGs
res_pan <- results(dds_pan, name = "response_responder_response_vs_no_response") %>% as.data.frame()
res_lung <- results(dds_lung, name = "response_responder_response_vs_no_response") %>% as.data.frame()
res_renal <- results(dds_renal, name = "response_responder_response_vs_no_response") %>% as.data.frame()
res_bladder <- results(dds_bladder, name = "response_responder_response_vs_no_response") %>% as.data.frame()

res_pan$genes = rownames(res_pan)
res_lung$genes = rownames(res_lung)
res_renal$genes = rownames(res_renal)
res_bladder$genes = rownames(res_bladder)

all_genes = res_pan$genes

res_pan = res_pan %>% filter(!(is.na(padj))) %>% filter(padj < 0.05) %>% as.data.frame()
res_lung = res_lung %>% filter(!(is.na(padj))) %>% filter(padj < 0.05) %>% as.data.frame()
res_renal = res_renal %>% filter(!(is.na(padj))) %>% filter(padj < 0.05) %>% as.data.frame()
res_bladder = res_bladder %>% filter(!(is.na(padj))) %>% filter(padj < 0.05) %>% as.data.frame()

## read in MIDAS predictions
midas_preds = read.csv(header = T, stringsAsFactors = F, file = "GiG_NeighbourLoaderKNN50_cvTestHout_predProba.csv")

midas_preds = midas_preds %>%
  mutate(DEG_pan = ifelse(Gene %in% res_pan$genes, yes = "DEG", no = "non-DEG")) %>%
  mutate(DEG_lung = ifelse(Gene %in% res_lung$genes, yes = "DEG", no = "non-DEG")) %>%
  mutate(DEG_renal = ifelse(Gene %in% res_renal$genes, yes = "DEG", no = "non-DEG")) %>%
  mutate(DEG_bladder = ifelse(Gene %in% res_bladder$genes, yes = "DEG", no = "non-DEG")) %>%
  as.data.frame()

deg_n_pan = table(midas_preds$DEG_pan[1:top_x])[1]
deg_n_lung = table(midas_preds$DEG_lung[1:top_x])[1] 
deg_n_renal = table(midas_preds$DEG_renal[1:top_x])[1]  
deg_n_bladder = table(midas_preds$DEG_bladder[1:top_x])[1]  


#### Random gene sets stratified by expression ####
empirical_pval = function(obs_val, null_dist) {
  return(length(which(null_dist >= obs_val)) / length(null_dist))
}


avg_cancer_spec_expression = function(dds_obj) {
  rna_count = dds_obj@assays@data@listData$counts
  rna_avg = data.frame(genes = rownames(rna_count), mean_exp = matrixStats::rowMeans2(rna_count))
return(rna_avg) }
  
pan_rna_avg = avg_cancer_spec_expression(dds_obj = dds_pan)
lung_rna_avg = avg_cancer_spec_expression(dds_obj = dds_lung)
bladder_rna_avg = avg_cancer_spec_expression(dds_obj = dds_bladder)
renal_rna_avg = avg_cancer_spec_expression(dds_obj = dds_renal)

output = data.frame()
for (curr_seed in 1:1e3) {
  
  ## random gene set stratified by expression in each cancer-type rather than at pan-can level
  set.seed(curr_seed)
  pan_gene_set = sample(x = all_genes, size = top_x, replace = F, prob = pan_rna_avg$mean_exp)
  
  set.seed(curr_seed)
  lung_gene_set = sample(x = all_genes, size = top_x, replace = F, prob = lung_rna_avg$mean_exp)
  
  set.seed(curr_seed)
  renal_gene_set = sample(x = all_genes, size = top_x, replace = F, prob = renal_rna_avg$mean_exp)
  
  set.seed(curr_seed)
  bladder_gene_set = sample(x = all_genes, size = top_x, replace = F, prob = bladder_rna_avg$mean_exp)
  
  ## compute overlap
  tmp = data.frame(iter = curr_seed,
                   Pan = length(dplyr::intersect(pan_gene_set, res_pan$genes)),
                   Lung = length(dplyr::intersect(lung_gene_set, res_lung$genes)), 
                   Renal = length(dplyr::intersect(renal_gene_set, res_renal$genes)),
                   Bladder = length(dplyr::intersect(bladder_gene_set, res_bladder$genes)))
  
  ## concatenate results
  output = rbind(output, tmp)
}
plot_df = reshape2::melt(output, id.vars = "iter")
pvals = data.frame(Pan = empirical_pval(obs_val = deg_n_pan, null_dist = output$Pan),
                   Lung = empirical_pval(obs_val = deg_n_lung, null_dist = output$Lung), 
                   Renal = empirical_pval(obs_val = deg_n_renal, null_dist = output$Renal),
                   Bladder = empirical_pval(obs_val = deg_n_bladder, null_dist = output$Bladder))

pvals = reshape2::melt(pvals)
pvals$variable = as.character(pvals$variable)
pvals[pvals$variable == "Renal", "variable"] = "Renal\n(n=133)"
pvals[pvals$variable == "Bladder", "variable"] = "Bladder\n(n=187)"
pvals[pvals$variable == "Lung", "variable"] = "Lung\n(n=485)"
pvals[pvals$variable == "Pan", "variable"] = "Pan\n(n=805)"

plot_df$variable = as.character(plot_df$variable)
plot_df[plot_df$variable == "Renal", "variable"] = "Renal\n(n=133)"
plot_df[plot_df$variable == "Bladder", "variable"] = "Bladder\n(n=187)"
plot_df[plot_df$variable == "Lung", "variable"] = "Lung\n(n=485)"
plot_df[plot_df$variable == "Pan", "variable"] = "Pan\n(n=805)"

plot_df$variable = factor(plot_df$variable, levels = c("Lung\n(n=485)", "Pan\n(n=805)", "Bladder\n(n=187)", "Renal\n(n=133)"))
annot_points = data.frame(variable = c("Pan\n(n=805)", "Lung\n(n=485)", "Renal\n(n=133)", "Bladder\n(n=187)"),
                          value = c(deg_n_pan, deg_n_lung, deg_n_renal, deg_n_bladder))
annot_points$variable = factor(annot_points$variable, levels = levels(plot_df$variable))
pvals$variable = factor(pvals$variable, levels = levels(plot_df$variable))

ggplot(data = plot_df, mapping = aes(x = variable, y = value, fill = variable)) +
  geom_col(data = data.frame(variable = c("Pan\n(n=805)", "Lung\n(n=485)", "Renal\n(n=133)", "Bladder\n(n=187)"),
                             value = c(deg_n_pan, deg_n_lung, deg_n_renal, deg_n_bladder)),
             mapping = aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(fill = "white") +
  geom_text(data = pvals, mapping = aes(x = variable, y = 54, size = 10, 
                                        label = paste0("empirical,\np = ", value))) +
  ylab("Number of DEGs") + xlab("Cancer type") +
  scale_fill_manual(values = c("Pan\n(n=805)" = "#EDC2A4", 
                               "Lung\n(n=485)" = "#F68B6B",  
                               "Renal\n(n=133)" = "#F3C986",
                               "Bladder\n(n=187)" = "#F1BB7F")) + 
  scale_colour_manual(values = c("Pan\n(n=805)" = "#EDC2A4",   
                                 "Lung\n(n=485)" = "#F68B6B",   
                                 "Renal\n(n=133)" = "#F3C986",
                                 "Bladder\n(n=187)" = "#F1BB7F")) + 
  theme_bw() +  theme(text = element_text(size = 20), legend.position = "none")
