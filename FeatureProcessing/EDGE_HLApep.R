###############################################################
# Explore HLA-peptidomic association with drug target status #
###############################################################
library(readxl)
library(dplyr)
library(reshape2)
library(data.table)
library(ggplot2)
library(ggpubr)


## Read in data files
ms_peptides = read.csv(file = "Bulik-Sullivan_Combined_MS_peptides.csv", header = T, stringsAsFactors = F)

sample_info = read_excel("Bulik-Sullivan_supplData1_sample_characteristics_ms_ngs_metrics.xlsx") %>% as.data.frame()

# Mapped Ensembl IDs to HGNC symbols 
rna_expression = read.csv(file = "Bulik-Sullivan_RNA_expression.csv", header = T, stringsAsFactors = F)

## Obtain sample level information
cancer_samples = sample_info[!(sample_info$`Tissue Specimen Type` %in% c("Normal", "Adjacent Normal")), "paper_sample_id"]

## Melt df to get column of each patients' HLA type
num_epitopes_per_pat = reshape2::melt(sample_info, measure.vars = c("A1", "A2", "B1", "B2", "C1", "C2"), 
                                      variable.name = "HLA_allele")
colnames(num_epitopes_per_pat)[length(num_epitopes_per_pat)] <- "HLA_type"

## summarise transcript counts to gene-level by summing across transcripts ##
# remove version number from gene_id (after decimal point)
rna_expression$gene_id_pure = unlist(lapply(strsplit(rna_expression$gene_id, split = "\\."), FUN = function(x) {x[1]}))

# summarise by summing across all transcripts that map to the same gene
rna_exp_sum = rna_expression[, !(colnames(rna_expression) %in% c("transcript_id", "gene_id"))] %>% 
  group_by(gene_id_pure) %>%
  summarise(across(everything(), sum)) %>%
  as.data.frame()

# melt data frame 
rna_sum_melt = reshape2::melt(rna_exp_sum, id.vars = c("gene_id_pure"), 
                              variable.name = "sample_id", value.name = "exp")
rna_sum_melt$sample_id = as.character(rna_sum_melt$sample_id)
rna_sum_melt$paper_sample_id = tolower(gsub(pattern = "\\.", replacement = "_", x = rna_sum_melt$sample_id))

## merge with sample info to get number of epitopes
rna_sum_melt = merge(x = rna_sum_melt, 
                     y = dplyr::distinct(num_epitopes_per_pat[, c("paper_sample_id", "peptides_q01", "peptides_q05", "peptides_q10",
                                                                  "Tissue Type", "Tissue Specimen Type", "Tumor Subtype")]), 
                     by = "paper_sample_id")

## Compute correlation
exp_hla_corr = rna_sum_melt %>%
  group_by(gene_id_pure) %>%
  summarise(pep01_corr = cor(peptides_q01, exp),
            pep05_corr = cor(peptides_q05, exp),
            pep10_corr = cor(peptides_q10, exp)) %>%
  as.data.frame()

# save to file 
write.csv(exp_hla_corr, row.names = F, file = "spearcorr_numPresPeptides_summedTranscriptGeneExp_noImputation.csv")
