################################################################
# Extract SNP-phenotype associations for phenotypes of interest
################################################################
library(tidyr)
library(dplyr)


# Read in gwas catalogue - downloaded from: https://www.ebi.ac.uk/gwas/docs/file-downloads. file = All associations v1.0
gwas_cat = read.csv(file = "gwas_catalog_v1.0-associations_e105_r2022-03-23.csv", header = T, stringsAsFactors = F)

a = data.frame(diseases = unique(gwas_cat$DISEASE.TRAIT))

#### Determine autoimmune/rheumatic/allergic conditions ####
# core list from: Body mass index and risk of autoimmune diseases: a study within the Danish National Birth Cohort, International Journal of Epidemiology, Mar 2014

auto_im <- c("Addison's disease", "Ankylosing spondylitis", "Autoimmune", "Behcet's disease ", "Buerger's syndrome", "Celiac disease", "Crohn's disease",
             "Dermatitis herpetiformis", "Diabetes mellitus", "Dupuytren's disease", "Erythema nodosum", "Goodpasture's syndrome", "Graves' disease", "Frontal fibrosing alopecia", "Alopecia areata",
             "Guillain-Barre syndrome", "Haemolytic anaemia", "Hashimoto thyroiditis", "Henoch-Schonlein purpura", "ITP", "Idiopathic thrombocytopenic purpura",
             "Kawasaki disease", "lupus erythematosus", "scleroderma", "Myasthenia gravis", "Multiple sclerosis$", "pemphigoid", "Pernicious anaemia",
             "Pemphigus", "arteritis", "myositis", "Primary biliary cirrhosis", "Psoriasis", "Rheumatic fever", "Psoriatic arthritis", "Rheumatoid factor seropositivity in rheumatoid arthritis",
             "Rheumatoid arthritis", "rheumatoid arthritis", "Raynaud's phenomenon", "Reiter's disease", "Sarcoidosis", "gren's syndrome$", "Sympathetic ophthalmia", 
             "Systemic lupus erythematosus", "Systemic scleroderma", "Ulcerative colitis", "Vitiligo", "Eosinophilic granulomatosis with polyangiitis",
             "Systemic seropositive rheumatic diseases", "Juvenile idiopathic arthritis", "Uveitis", "Alloimmunization", "Membranous nephropathy", #"Paget's disease",
             "juvenile idiopathic", "Eczema", "Eosinophilic esophagitis", "Food allergy", "Food antigen IgG levels", "Hay fever and/or eczema", "Hydrolysed wheat protein allergy",  # taken from inspecting the unique disease traits
             "Idiopathic inflammatory myopathy", "IgA nephropathy", "Immunoglobulin A vasculitis", "amyloidosis", "Obstetric antiphospholipid syndrome",  # taken from inspecting the unique disease traits
             "Ocular sarcoidosis", "Peanut allergy", "Penicillin allergy", "Recalcitrant atopic dermatitis", "Relapse in multiple sclerosis", "Seborrheic dermatitis", "Systemic sclerosis",  # taken from inspecting the unique disease traits
             "Type 1 diabetes", "Vogt-Koyanagi-Harada syndrome", "Atopic dermatitis", "Contact dermatitis", "Allergic", "allergic",
             "vasculitis", "Vasculitis", "asthma")  # taken from inspecting the unique disease traits

auto_im_dis = data.frame(diseases = grep(paste0(auto_im, collapse = "|"), a$diseases, ignore.case = T, value = T))
length(unique(auto_im_dis$diseases)) == nrow(auto_im_dis)  # TRUE

### Read in auto-immune/rheumatic/allergic features - manually reviewed the phenotypes and decided if they should be included
auto_im_dis = read.csv(stringsAsFactors = F, header = T, file = "autoImmune_allergic_rheumatic_phenotypes.csv")
auto_im_dis = auto_im_dis[auto_im_dis$Keep == 1, ] 

#### Determine immune phenotypes ####
immune_phenotypes = c("Anti-Epstein-Barr virus nuclear antigen",
                      "Basophil count", "Basophil percentage of granulocytes", "Basophil percentage of white cells", "C-reactive protein",
                      "C-X-C motif chemokine 10", "Cerebrospinal fluid immune biomarker levels",  # CSF immune biomarkers may need to be removed
                      "cytokine", "immunoglobulin", "complement", "CXC", "CCL", "CCR", "Ig", "IL", "CD",
                      "Cytomegalovirus antibody response", "E-selectin",  # E-selectin = CD26E = leukocyte-endothelial cell adhesion molecule 2 and is expressed only on endothelial cells activated by cytokines. important in inflammation  
                      "Eosinophil counts", "Epstein-Barr virus immune response", 
                      "lymphocyte", "monocyte", "macrophage", "granulocyte", "basophil", "eosinophil", "natural killer", "leukocyte", "dendritic", 
                      "neutrophil", "White blood", "cell", "myeloid", "lymphoid", "TNF", "antibody",
                      "Ferritin", "Fibrinogen", "Gamma glutamyl transferase", ## GGT levels here as they can be a marker of hepatic inflammation
                      "Hepatitis B vaccine response", "HIV-1 susceptibility",
                      "Interleukin", "Mastocytosis", "interferon-related traits", "Interferon gamma levels", "Interferon gamma-induced protein 10 levels",                                                                                                       
                      "Monokine induced by gamma interferon levels",
                      "Matrix metalloproteinase", "MHC class I polypeptide-related sequence A", "Monocyte chemoattractant protein-1",  # MMP8 levels is produced by neutrophils and associated with many inflammatory conditions
                      "Platelet count", "Resistin", "Serum alkaline phosphatase levels",
                      "Serum immune biomarker levels", "Serum sclerostin levels", "Sphingomyelin",  # hits for sphingolipid being inflammatory on google
                      "Tumor necrosis factor receptor II", "Uridine levels")  # PMID: 26369416 for uridine 

imm_phen_dis = data.frame(diseases = unique(c(grep(paste0(immune_phenotypes[!(immune_phenotypes %in% c("Ig", "IL", "CD", "CCL", "CCR"))], collapse = "|"), a$diseases, ignore.case = T, value = T),
                                              grep(paste0(c("Ig", "IL", "CD", "CCL", "CCR"), collapse = "|"), a$diseases, ignore.case = F, value = T),
                                              grep(paste0(unique($SYMBOL), collapse = "|"), a$diseases, ignore.case = T, value = T))))


## Manually checked which of the entries should be included as immune phenotypes
imm_phen_dis = read.csv(header = T, stringsAsFactors = F, file = "immune_phenotypes_gwas_catalog_v2.csv")
imm_phen_dis[is.na(imm_phen_dis$alt_category), "alt_category"] <- ""
imm_phen_dis[is.na(imm_phen_dis$evidence), "evidence"] <- ""
imm_phen_dis[is.na(imm_phen_dis$alternative_to_grep), "alternative_to_grep"] <- ""

imm_phen_dis = imm_phen_dis[(!(is.na(imm_phen_dis$keep))) & (imm_phen_dis$keep == 1), ]


#### Define immune modulation ####
imm_modulation = c("response", "steroid", "vaccine", "suppress", "glucocorticoid", "NSAID", "steroidal", "immunization", "inhibitor", "agonist", "drug-induced", "mab",
                   "Methotrexate-induced interstitial lung disease in rheumatoid arthritis",
                   "Alanine aminotransferase level after methotrexate initiation in rheumatoid arthritis", "Tacrolimus trough concentration in kidney transplant patients",
                   "Infliximab-induced mucosal healing in Crohn's disease", "Severe prolonged lymphopenia in dimethyl fumarate-treated relapsing-remitting multiple sclerosis",
                   "Moderate or severe prolonged lymphopenia in dimethyl fumarate-treated relapsing-remitting multiple sclerosis", "Drug-induced liver injury in interferon-beta-treated multiple sclerosis",
                   "Frequency of relapse in mepolizumab-treated eosinophilic granulomatosis with polyangiitis", "Average oral glucocorticoid dose in mepolizumab-treated eosinophilic granulomatosis with polyangiitis",
                   "Accrued duration of remission in mepolizumab-treated eosinophilic granulomatosis with polyangiitis",
                   "oral corticosteroid dose interaction", "Aspirin exacerbated respiratory disease in asthmatics", "Anti-drug antibodies in autoimmune disease",
                   "Asthma exacerbations in inhaled corticosteroid treatment", "Asthma control x inhaled corticosteroid treatment interaction",
                   "Thiopurine-induced", "Sulfasalazine-induced agranulocytosis", "Drug-induced liver injury in interferon-beta-treated multiple sclerosis")

imm_mod_resp = data.frame(diseases = grep(paste0(imm_modulation, collapse = "|"), a$diseases, ignore.case = T, value = T))
write.csv(imm_mod_resp, row.names = F,  file = "immune_modulatory_response_gwas_catalog.csv")

## Manually checked which of the entries should be included as immune phenotypes
imm_mod_resp = read.csv(header = T, stringsAsFactors = F, file = "immune_modulatory_response_gwas_catalog.csv")

imm_mod_resp = imm_mod_resp[imm_mod_resp$keep == 1, ]


#### Identify SNPs and genes associated with phenotypes of interest ####
all_imm_phen = unique(c(auto_im_dis$diseases, imm_phen_dis$diseases, imm_mod_resp$diseases))
gwas_imm_phen = gwas_cat[gwas_cat$DISEASE.TRAIT %in% all_imm_phen, ]
rm(all_imm_phen)

## explode the SNP_GENE_IDS column (Entrez ID)
gwas_imm_phen = separate_rows(data = gwas_imm_phen, "SNP_GENE_IDS", sep = ", ") %>% as.data.frame()

## Filter by p-values
gwas_imm_phen = gwas_imm_phen[gwas_imm_phen$P.VALUE < 5e-8, ]  # typical threshold using Bonferroni correction with alpha = 0.05


### blood counts ###
blood_counts = c("lymphocyte", "monocyte", "macrophage", "granulocyte", "basophil", "eosinophil", "natural killer", "leukocyte", "dendritic", 
                 "neutrophil", "White blood", "cell", "myeloid", "lymphoid", "Mastocytosis", "Platelet count") 

blood_counts_phen = data.frame(phenotype = unique(grep(paste0(blood_counts, collapse = "|"), gwas_imm_phen$DISEASE.TRAIT, ignore.case = T, value = T)))
write.csv(blood_counts_phen, row.names = F, file = "blood_counts_phenotypes.csv")

## Manually decided which of the blood counts should be kept
blood_counts_phen = read.csv(stringsAsFactors = F, header = T, file = "blood_counts_phenotypes.csv")
blood_counts_phen = blood_counts_phen[blood_counts_phen$keep == 1, ]


### cytokines ###
cytokines = c("C-reactive protein", "CRP", "cytokine", "CXC", "CCL", "CCR", "IL", "TNF",
              "Interleukin", "Interferon", "Tumor necrosis factor", "C-X-C", "chemokine") 

cytokine_phen = data.frame(phenotype = unique(grep(paste0(cytokines, collapse = "|"), gwas_imm_phen$DISEASE.TRAIT, ignore.case = T, value = T)))

write.csv(cytokine_phen, row.names = F, file = "cytokines_chemokines_phenotypes.csv")

## Manually decided which of the blood counts should be kept
cytokine_phen = read.csv(stringsAsFactors = F, header = T, file = "cytokines_chemokines_phenotypes.csv")
cytokine_phen = cytokine_phen[cytokine_phen$keep == 1, ]


# SNP functional class - exclude purely intergenic SNP
gwas_imm_phen = gwas_imm_phen[!(gwas_imm_phen$CONTEXT %in% c("intergenic_variant", "intergenic_variant x intergenic_variant")), ]

# As you will be mapping these to each gene, remove any rows which are intergenic
gwas_imm_phen = gwas_imm_phen[!(gwas_imm_phen$INTERGENIC %in% c(1)), ]

### Extract unique genes ###
## get unique genes
get_unique_genes_from_gwas_df = function(gwas_df) {
  
  ## concatenate all gene entries into 1 vector
  all_entries = unique(gwas_df$SNP_GENE_IDS)
  
  ## get rid of non-gene entries
  all_genes = unique(all_entries[!(all_entries %in% c("", "NR", "No mapped genes"))])
  return(all_genes)}


#### Check count of SNPs per association ####
get_snp_counts_per_gene <- function(gwas_df, phenotype_vec) {
  gwas_df <- gwas_df[gwas_df$DISEASE.TRAIT %in% phenotype_vec, ]
  
  # get unique genes for the current gene-phenotype associations
  gene_vec <- get_unique_genes_from_gwas_df(gwas_df)
  
  # create output data frame - default value is 0
  snp_counts = data.frame(genes = gene_vec, snps_count = 0)
  
  
  ## Get count of SNPs from single gene SNPs
  for (i in 1:length(gene_vec)) {
    curr_gene <- gene_vec[i]
    curr_data <- gwas_df[(gwas_df$SNP_GENE_IDS == curr_gene), ]
    if (nrow(curr_data) > 0) {
      snp_counts[i, "snps_count"] <- length(unique(curr_data$SNPS))
    }
  }
  return(snp_counts)}


autoim_snp_counts = get_snp_counts_per_gene(gwas_df = gwas_imm_phen, phenotype_vec = auto_im_dis$diseases)
write.csv(autoim_snp_counts, row.names = F, file = "autoimmRheumAllergy_snpCounts_perGWAScatalogEntrezGene.csv")

blood_snp_counts = get_snp_counts_per_gene(gwas_df = gwas_imm_phen, phenotype_vec = blood_counts_phen$phenotype)
write.csv(blood_snp_counts, row.names = F, file = "bloodCounts_snpCounts_perGWAScatalogEntrezGene.csv")

cytokine_snp_counts = get_snp_counts_per_gene(gwas_df = gwas_imm_phen, phenotype_vec = cytokine_phen$phenotype)
write.csv(cytokine_snp_counts, row.names = F, file = "cytokineLevel_snpCounts_perGWAScatalogEntrezGene.csv")
