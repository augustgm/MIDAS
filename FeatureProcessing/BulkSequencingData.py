#######################################
# Functions to load and aggregate bulk data by response status
# for GNN node features
########################################
import pickle
import os
import pandas as pd
import numpy as np


def load_process_mutation_resp_data(DATABASE_PATH, FILE_NAME, mapped_genes_path):
    def agg_mut_by_resp_normalise(x):
        return np.sum(x) / len(x)

    # Read in data - mutDataOrdinal_tmb_extrectNotGCcorrected_logQuantNorm.csv
    mut_data = pd.read_csv(fr"{DATABASE_PATH}{FILE_NAME}")
    with open(f"{mapped_genes_path}OrdMut_GeneNames_Latest.pkl", "rb") as f:
        mappings = pickle.load(f)
    mappings = mappings.rename(columns={"Unnamed: 0": "index"}).set_index("index")

    # Define cols to exclude
    non_gene_cols = ['B-cells', 'CD45', 'CD8 T cells', 'Cytotoxic cells', 'DC', 'Exhausted CD8', 'Macrophages',
                     'Mast cells', 'Neutrophils', 'NK CD56dim cells', 'NK cells', 'T-cells', 'Th1 cells', 'Treg',
                     'tumour.tcell.fraction', 'normal.tcell.fraction']
    mut_data_map = mut_data.drop(columns=non_gene_cols)

    # Perform mapping
    mappings = mappings.loc[(~mappings["Input"].isna()) |
                            (~mappings["Output"].isna())].reset_index().drop(columns="index")
    mut_data_map = mut_data_map.set_index("Patient_ID").transpose().reset_index().rename(columns={"index": "gene"})
    mappings = pd.concat([mappings,
                          pd.DataFrame({"Input": "Response_responder", "Output": "Response_responder", "Matched": "Yes"},index=[0])],
                         axis=0).reset_index().drop(columns="index")

    mut_data_map = mut_data_map.merge(right=mappings.loc[(mappings["Matched"] != "No Hugo ID match")],
                                      right_on="Input", how="inner", left_on="gene")

    # Tidy data frame and aggregate by response status (count in R and NR normalised by number of R and NR)
    mut_data_map = mut_data_map.drop(columns=["Input", "Matched", "Source", "gene"]).rename(columns={"Output": "gene"}).set_index("gene").transpose()

    # summing the ordinal values so that scores of 2 (LoF) will be counted as higher than scores of 1 (missense)
    mut_data_map = mut_data_map.groupby("Response_responder").apply(agg_mut_by_resp_normalise).\
        drop(columns="Response_responder").rename(index={0.0: "norm_count_NR", 1.0: "norm_count_R"})
    return mut_data_map


def load_process_bulk_rna_resp_data(DATABASE_PATH, FILE_NAME, mapped_genes_path):
    # Read in data - quantile_norm_lm_study_tissueType_resid_v2.csv
    rna_data = pd.read_csv(fr"{DATABASE_PATH}{FILE_NAME}")
  
    with open(f"{mapped_genes_path}BulkRNAseq_GeneNames.pkl", "rb") as f:
        mappings = pickle.load(f)
    mappings = mappings.rename(columns={"Unnamed: 0": "index"}).set_index("index")

    # Process clinical annotations
    clin_data = pd.read_csv(fr"{DATABASE_PATH}master_sample_sheet6.csv")
    clin_data = clin_data.loc[~clin_data["Response_responder"].isna()]

    # Perform mapping
    mappings = mappings.loc[(~mappings["Input"].isna()) |
                            (~mappings["Output"].isna())].reset_index().drop(columns="index")
    rna_data_map = rna_data.set_index("Patient").transpose().reset_index().rename(columns={"index": "gene"})
    rna_data_map = rna_data_map.merge(right=mappings.loc[(mappings["Matched"] != "No Hugo ID match")],
                                      right_on="Input", how="inner", left_on="gene")

    # Tidy data frame
    rna_data_map = rna_data_map.drop(columns=["Input", "Matched", "Source", "gene"]).rename(columns={"Output": "gene"}).set_index("gene").transpose()

    # Annotate with response data
    rna_data_map = rna_data_map.reset_index().rename(columns={"index": "Patient"})
    rna_data_map = rna_data_map.merge(right=clin_data[["Patient", "Response_responder"]], on="Patient", how="inner")
    rna_data_map.set_index("Patient", inplace=True)

    # Aggregate by response status (median in R and NR)
    rna_data_map = rna_data_map.groupby("Response_responder").median().rename(index={"no_response": "median_exp_NR", "response": "median_exp_R"})
    return rna_data_map


def load_process_cna_logR_resp_data(DATABASE_PATH, FILE_NAME, mapped_genes_path, long_format=False):
    # Read in data - cna_data_genelevel_imp_logR.csv
    cna_data = pd.read_csv(fr"{DATABASE_PATH}{FILE_NAME}", index_col=0)
  
    with open(f"{mapped_genes_path}BulkRNAseq_GeneNames.pkl", "rb") as f:
        mappings = pickle.load(f)
    mappings = mappings.rename(columns={"Unnamed: 0": "index"}).set_index("index")
    clin_data = pd.read_csv(fr"{DATABASE_PATH}master_sample_sheet6.csv")

    # Process clinical annotations
    clin_data = clin_data.loc[~clin_data["Response_responder"].isna()]

    # Perform mapping
    mappings = mappings.loc[(~mappings["Input"].isna()) |
                            (~mappings["Output"].isna())].reset_index().drop(columns="index")
    cna_data_map = cna_data.transpose().reset_index().rename(columns={"index": "gene"})
    cna_data_map = cna_data_map.merge(right=mappings.loc[(mappings["Matched"] != "No Hugo ID match")],
                                      right_on="Input", how="inner", left_on="gene")

    # Tidy data frame
    cna_data_map = cna_data_map.drop(columns=["Input", "Matched", "Source", "gene"]).rename(columns={"Output": "gene"}).set_index("gene").transpose()

    # Annotate with response data
    cna_data_map = cna_data_map.reset_index().rename(columns={"index": "Patient"})
    cna_data_map = cna_data_map.merge(right=clin_data[["Patient", "Response_responder"]], on="Patient", how="inner")
    cna_data_map.set_index("Patient", inplace=True)

    # Aggregate by response status (median logR for R and NR)
    cna_data_map = cna_data_map.groupby("Response_responder").median().rename(index={"no_response": "median_logR_NR", "response": "median_logR_R"})
    return cna_data_map


def load_topspec_scores(path_list, MAPPED_GENES_PATH):
    # Read protein coding genes and remove mitochondrial genes
    mapped_genes = pd.read_csv(f"{MAPPED_GENES_PATH}scRNAseq_genes_mappedHUGOids_v2.csv")
    pc_genes = mapped_genes.loc[mapped_genes["locus_group"] == "protein-coding gene"]

    # List topological specificity score files
    banned_cells = ["0_Alveolar_Mac", "9_Unknown_Mac", "17_Normal_Mac_2", "5_Normal_Mac", "16_Myeloid+Tcell",
                    "T_cDC2_Doublets", "mixed_T", "Non-T", "Dying"]

    if return_wide:
        output = pd.DataFrame()
    else:
        output = pd.DataFrame()

    # Iterate through each cell supertype and identify topological specificity scores
    for file_path in path_list:
        tmp_files = os.listdir(file_path)
        tmp_files = [curr_file for curr_file in tmp_files if (("SCT" in curr_file) and ("noBatchCorr" in curr_file) and ("topospec" in curr_file))]
        tmp_files = [curr_file for curr_file in tmp_files if (curr_file.split("_topospec")[0]) not in banned_cells]

        # Read, map gene IDs, and concatenate topological scores
        for this_file in tmp_files:
            tmp_spec = pd.read_csv(f"{file_path}{this_file}")

            # Subset to protein coding genes
            tmp_spec = tmp_spec.merge(right=pc_genes[["sc_genes", "HUGO_approved_symbol"]],
                                      left_on="gene", right_on="sc_genes", how="inner")

            if len(tmp_spec["HUGO_approved_symbol"].unique()) == len(tmp_spec):
                tmp_spec.drop(columns=["gene", "sc_genes"], inplace=True)
                if return_wide:
                    tmp_spec.set_index("HUGO_approved_symbol", inplace=True)
                    tmp_spec.rename(columns={"topo_spec": f'{this_file.split("_topospec")[0]}_topo_spec'}, inplace=True)
                else:
                    tmp_spec["celltype"] = this_file.split("_topospec")[0]
                    tmp_spec.rename(columns={"HUGO_approved_symbol": "gene"}, inplace=True)
            else:
                raise Exception("Duplicate HUGO approved symbols")

            # Concatenate results
            output = pd.concat([output, tmp_spec.transpose()], axis=0)
    return output

