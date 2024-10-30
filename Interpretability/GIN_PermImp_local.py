#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#################################################################################################
# Perform degree-preserving permutation edge importance to explain individual genes of interest #
#     Permutes edges connecting genes involved in pathways relevant to genes being explained    #
#                              on optimised, best GIN model                                     #
#                       Written by Marcellus Augustine on 15/04/2024                            #
#################################################################################################
# Standard Libraries
import pickle
import os
import sys

# Third-party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import optuna
import xswap  

# Imblearn Libraries
from imblearn.under_sampling import RandomUnderSampler

# Scikit-learn Libraries
from sklearn.model_selection import RepeatedStratifiedKFold

# PyTorch Libraries
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.loader import NeighborLoader

############################################################################################################
# Filenames, Global Variables and Parameters
############################################################################################################
location = "cs_cluster"
KNN = 50 
fold_to_start_from = "None"  # which CV fold to start from ("None" to start from beginning)
des_set = "sub_targs"

# which seed to start from for rewiring ("None to start from beginning"). will go from this to nearest 100
seed_to_start_from = None 

# gene index to start from (0 to start from beginning)
gene_start_ind = 0 

# Process fold to start from
if fold_to_start_from == "None":
    fold_to_start_from = 0
else:
    fold_to_start_from = int(fold_to_start_from)

# Process seed to start from
if seed_to_start_from == "None":
    seed_to_start_from = 0
else:
    seed_to_start_from = int(seed_to_start_from)
end = seed_to_start_from + 100

# Define paths
DATABASE_PATH = "MIDAS_analysis/MIDAS_database/"
OPTUNA_DATABASE_PATH = f"optuna_db/GINBiDir/"
MODEL_PATH = f"MIDAS_analysis/GraphModels/models/GINBiDir/"

# Node classification data - used for split indices 
DATA_FILE_TRAIN = 'train_imputed_exclMissCrispr_crispr_gsnp_hlapep_autoimRheum_scElnetBulkXgbShap.csv'
DATA_FILE_TEST = 'test_imputed_exclMissCrispr_crispr_gsnp_hlapep_autoimRheum_scElnetBulkXgbShap.csv'

# Number of epochs
EPOCHS = int(os.getenv("EPOCHS", 100))

# Study name
STUDY_NAME = f'GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_GiG_KNN{KNN}'

# Optuna file name
TRAINING_DATABASE_FILE = f'sqlite:///{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db'  


##########################################################
# Prepare pathways for genes of interest for permutation #
##########################################################
# Read in original predictions from each fold model
all_preds = pd.read_csv(f"{MODEL_PATH}predictions/Predictions_Best_GINBiDir_WithCVModels_GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_GiG_KNN50.csv")
all_preds = all_preds.loc[all_preds["Set"] != "Train (fold)"]

# Read in gene set and manually curated pathway database
gene_set = pd.read_csv(f'{DATABASE_PATH}SubmittedCandTargs_manCurateReactomePathways.csv')
gene_set.rename(columns={"gene": "gene_expl"}, inplace=True)

pathways = pd.read_csv(f'{DATABASE_PATH}CTD_genes_pathways_reactome.csv')
pathways.rename(columns={pathways.columns[0]: "gene"}, inplace=True)
pathways = pathways.loc[pathways["PathwayID"].str.contains("REACT")]

# Merge to annotate both target gene (to be explained) and other genes involved in each pathway
gene_set = gene_set.merge(right=pathways.drop(columns=["PathwayName"]), on="PathwayID", how="inner")
genes_oi = gene_set["gene_expl"].unique().tolist()

############################################################################################################
# GNN class model
############################################################################################################
class GNN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate,
                 eps, train_eps):

        super(GNN, self).__init__()

        # Method for preprocessing/postprocessing layers
        def create_mlp_layers(n_layers, n_in, n_out):
            layers = [Sequential(Linear(n_in, n_out), ReLU(), BatchNorm1d(n_out))]
            for _ in range(n_layers - 1):
                layers.append(Sequential(Linear(n_out, n_out), ReLU(), BatchNorm1d(n_out)))
            return layers

        self.preprocess = nn.ModuleList(
            create_mlp_layers(num_layers_pre, n_features, n_hidden)
        )

        # Create GINConv layers
        conv_mult = [2 if i < (num_layers - 1) else 1 for i in range(num_layers)]
        self.convs = nn.ModuleList([
            GINConv(Linear(n_hidden * (conv_mult[i - 1] if i > 0 else 1), n_hidden * mult), eps=eps,
                    train_eps=train_eps) for i, mult in enumerate(conv_mult)

        ])
        self.conv_norms = nn.ModuleList([BatchNorm1d(n_hidden * mult) for mult in conv_mult])
        self.conv_relu = nn.ModuleList([ReLU() for _ in range(num_layers)])
        self.postprocess = nn.ModuleList(
            create_mlp_layers(num_layers_post, n_hidden, n_hidden)
        )
        max_layers = max(num_layers_pre, num_layers, num_layers_post)
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout * (rate ** i)) for i in range(max_layers)])
        self.classifier = Linear(n_hidden, n_classes)

    # def forward(self, x, edge_index, edge_weight):
    def forward(self, x, edge_index):

        # Helper method for applying sequential layers and dropout
        def apply_layers_with_dropout(layers, xx):
            for i, layer in enumerate(layers):
                xx = layer(xx)
                xx = self.dropouts[i](xx)
            return xx

        x = apply_layers_with_dropout(self.preprocess, x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.conv_norms[i](x)
            x = self.conv_relu[i](x)
            x = self.dropouts[i](x)

        x = apply_layers_with_dropout(self.postprocess, x)
        x = self.classifier(x)
        return x


############################################################################################################
# Functions that calculates importances for each fold model and node in the test fold and ht data set
############################################################################################################
def create_neighbor_loader(data, num_layers, mask, xx, n_b, nodes_list):
    return NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,
        batch_size=sum([True if i in nodes_list else False for i in xx.index]),
        input_nodes=mask
    )


def get_predictions(xx, loader, model, fold_id, subset):
    torch.manual_seed(seed=0)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
    if subset == "test":
        predictions = torch.sigmoid(out[batch.test_mask]).detach().numpy()
        pred = pd.DataFrame({
            'Gene': xx.index[batch.n_id[batch.test_mask].detach().numpy()],
            'Prob': predictions[:, 1],
            'Target': batch.y[batch.test_mask].detach().numpy(),
            'Fold_ID': np.repeat(fold_id, len(predictions[:, 1]))
        })
    elif subset == "train":
        predictions = torch.sigmoid(out[batch.train_mask]).detach().numpy()
        pred = pd.DataFrame({
            'Gene': xx.index[batch.n_id[batch.train_mask].detach().numpy()],
            'Prob': predictions[:, 1],
            'Target': batch.y[batch.train_mask].detach().numpy(),
            'Fold_ID': np.repeat(fold_id, len(predictions[:, 1]))
        })
    else:
        predictions = torch.sigmoid(out[batch.val_mask]).detach().numpy()
        pred = pd.DataFrame({
            'Gene': xx.index[batch.n_id[batch.val_mask].detach().numpy()],
            'Prob': predictions[:, 1],
            'Target': batch.y[batch.val_mask].detach().numpy(),
            'Fold_ID': np.repeat(fold_id, len(predictions[:, 1]))
        })
    return pred


def calc_perm_edge_imp(train_idx=None, test_idx=None, fold_id=None, xx=None, yy=None,
                       n_features=None, n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None,
                       num_layers_post=None, dropout=None, nodes_train=None, nodes_test=None, n_b=None, Network=None,
                       rate=None, eps=None, train_eps=None, edge_list_all=None):
    # Create model instance
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate, eps, train_eps)

    # Load pre-trained model
    filename = f"{MODEL_PATH}Best_GINBiDir_train_foldID_{fold_id}{STUDY_NAME}.pth"
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Create current CV fold train and test datasets
    mask_train = [i in nodes_train for i in xx.index]
    X_folds = xx.loc[mask_train]
    y_folds = yy[mask_train]
    X_train, y_train = X_folds.iloc[train_idx], y_folds[train_idx]
    X_test = X_folds.iloc[test_idx]

    # Perform undersampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    X_res.index = X_train.index[rus.sample_indices_]

    # Iterate through each gene of interest
    for gene_counter, curr_gene in enumerate(genes_oi):

        # Enable to start from a desired gene index to help run multiple versions in parallel
        if gene_counter < gene_start_ind:
            print(f"\t\tSkipping gene_counter={gene_counter}: {curr_gene} as not desired start ({gene_counter})...")
            continue

        # To save time and memory, skip if file already exists
        curr_file = f"{MODEL_PATH}/interpretability/permEdgeImp/GINBiDir_KNN{KNN}_{curr_gene}_foldID{fold_id}_permEdgeScores_{seed_to_start_from}-{end - 1}.csv"
        if os.path.exists(curr_file):
            print(f"\t\tskipping {curr_file} as file already exists")
            continue

        # Obtain prediction for current gene and current fold model - if not present in current fold, skip
        curr_pred_orig = all_preds.loc[(all_preds["Gene"] == curr_gene) & (all_preds["Fold_ID"] == fold_id), "Prob"]
        if len(curr_pred_orig) == 0:
            continue
        else:
            curr_pred_orig = curr_pred_orig.iloc[0]

        # Extract pathways pertaining to the current gene
        rel_paths = gene_set.loc[gene_set["gene_expl"] == curr_gene]
        unique_paths = rel_paths["PathwayID"].unique().tolist()
        perm_imps = pd.DataFrame()
        for path_counter, curr_path in enumerate(unique_paths):
            print(f"\tGene={curr_gene}, fold={fold_id}: about to permute pathway {path_counter}/{len(unique_paths)}={curr_path}...")

            # Subset edge list to only the edges that must be permuted for the current pathway (only genes in curr_path)
            rel_path_genes = rel_paths.loc[rel_paths["PathwayID"] == curr_path, "gene"].to_list()
            rel_edge_list = edge_list_all.loc[edge_list_all["source_name"].isin(rel_path_genes) &
                                              edge_list_all["target_name"].isin(rel_path_genes)]

            # Create the edge_index tensor (2D tensor) number of edge_index cols equals length of edge_weight,
            # which equals to number of non-0 entries in the original weight matrix (Wl)
            # Use this format as the indices in edge_index must match with the indices in X_all_scaled
            node_idx_source = [list(xx.index).index(i) for i in list(rel_edge_list['source_name'])]
            node_idx_target = [list(xx.index).index(i) for i in list(rel_edge_list['target_name'])]
            rel_edge_index = torch.tensor([node_idx_source,  # source nodes
                                           node_idx_target,  # target nodes
                                          ], dtype=torch.long)

            # Convert the subsetted network into required format for xswap: list[Tuple[int, int]]
            true_sub_net_df = pd.DataFrame(rel_edge_index.detach().numpy()).transpose()
            true_sub_net = [tuple(row) for row in true_sub_net_df.to_records(index=False)]

            # Some pathways have no edges spanning the member genes and so they are skipped
            if len(true_sub_net) == 0:
                print(f"\tskipping pathway: {path_counter}/{len(pathways)}={curr_path} as no edges are present....")
                curr_imp = {f"gene_expl": curr_gene,
                            f"CV_fold": fold_id,
                            f"perm_type": "edge_only",
                            f"pathway_id": curr_path,
                            f"pathway_name": rel_paths.loc[rel_paths["PathwayID"] == curr_path, "PathwayName"].unique()[0],
                            f"permutation": pd.NA,
                            f"orig_pred": curr_pred_orig,
                            f"perm_pred": pd.NA}
                perm_imps = pd.concat([perm_imps, pd.DataFrame(curr_imp, index=[0])], axis=0, ignore_index=True)
                continue

            # Delete the edges that will be permuted from the list of edges (yielding so-called static edges):
            # Step 1: detach the full network and store as a dataframe
            full_network = pd.DataFrame(Network.detach().numpy()).transpose()

            # Step 2: merge the full_network with the relevant sub-network
            merged_df = pd.merge(full_network, true_sub_net_df, how='left', indicator=True)

            # Step 3: remove rows that are present only in the left df (full_network) and reset index
            static_edges = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
            static_edges.reset_index(drop=True, inplace=True)
            del merged_df, full_network, true_sub_net_df, node_idx_source, node_idx_target, rel_path_genes, rel_edge_list, rel_edge_index

            # Iterate through seeds and perform permutations
            for curr_seed in range(seed_to_start_from, end):
                # Perform degree-preserving edge permutation - edge_list must be list[Tuple[int, int]]
                perm_edges, perm_stats = xswap.permute_edge_list(edge_list=true_sub_net, seed=curr_seed,
                                                                 allow_self_loops=False, allow_antiparallel=False,
                                                                 multiplier=100) 

                # Combine permuted edges with the non-permuted edges (latter=edges from pathways not being tested)
                perm_edges = pd.DataFrame(perm_edges)
                perm_edges = pd.concat([static_edges, perm_edges], axis=0)

                # Convert perm_edges to a tensor
                perm_edges = torch.tensor(perm_edges.to_numpy(), dtype=torch.long).t()

                # Make permuted edges bi-directional
                perm_edges = torch.cat([perm_edges, perm_edges.flip([0])], dim=1)  # Bidirectional

                # Create data object
                data = Data(
                    x=torch.tensor(xx.values, dtype=torch.float),
                    edge_index=perm_edges,
                    y=torch.tensor(yy, dtype=torch.float),
                    train_mask=torch.tensor([i in X_res.index for i in xx.index], dtype=torch.bool),
                    test_mask=torch.tensor([i in X_test.index for i in xx.index], dtype=torch.bool),
                    val_mask=torch.tensor([i in nodes_test for i in xx.index], dtype=torch.bool)
                )

                # Get predictions from permuted edges
                torch.manual_seed(seed=0)
                nb_loader_test = create_neighbor_loader(data, num_layers, data.test_mask, xx, n_b, X_test.index)
                preds_test = get_predictions(xx, nb_loader_test, model, fold_id, "test")

                torch.manual_seed(seed=0)
                nb_loader_hout = create_neighbor_loader(data, num_layers, data.val_mask, xx, n_b, nodes_test)
                preds_hout = get_predictions(xx, nb_loader_hout, model, fold_id, "held-out")

                preds = pd.concat([preds_test, preds_hout], axis=0, ignore_index=True)

                # Compute prediction on permuted data and concatenate
                curr_imp = {f"gene_expl": curr_gene,
                            f"CV_fold": fold_id,
                            f"perm_type": "edge_only",
                            f"pathway_id": curr_path,
                            f"pathway_name": rel_paths.loc[rel_paths["PathwayID"] == curr_path, "PathwayName"].unique()[0],
                            f"permutation": curr_seed,
                            f"orig_pred": curr_pred_orig,
                            f"perm_pred": preds.loc[preds["Gene"] == curr_gene, "Prob"].iloc[0]}

                perm_imps = pd.concat([perm_imps, pd.DataFrame(curr_imp, index=[0])], axis=0, ignore_index=True)
                print(f"\t\tfold_id={fold_id}/99: completed permutation={curr_seed}")

        # Save permutation edge importance to file
        perm_imps.to_csv(curr_file, index=False)
    return 'Done'


############################################################################################################
# Helper functions
############################################################################################################
def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from the given path."""  
    df = pd.read_csv(file_path)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


def get_data(file_name):
    return pd.read_csv(f'{file_name}')


def main():
    ############################################################################################################
    # Create matrix that was used in the analysis
    ############################################################################################################
    # load them
    X_all_scaled = pd.read_csv(f'{DATABASE_PATH}X_all_scaledAllSep_from_v3_LoadFoldIDs.csv', index_col=[0])
    y_all = get_data(f'{DATABASE_PATH}y_all_scaledAllSep_from_v3_LoadFoldIDs.csv')  # Same order as X_all_scaled
    y_all = y_all["targ_stat"].to_numpy()

    # Read in train and held-out sets for node indices
    X, Xht = map(load_data, [f'{DATABASE_PATH}{DATA_FILE_TRAIN}', f'{DATABASE_PATH}{DATA_FILE_TEST}'])
    Nodes_train = X.index.values
    Nodes_ht = Xht.index.values

    #######################################################################################################
    # Load the network (mKNN+mst or scinet)
    #######################################################################################################
    # Read in edge list for desired hetionet graph 
    with open(f"{DATABASE_PATH}/hetionet_graph/GiG_HGNC.pkl", "rb") as f:
        edge_list = pickle.load(f)
    edge_list.drop(columns=["Unnamed: 0"], inplace=True)

    # Subset list of train and held-out test nodes to common nodes
    Nodes_train = np.array(list(set(X_all_scaled.index) & set(Nodes_train)))
    Nodes_ht = np.array(list(set(X_all_scaled.index) & set(Nodes_ht)))

    # Subset edge list for graph to common nodes
    edge_list_all = edge_list.loc[np.isin(edge_list['source_name'], X_all_scaled.index), :]
    edge_list_all = edge_list_all.loc[np.isin(edge_list_all['target_name'], X_all_scaled.index), :]

    # Create the edge_index tensor (2D tensor) number of edge_index cols equals length of edge_weight,
    node_idx_source = [list(X_all_scaled.index).index(i) for i in list(edge_list_all['source_name'])]
    node_idx_target = [list(X_all_scaled.index).index(i) for i in list(edge_list_all['target_name'])]

    # Consider: Perform below in the calc importances function
    edge_index = torch.tensor([node_idx_source,  # source nodes
                               node_idx_target,  # target nodes
                               ], dtype=torch.long)

    ############################################################################################################
    # Define the cross-validation scheme
    ############################################################################################################
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    ############################################################################################################
    # Load study
    ############################################################################################################
    study = optuna.load_study(study_name=STUDY_NAME, storage=TRAINING_DATABASE_FILE)
    print(study.best_params)

    #############################################################################################################
    # Select the best parameters
    #############################################################################################################
    lr = study.best_params['lr']
    n_hidden = study.best_params['n_hidden']
    dropout = study.best_params['dropout']
    weight_decay = study.best_params['weight_decay']
    num_layers = study.best_params['num_layers']
    eps = study.best_params['eps']
    train_eps = study.best_params['train_eps']
    num_layers_pre = study.best_params['num_layers_pre']
    num_layers_post = study.best_params['num_layers_post']
    batch_size = study.best_params['batch_size']
    num_epochs = EPOCHS
    rate = study.best_params['rate']
    n_b = KNN
    n_features = len(X_all_scaled.columns)
    n_classes = len(np.unique(y_all))

    #############################################################################################################
    # Calculate importances for each fold model
    #############################################################################################################
    X_folds = X_all_scaled.loc[[True if i in Nodes_train else False for i in X_all_scaled.index], :]
    y_folds = y_all[[True if i in Nodes_train else False for i in X_all_scaled.index]]

    # Load fold IDs
    splits_df = pd.read_csv(f'{MODEL_PATH}/predictions/Predictions_Best_GINBiDir_WithCVModels_{STUDY_NAME}.csv')

    # Unique fold ids
    fold_ids = splits_df['Fold_ID'].unique()
    args = []
    for fold_id in fold_ids:
        # if starting from a specific fold, and fold_id < desired fold, skip until you reach the desired fold
        if (fold_to_start_from != 0) and (fold_id < fold_to_start_from):
            continue

        # Get rows for each fold
        fold_data = splits_df[splits_df['Fold_ID'] == fold_id]

        # Determine train and test genes
        train_genes = fold_data[fold_data['Set'] == 'Train (fold)']['Gene']
        test_genes = fold_data[fold_data['Set'] == 'Test (fold)']['Gene']

        # Determine the indices in the dataframe X_all_scaled
        train_idx = X_folds.index.get_indexer(train_genes)
        test_idx = X_folds.index.get_indexer(test_genes)

        # Appending dictionary to args
        args.append(dict(train_idx=train_idx, test_idx=test_idx, fold_id=fold_id, xx=X_all_scaled, yy=y_all,
                         n_features=n_features, n_classes=n_classes, n_hidden=n_hidden, num_layers=num_layers,
                         num_layers_pre=num_layers_pre, num_layers_post=num_layers_post, dropout=dropout,
                         nodes_train=Nodes_train, nodes_test=Nodes_ht, n_b=n_b,
                         Network=edge_index, rate=rate, eps=eps, train_eps=train_eps, edge_list_all=edge_list_all))

    results = Parallel(n_jobs=-1)(delayed(calc_perm_edge_imp)(**arg) for arg in args)


if __name__ == "__main__":
    main()
    print("done script")
