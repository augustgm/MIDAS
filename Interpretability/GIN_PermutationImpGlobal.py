#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#######################################################################################
# Perform permutation feature importance on training set on optimised, best GIN model #
#######################################################################################
# Standard Libraries
import pickle
import os
import sys

# Third-party Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import optuna

# Imblearn Libraries
from imblearn.under_sampling import RandomUnderSampler

# Scikit-learn Libraries
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

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
KNN = 50 
fold_to_start_from = "None"

if fold_to_start_from == "None":
    fold_to_start_from = 0
else:
    fold_to_start_from = int(fold_to_start_from)


# Define paths
DATABASE_PATH = "MIDAS_analysis/MIDAS_database/"
OPTUNA_DATABASE_PATH = f"optuna_db/GINBiDir/"
MODEL_PATH = f"MIDAS_analysis/GraphModels/models/GINBiDir/"

# Node classification data - used only to get IDs for train and held-out sets
DATA_FILE_TRAIN = 'train_imputed_exclMissCrispr_crispr_gsnp_hlapep_autoimRheum_scElnetBulkXgbShap.csv'
DATA_FILE_TEST = 'test_imputed_exclMissCrispr_crispr_gsnp_hlapep_autoimRheum_scElnetBulkXgbShap.csv'  

# Additional target information
TARGET_FILE = 'CRI2017-2019_IOtargets_exclAgbased_CellTherapy_OncolyticVirus.csv'

# Set global seed for all of pytorch and pytorch geometric
torch.manual_seed(seed=0)

# Number of epochs
EPOCHS = int(os.getenv("EPOCHS", 100))

# Study name
STUDY_NAME = f'GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_GiG_KNN{KNN}'

# Optuna file name
TRAINING_DATABASE_FILE = f'sqlite:///{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db' 

############################################################################################################
# GIN class model
############################################################################################################
class GNN(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate, eps,
                 train_eps):

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
    def forward(self, x, edge_index):  # GINConv does not accept weights

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
# Function that performs the predictions for one fold
############################################################################################################
def create_neighbor_loader(data, num_layers, mask, xx, n_b, nodes_list):
    return NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,
        batch_size=sum([True if i in nodes_list else False for i in xx.index]),
        input_nodes=mask
    )


def get_train_outputs_and_preds(xx, loader, model, fold_id):
    torch.manual_seed(seed=0)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
    predictions = torch.sigmoid(out[batch.train_mask]).detach().numpy()
    pred = pd.DataFrame({
        'Gene': xx.index[batch.n_id[batch.train_mask].detach().numpy()],
        'Prob': predictions[:, 1],
        'Target': batch.y[batch.train_mask].detach().numpy(),
        'Fold_ID': np.repeat(fold_id, len(predictions[:, 1]))
    })
    return pred, out[batch.train_mask]


def calc_perm_importance(train_idx=None, test_idx=None, fold_id=None, xx=None, yy=None,
                         n_features=None, n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None,
                         num_layers_post=None, dropout=None, nodes_train=None, nodes_test=None, n_b=None, Network=None,
                         rate=None, eps=None, train_eps=None):

    # Create model instance
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate, eps, train_eps)

    # Create current CV fold train and test datasets
    mask_train = [i in nodes_train for i in xx.index]
    X_folds = xx.loc[mask_train]
    y_folds = yy[mask_train]
    X_train, y_train = X_folds.iloc[train_idx], y_folds[train_idx]
    X_test = X_folds.iloc[test_idx]

    # Permute the CV train set
    perm_imps = pd.DataFrame()
    for feat_ind, curr_feat in enumerate(xx.columns):
        print(f"feat_ind={feat_ind}: fold_id={fold_id}/99")

        for curr_seed in range(500):
            xx_perm = xx.copy(deep=True)
            xx_perm[curr_feat] = shuffle(xx[curr_feat].values, random_state=curr_seed)

            # Perform undersampling
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)
            X_res.index = X_train.index[rus.sample_indices_]

            # Create Data object
            data = Data(
                x=torch.tensor(xx_perm.values, dtype=torch.float),
                edge_index=Network,
                y=torch.tensor(yy, dtype=torch.float),
                train_mask=torch.tensor([i in X_res.index for i in xx_perm.index], dtype=torch.bool),
                test_mask=torch.tensor([i in X_test.index for i in xx_perm.index], dtype=torch.bool),
                val_mask=torch.tensor([i in nodes_test for i in xx_perm.index], dtype=torch.bool)
            )

            # Load pre-trained model
            filename = f"{MODEL_PATH}Best_GINBiDir_train_foldID_{fold_id}{STUDY_NAME}.pth"
            model.load_state_dict(torch.load(filename))

            # Get predictions from permuted data
            model.eval()
            torch.manual_seed(seed=0)
            nb_loader_train = create_neighbor_loader(data, num_layers, data.train_mask, xx_perm, n_b, X_res.index)
            preds, raw_outs = get_train_outputs_and_preds(xx_perm, nb_loader_train, model, fold_id)

            # Compute performance on permuted data and concatenate
            loss_func = nn.BCEWithLogitsLoss()
            curr_imp = {f"CV_fold": fold_id,
                        f"feature": curr_feat,
                        f"permutation": curr_seed,
                        f"bce_loss": loss_func(raw_outs[:, 1], data.y[data.train_mask]).detach().numpy(),
                        f"perm_roc_auc": roc_auc_score(y_true=data.y[data.train_mask].detach().numpy(),
                                                       y_score=preds["Prob"])}
            perm_imps = pd.concat([perm_imps, pd.DataFrame(curr_imp, index=[0])], axis=0, ignore_index=True)
            print(f"\tfeat_ind={feat_ind}: fold_id={fold_id}/99: completed permutation={curr_seed}")

    # Save results of permutation
    perm_imps.to_csv(f'{MODEL_PATH}/interpretability/permImp/GINBiDir_KNN{KNN}_foldID{fold_id}_CVTrainSet_permScores.csv',
                     index=False)
    return 'Done'


def get_data(file_name):
    return pd.read_csv(f'{file_name}')


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from the given path.""" 
    df = pd.read_csv(file_path) 
    df.set_index(df.iloc[:, 0], inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


def main():
    try:
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
        print(f"Length of Nodes_train after subsetting to nodes in X_all_scaled: {len(Nodes_train)}")

        # Subset edge list for graph to common nodes
        edge_list_all = edge_list.loc[np.isin(edge_list['source_name'], X_all_scaled.index), :]
        edge_list_all = edge_list_all.loc[np.isin(edge_list_all['target_name'], X_all_scaled.index), :]

        # Create the edge_index tensor (2D tensor) number of edge_index cols equals length of edge_weight,
        node_idx_source = [list(X_all_scaled.index).index(i) for i in list(edge_list_all['source_name'])]
        node_idx_target = [list(X_all_scaled.index).index(i) for i in list(edge_list_all['target_name'])]
        edge_index = torch.tensor([node_idx_source,  # source nodes
                                   node_idx_target,  # target nodes
                                   ], dtype=torch.long)

        edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)  # Bidirectional

        # Create a 1D torch.tensor (edge_weights) for the edge weights of 1s
        edge_weights = torch.tensor([1 for i in range(edge_index.shape[1])], dtype=torch.float)

        del edge_list, node_idx_source, node_idx_target

        ############################################################################################################
        # Define the cross-validation scheme
        ############################################################################################################
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

        ############################################################################################################
        # Load optimization
        ############################################################################################################
        study = optuna.load_study(study_name=STUDY_NAME, storage=TRAINING_DATABASE_FILE)

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
        # Predictions with fold models
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
                             Network=edge_index, rate=rate, eps=eps, train_eps=train_eps))

        Parallel(n_jobs=-1)(delayed(calc_perm_importance)(**arg) for arg in args)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
    print("done script")
