#!/usr/bin/env python3
# -*- coding: utf-8 -*-
############################################
# Functions to create, train, optimise and run GINBiDir
#############################################
# Standard Libraries
import pickle
import logging
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
location = "myriad"
KNN = 50 
hetio_graph = "GiG" 
num_trials = 100

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
STUDY_NAME = f'GINBiDir_Optuna_inductive_SCINETtopologyScores_OptLayers_{hetio_graph}_KNN{KNN}'

# Optuna file name
TRAINING_DATABASE_FILE = f'sqlite:///{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db'  # SQLite database - changed path


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
# Objective function for a single set of hyperparameters in Optuna
############################################################################################################
def base_objective_single_arg(train_idx=None, test_idx=None, xx=None, yy=None, n_features=None,
                              n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None, num_layers_post=None,
                              dropout=None, lr=None, weight_decay=None,
                              nodes_train=None, batch_size=None, num_epochs=None, n_b=None,
                              Network=None, edge_weights=None, rate=None, eps=None,
                              train_eps=False):  # I kept weights here but they are not used
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate, eps,
                train_eps)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split the data into train and test sets
    X_folds = xx.loc[[True if i in nodes_train else False for i in xx.index], :]
    y_folds = yy[[True if i in nodes_train else False for i in xx.index]]
    X_train, y_train = X_folds.iloc[train_idx, :], y_folds[train_idx]
    X_test, y_test = X_folds.iloc[test_idx, :], y_folds[test_idx]

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    X_res.index = X_train.index[rus.sample_indices_]

    # Create torch geometric data object
    Xgraph = torch.tensor(xx.values, dtype=torch.float)

    # Labels for a binary classification task
    ygraph = torch.tensor(yy, dtype=torch.float)
    data = Data(x=Xgraph, edge_index=Network, edge_attr=edge_weights, y=ygraph,
                train_mask=torch.tensor([True if i in X_res.index else False for i in xx.index], dtype=torch.bool),
                test_mask=torch.tensor([True if i in X_test.index else False for i in xx.index], dtype=torch.bool))
    torch.manual_seed(seed=0)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,  
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )
    criterion = nn.BCEWithLogitsLoss() 
    model.train()

    for epoch in range(num_epochs + 1):
        total_loss = 0

        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            # out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask, 1], batch.y[batch.train_mask])
            total_loss += loss
            loss.backward()
            optimizer.step()

    # Make predictions on test fold
    model.eval()

    # Create a NeighborLoader for the test nodes
    torch.manual_seed(seed=0)
    test_loader = NeighborLoader(data, num_neighbors=[n_b] * num_layers, 
                                 batch_size=sum([True if i in X_test.index else False for i in xx.index]),
                                 input_nodes=data.test_mask)

    # Extract the single batch from the loader
    test_batch = next(iter(test_loader))

    # Evaluate the model
    with torch.no_grad():
        out = model(test_batch.x, test_batch.edge_index)
    y_pred = torch.sigmoid(out[test_batch.test_mask]).detach().numpy()[:, 1] 

    # Evaluate the predictions
    score = roc_auc_score(test_batch.y[test_batch.test_mask].detach().numpy(), y_pred)
    # Report the score to Optuna
    return score


############################################################################################################
# Optimization function
############################################################################################################
def run_optimization(n_trials, X=None, y=None, cv=None, nodes_train=None, nodes_test=None, Network=None,
                     edge_weights=None, num_epochs=None): 
    """
    Wrapper function for the objective function.
    """
    # Prepare arguments for parallel execution
    X_folds = X.loc[[True if i in nodes_train else False for i in X.index], :]
    y_folds = y[[True if i in nodes_train else False for i in X.index]]
    cv_splits = list(cv.split(X_folds, y_folds))
    args = [dict(train_idx=train_idx, test_idx=test_idx, xx=X, yy=y)
            for fold_id, (train_idx, test_idx) in enumerate(cv_splits)]

    # Run the optimization in parallel

    def objective(trial):
        """
                Objective function for Optuna optimization.
        """
        lr = trial.suggest_categorical("lr", [0.01, 0.02, 0.03])
        n_hidden = trial.suggest_categorical("n_hidden", [16, 32, 64, 128, 256, 512, 1024])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])

        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])

        eps = trial.suggest_categorical("eps", [0.001, 0.01, 0.15])
        train_eps = trial.suggest_categorical("train_eps", [False, True])

        num_layers_pre = trial.suggest_categorical('num_layers_pre', [1, 2, 3])
        num_layers_post = trial.suggest_categorical('num_layers_post', [1, 2, 3])
        rate = trial.suggest_categorical("rate", [0.1, 0.2, 0.3, 0.4, 0.5])
        n_b = KNN  # necessary for the NeighborLoader
        n_features = len(X.columns)
        n_classes = len(np.unique(y))

        # Use Joblib's Parallel to parallelize the execution
        scores = Parallel(n_jobs=-1)(
            delayed(base_objective_single_arg)(**arg, n_features=n_features, n_classes=n_classes, n_hidden=n_hidden,
                                               num_layers=num_layers, num_layers_pre=num_layers_pre,
                                               num_layers_post=num_layers_post, dropout=dropout, lr=lr,
                                               weight_decay=weight_decay,
                                               nodes_train=nodes_train, batch_size=batch_size,
                                               num_epochs=num_epochs, n_b=n_b, Network=Network,
                                               edge_weights=edge_weights, rate=rate, eps=eps, train_eps=train_eps) for
            arg in args)

        # Calculate the mean score
        mean_score = np.mean(scores)
        return mean_score

    if os.path.exists(f'{OPTUNA_DATABASE_PATH}{STUDY_NAME}.db'):
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.load_study(study_name=STUDY_NAME, storage=TRAINING_DATABASE_FILE, sampler=sampler)
    else:
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(study_name=STUDY_NAME, storage=TRAINING_DATABASE_FILE, direction='maximize',
                                    load_if_exists=False, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


############################################################################################################
# Function that performs the predictions for one fold
############################################################################################################
def make_predictions(train_idx=None, test_idx=None, fold_id=None, xx=None, yy=None,
                     n_features=None, n_classes=None, n_hidden=None, num_layers=None, num_layers_pre=None,
                     num_layers_post=None, dropout=None, lr=None, weight_decay=None, nodes_train=None,
                     nodes_test=None, batch_size=None, num_epochs=None, n_b=None, Network=None, edge_weights=None,
                     rate=None, eps=None, train_eps=False):
    """
    Function that performs the predictions for each fold with the best model devised with the optimum hyperparameters
    found by Optuna. The rest of the parameters are fitted to the training fold data and the predictions are made in the
    validation fold
    """
    model = GNN(n_features, n_classes, n_hidden, num_layers, num_layers_pre, num_layers_post, dropout, rate, eps, train_eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split the data into train and test sets
    X_folds = xx.loc[[True if i in nodes_train else False for i in xx.index], :]
    y_folds = yy[[True if i in nodes_train else False for i in xx.index]]
    X_train, y_train = X_folds.iloc[train_idx, :], y_folds[train_idx]
    X_test, y_test = X_folds.iloc[test_idx, :], y_folds[test_idx]

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    X_res.index = X_train.index[rus.sample_indices_]

    # Create torch geometric data object
    Xgraph = torch.tensor(xx.values, dtype=torch.float)

    # Edge list
    edge_index = Network

    # Labels for a binary classification task
    ygraph = torch.tensor(yy, dtype=torch.float)
    data = Data(x=Xgraph, edge_index=edge_index, edge_attr=edge_weights, y=ygraph,
                train_mask=torch.tensor([True if i in X_res.index else False for i in xx.index], dtype=torch.bool),
                test_mask=torch.tensor([True if i in X_test.index else False for i in xx.index], dtype=torch.bool),
                val_mask=torch.tensor([True if i in nodes_test else False for i in xx.index], dtype=torch.bool))
    torch.manual_seed(seed=0)
    train_loader = NeighborLoader(
        data,
        num_neighbors=[n_b] * num_layers,  # [10, 10],
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )

    criterion = nn.BCEWithLogitsLoss() 
    model.train()
    for epoch in range(num_epochs + 1):
        total_loss = 0
        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask, 1], batch.y[batch.train_mask])
            total_loss += loss
            loss.backward()
            optimizer.step()

    filename = f"{MODEL_PATH}Best_GINBiDir_train_foldID_{fold_id}{STUDY_NAME}.pth"
    torch.save(model.state_dict(), filename)

    model.eval()
    # Create a NeighborLoader for the train nodes
    torch.manual_seed(seed=0)
    train_eval_loader = NeighborLoader(data,
                                       num_neighbors=[n_b] * num_layers,
                                       batch_size=sum([True if i in X_res.index else False for i in xx.index]),
                                       input_nodes=data.train_mask)

    # Extract the single batch from the loader
    train_eval_batch = next(iter(train_eval_loader))

    # Evaluate the model
    with torch.no_grad():
        out = model(train_eval_batch.x, train_eval_batch.edge_index)
    predictions_train = torch.sigmoid(out[train_eval_batch.train_mask]).detach().numpy()
    predictions_train = pd.DataFrame(
        dict(Gene=xx.index[train_eval_batch.n_id[train_eval_batch.train_mask].detach().numpy()],
             Prob=predictions_train[:, 1],
             Target=train_eval_batch.y[train_eval_batch.train_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_train[:, 1])),
             Set=np.repeat('Train (fold)',
                           len(predictions_train[:, 1]))))

    # Create a NeighborLoader for the test nodes
    torch.manual_seed(seed=0)
    test_loader = NeighborLoader(data,
                                 num_neighbors=[n_b] * num_layers, 
                                 batch_size=sum([True if i in X_test.index else False for i in xx.index]),
                                 input_nodes=data.test_mask)

    # Extract the single batch from the loader
    test_batch = next(iter(test_loader))
    with torch.no_grad():
        out = model(test_batch.x, test_batch.edge_index)
    predictions_test = torch.sigmoid(out[test_batch.test_mask]).detach().numpy()
    predictions_test = pd.DataFrame(
        dict(Gene=xx.index[test_batch.n_id[test_batch.test_mask].detach().numpy()], Prob=predictions_test[:, 1],
             Target=test_batch.y[test_batch.test_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_test[:, 1])),
             Set=np.repeat('Test (fold)',
                           len(predictions_test[:, 1])))) 

    # Create a NeighborLoader for the held-out test nodes
    torch.manual_seed(seed=0)
    ht_loader = NeighborLoader(data,
                               num_neighbors=[n_b] * num_layers, 
                               batch_size=sum([True if i in nodes_test else False for i in xx.index]),
                               input_nodes=data.val_mask)

    # Extract the single batch from the loader
    ht_batch = next(iter(ht_loader))
    with torch.no_grad():
        out = model(ht_batch.x, ht_batch.edge_index) 
    predictions_ht = torch.sigmoid(out[ht_batch.val_mask]).detach().numpy() 
    predictions_ht = pd.DataFrame(
        dict(Gene=xx.index[ht_batch.n_id[ht_batch.val_mask].detach().numpy()], Prob=predictions_ht[:, 1],
             Target=ht_batch.y[ht_batch.val_mask].detach().numpy(),
             Fold_ID=np.repeat(fold_id, len(predictions_ht[:, 1])),
             Set=np.repeat('Held out (pred with fold model)',
                           len(predictions_ht[:, 1]))))
    pr = pd.concat([predictions_train, predictions_test, predictions_ht], axis=0)
    return pr


def get_data(file_name):
    return pd.read_csv(f'{file_name}')


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from the given path."""
    df = pd.read_csv(file_path)  # you need to set the index for each data type
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
        # Read in edge list for desired hetionet graph (contains unique edges - each is counted once)
        with open(f"{DATABASE_PATH}/hetionet_graph/{hetio_graph}_HGNC.pkl", "rb") as f:
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
        # Run optimization
        ############################################################################################################
        study = run_optimization(n_trials=num_trials, X=X_all_scaled, y=y_all, cv=cv, nodes_train=Nodes_train,
                                 nodes_test=Nodes_ht, Network=edge_index, edge_weights=edge_weights, num_epochs=EPOCHS)

        with open(f'{OPTUNA_DATABASE_PATH}{STUDY_NAME}.pkl', 'wb') as f:
            pickle.dump(study, f)

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

        args = [dict(train_idx=train_idx, test_idx=test_idx, fold_id=fold_id, xx=X_all_scaled, yy=y_all,
                     n_features=n_features, n_classes=n_classes, n_hidden=n_hidden, num_layers=num_layers,
                     num_layers_pre=num_layers_pre, num_layers_post=num_layers_post, dropout=dropout, lr=lr,
                     weight_decay=weight_decay, nodes_train=Nodes_train,
                     nodes_test=Nodes_ht, batch_size=batch_size, num_epochs=num_epochs, n_b=n_b, Network=edge_index,
                     edge_weights=edge_weights, rate=rate, eps=eps, train_eps=train_eps) for
                fold_id, (train_idx, test_idx) in enumerate(cv.split(X_folds, y_folds))]
        results = Parallel(n_jobs=-1)(delayed(make_predictions)(**arg) for arg in args)
        pred = pd.concat(results, axis=0)

        #############################################################################################################
        # Save predictions
        #############################################################################################################
        filename = f"{MODEL_PATH}/predictions/Predictions_Best_GINBiDir_WithCVModels_{STUDY_NAME}.csv"
        pred.to_csv(filename, index=True)

    except Exception as e:
        print(f"Error in main: {e}")
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
    print("done script")
