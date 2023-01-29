import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold

def evaluate_model(num_leaves, min_child_weight, feature_fraction, subsample, drop_rate, max_depth):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.07,
        "verbosity": -1,
        "num_leaves": int(num_leaves),
        "min_child_weight": min_child_weight,
        "feature_fraction": feature_fraction,
        "subsample": subsample,
        'drop_rate': drop_rate,
        'max_depth': int(max_depth)
    }
    num_boost_round = 10000

    # define the number of folds for cross-validation
    n_folds = 5

    # create a stratified k-fold iterator
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

    # initialize a list to store the evaluation metric for each fold
    scores = []

    X = pd.read_csv("X_Train.csv")
    y_train = pd.read_csv("y_train.csv")

    # iterate over the folds
    for id_train, id_val in skf.split(X, y_train):
        # get the training and validation data for this fold
        X_train_fold = X.iloc[id_train]
        y_train_fold = y_train[id_train]
        X_val_fold = X.iloc[id_val]
        y_val_fold = y_train[id_val]

        xgb_train = xgb.Dataset(X_train_fold, y_train_fold)
        xgb_val = xgb.Dataset(X_val_fold, y_val_fold)

        # train the model with the specified parameters on the training data
        model = xgb.train(params, xgb_train, num_boost_round, valid_sets=xgb_val, verbose_eval=100, early_stopping_rounds=100)
        scores.append(model.best_score['valid_0']['gini'])

    # return the mean evaluation metric across all folds
    return np.mean(scores)

# define the hyperparameters to be optimised
hyperparameters = {
    "num_leaves": (4, 50),
    "min_child_weight": (0.001, 150),
    "feature_fraction": (0.1, 0.9),
    "subsample": (0.1, 1),
    'drop_rate': (0.1, 0.8),
    'max_depth': (3, 20)
}
#UNCOMMENT TO START BAYESIAN OPTIMISATION ~10 MINS

# perform Bayesian optimisation to find the optimal hyperparameters
optimizer = BayesianOptimization(evaluate_model, hyperparameters)
optimizer.maximize(n_iter=10)

# display the optimal values of the hyperparameters
print("Optimal hyperparameters:")
print(optimizer.max)

