import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

def evaluate_model(max_leaves, min_child_weight, colsample_bytree, subsample, max_depth):
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "learning_rate": 0.07,
        "verbosity": 1,
        "max_leaves": int(max_leaves),
        "min_child_weight": min_child_weight,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        'max_depth': int(max_depth)
    }
    num_boost_round = 10000

    # define the number of folds for cross-validation
    n_folds = 5

    # create a stratified k-fold iterator
    kf =KFold(n_splits=n_folds, shuffle=True, random_state=1)

    # initialize a list to store the evaluation metric for each fold
    scores = []

    X = pd.read_csv("X_Train.csv")
    y_train = pd.read_csv("y_train.csv")

    # iterate over the folds
    for id_train, id_val in kf.split(X, y_train):
        # get the training and validation data for this fold
        X_train_fold = X.iloc[id_train]
        y_train_fold = y_train.iloc[id_train]
        X_val_fold = X.iloc[id_val]
        y_val_fold = y_train.iloc[id_val]

        # train the model with the specified parameters on the training data
        dtrain = xgb.DMatrix(X_train_fold, y_train_fold)
        dval = xgb.DMatrix(X_val_fold, y_val_fold)
        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round, evals=evals, early_stopping_rounds=100, verbose_eval=100)
        #print(model.best_score)
        scores.append(model.best_score)

    # return the mean evaluation metric across all folds
    return np.mean(scores)

# define the hyperparameters to be optimised
hyperparameters = {
    "max_leaves": (4, 50),
    "min_child_weight": (0.001, 150),
    "colsample_bytree": (0.1, 0.9),
    "subsample": (0.1, 1),
    'max_depth': (3, 20)
}
#UNCOMMENT TO START BAYESIAN OPTIMISATION ~10 MINS

# perform Bayesian optimisation to find the optimal hyperparameters
optimizer = BayesianOptimization(evaluate_model, hyperparameters)
optimizer.maximize(n_iter=10)

# display the optimal values of the hyperparameters
print("Optimal hyperparameters:")
print(optimizer.max)

