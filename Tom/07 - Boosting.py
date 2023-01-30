import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

params = {'colsample_bytree': 0.1, 'max_depth': 11.634745946255054, 'max_leaves': 36.048274745276096, 'min_child_weight': 114.17706174190377, 'subsample': 0.1}

X_train = pd.read_csv("X_Train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Create the XGBoost model
xgb_model = xgb.XGBRegressor()

# Create the KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the evaluation metrics for each fold
scores = []

mae_scores = []
baseline_scores = []
# Iterate over the folds
for train_index, val_index in kf.split(X_train):
    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the model on the training data
    xgb_model.fit(X_train_fold, y_train_fold)

    # Make predictions on the validation data
    y_pred_fold = xgb_model.predict(X_val_fold)
    # Compute the MAE score
    baseline = mean_absolute_error(y_val_fold, np.array([np.mean(y_train['IMDbRating'])] * len(y_pred_fold)))
    mae = mean_absolute_error(y_val_fold, y_pred_fold)
    print(mae)
    print(baseline)
    mae_scores.append(mae)
    baseline_scores.append(baseline)

# Compute the average MAE score
average_mae = sum(mae_scores) / len(mae_scores)
print("Average MAE:", average_mae)
average_baseline = sum(baseline_scores) / len(baseline_scores)
print("Average Baseline:", average_baseline)

y_pred = xgb_model.predict(X_test)
print(y_pred)
print(y_test)
results = pd.DataFrame({'Actual': y_test['IMDbRating'], 'Prediction': y_pred})
results.to_csv('results.csv', index=False)