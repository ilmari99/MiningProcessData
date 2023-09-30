""" 
Train a model, that predicts the silica concentrate from averaged hourly data.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from check_data import  compute_outlier_mask
from load_data import remove_sus_period

SHUFFLE_TRAIN = True
APPLY_PCA = False
NHOURS = 1

df = pd.read_csv(f"miningdata/{NHOURS}hourly_heuristic_train.csv", parse_dates=['date'])
df_validation = pd.read_csv(f"miningdata/{NHOURS}hourly_heuristic_validation.csv", parse_dates=['date'])
#df = remove_sus_period(df)
df_test = pd.read_csv(f"miningdata/{NHOURS}hourly_heuristic_test.csv", parse_dates=['date'])

df = pd.concat([df, df_validation])
# Only take every Nth row
#df = df.iloc[::NHOURS, :]

#df = pd.read_csv("miningdata/data_flattened_hourly.csv", parse_dates=['date'])
#df = pd.read_csv("miningdata/data_combined_hourly_mean_max_min_std.csv", parse_dates=['date'])
exclude_cols = ['% Iron Feed', '% Silica Feed', '% Iron Concentrate', '% Silica Concentrate']

df = df.drop(columns=["date"])
df_test = df_test.drop(columns=["date"])

silica_conc_idx = df.columns.get_loc(f"% Silica Concentrate{'_' + str(NHOURS-1) if NHOURS > 1 else ''}")

outlier_mask = compute_outlier_mask(df, outlier_z_thresh=4)
print(f"Number of outliers:\n{outlier_mask.sum()}")
#df = df[~outlier_mask.any(axis=1)]

# Split the data into train and test sets
X_train = df
X_test = df_test

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=df.columns)
X_test = pd.DataFrame(X_test, columns=df.columns)

# Y is the next hour silica concentrate.
# We need to shift the values by one hour, since we want to predict the Next hour silica concentrate.
y_train = X_train.iloc[:, silica_conc_idx].shift(-1)
y_test = X_test.iloc[:, silica_conc_idx].shift(-1)

# Drop the last NHOURS rows, since they have no y value
X_train = X_train.iloc[:-1, :]
X_test = X_test.iloc[:-1, :]
y_train = y_train.iloc[:-1]
y_test = y_test.iloc[:-1]

if SHUFFLE_TRAIN:
    # Shuffle but keep the same indices for X and y
    idxs = np.arange(len(X_train))
    np.random.shuffle(idxs)
    X_train = X_train.iloc[idxs, :]
    y_train = y_train.iloc[idxs]

if APPLY_PCA:
    pca = PCA(n_components=0.8)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"Number of components: {pca.n_components_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    

random_forest_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5],
    'min_samples_split': [2, 4, 5],
    'min_samples_leaf': [1, 4, 5],
    'criterion': ["squared_error"]
}
svr_param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}


# Find bet model with GridSearchCV
model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="squared_error")
model.fit(X_train, y_train)
#param_grid = random_forest_param_grid

#grid_search = RandomizedSearchCV(model, param_grid, n_iter=1, cv=5, verbose=2, n_jobs=6,refit=True, scoring = 'neg_mean_squared_error')
#grid_search.fit(X_train, y_train)
#print(grid_search.best_params_)
#model = grid_search.best_estimator_

# Predict the test set
y_pred = model.predict(X_test)
#y_pred = y_pred[:-1]

# Get scaler coefficients for y
y_mean_scale = scaler.mean_[silica_conc_idx]
y_std_scale = scaler.scale_[silica_conc_idx]
# Unscale y_pred
y_pred = y_pred*y_std_scale + y_mean_scale

y_test = df_test.iloc[:, silica_conc_idx].shift(-1)[:-1]

# Calculate average absolute change in silica concentrate
silica_abs_change = np.abs(np.diff(y_test))
mean_abs_change = np.mean(silica_abs_change)
print(f"Mean absolute change in silica concentrate in {NHOURS}: {mean_abs_change:.2f}")

if NHOURS > 1:
    # Calculate the mean change in just 1 hour
    silica_concentrate_index_0 = df.columns.get_loc(f"% Silica Concentrate_0")
    silica_concentrate_index_1 = df.columns.get_loc(f"% Silica Concentrate_1")
    # Difference in each row between the silica concentrate at hour 0 and 1
    silica_concentrate_diff = df.iloc[:, silica_concentrate_index_1] - df.iloc[:, silica_concentrate_index_0]
    # Mean change in silica concentrate in 1 hour
    mean_change = np.mean(np.abs(silica_concentrate_diff))
    print(f"Mean change in silica concentrate in 1 hour: {mean_change:.2f}")



# Compute the mse
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute error: {mae:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2:.2f}")



