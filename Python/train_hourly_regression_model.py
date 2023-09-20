""" 
Train a model, that predicts the silica concentrate from averaged hourly data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from check_data import  compute_outlier_mask

df = pd.read_csv("miningdata/data_flattened_hourly.csv", parse_dates=['date'])
df = df.drop(columns=['date'])
outlier_mask = compute_outlier_mask(df, outlier_z_thresh=3.5)
print(f"Number of outliers:\n{outlier_mask.sum()}")
#df = df[~outlier_mask.any(axis=1)]

silica_conc_idx = df.columns.get_loc('% Silica Concentrate')

# Split the data into train and test sets
X_train, X_test = train_test_split(df, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=df.columns)
X_test = pd.DataFrame(X_test, columns=df.columns)

# Y is the next hour silica concentrate.
# We need to shift the values by one hour, since we want to predict the Next hour silica concentrate.
y_train = X_train.iloc[:, silica_conc_idx].shift(-1)
y_test = X_test.iloc[:, silica_conc_idx].shift(-1)

# Drop the last row, since it has no corresponding y value.
X_train = X_train.iloc[:-1, :]
X_test = X_test.iloc[:-1, :]
y_train = y_train.iloc[:-1]
y_test = y_test.iloc[:-1]


# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Compute the mse
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute error: {mae:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2:.2f}")



