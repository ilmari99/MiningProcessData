""" 
In this file we train a neural network, which is used as a soft sensor to predict silica concentration
"""

import tensorflow as tf
from load_data import load_to_dataframe
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from check_data import check_data_timeseries, OutlierDetector, remove_dates
    

def get_data_hourly(df : pd.DataFrame, hours=1, add_dim=False, remove_outliers=True):
    """
    Process the dataframe, so that X has sequences of 1 hour + 1 and y has the next silica concentration.
    so Y is the silica concentration at the last vector in X, and hence silica concentration is set to -1 for the last vector in X.
    Also '\% iron concentrate' is set to -1 for the last vector in X.
    The measurements are made every 20 seconds, so X has sequences of 180 vectors and y has the next silica concentration.
    """
    # Divide in to sequences of 1 hour + 1 vector
    X = []
    y = []
    steps = 180*hours
    date_to_index = {}
    for i_enum, i in enumerate(range(0, len(df)-steps, steps)):
        date_to_index[df['date'].iloc[i]] = i_enum
        temp = df.iloc[i:i+steps+1]
        unique_dates = temp['date'].unique()
        date_counts = temp['date'].value_counts()
        #print(f"Lenght of temp: {len(temp)}")
        #print(f"Unique dates: {unique_dates}")
        #print(f"Date counts: {date_counts}")
        # Check that there are only 2 unique dates
        assert len(unique_dates) == hours+1, "There are more than 2 unique dates in the sequence"
        # Check that the last row is the only one with the second date
        assert hours == 1 and date_counts.iloc[-1] == 1 and date_counts.iloc[0] == steps, "The last row is not the only one with the second date"
        # Get y
        y.append(temp['% Silica Concentrate'].iloc[-1])
        # Set silica and iron to -1 for the last vector in X
        temp['% Silica Concentrate'].iloc[-1] = -1
        temp['% Iron Concentrate'].iloc[-1] = -1
        # Get X
        # drop the date column
        temp = temp.drop(columns=['date'])
        X.append(temp)
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Remove dates between
    if remove_outliers:
        date_lower1 = pd.to_datetime("2017-05-13 00:00:00")
        date_upper1 = pd.to_datetime("2017-06-15 00:00:00")
        date_lower2 = pd.to_datetime("2017-07-10 00:00:00")
        date_upper2 = pd.to_datetime("2017-08-10 00:00:00")
        indices = []
        index_to_date = {v: k for k, v in date_to_index.items()}
        for i in range(len(X)):
            # Convert index to date object
            date = index_to_date[i]
            if (date_lower1 < date < date_upper1) or (date_lower2 < date < date_upper2):
                indices.append(i)
        X = np.delete(X, indices, axis=0)
        y = np.delete(y, indices, axis=0)
        print(f"Removed {len(indices)} sequences")
    if add_dim:
        X = np.expand_dims(X, axis=-1)
    return X, y

def get_lstm_model(input_shape):
    """
    Create the LSTM model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        #tf.keras.layers.LSTM(128),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
    return model

if __name__ == "__main__":
    # Clear backend
    tf.keras.backend.clear_session()
    df = load_to_dataframe(remove_first_days=True)
    #check_data_timeseries(df)
    # Get X and y
    X, y = get_data_hourly(df, hours=1, add_dim = False,remove_outliers=True)
    print(y)
    print(X)
    print(f"X shape: {X.shape}")    # (number of sequences, sequence length, number of features) = (3646, 181, 23)
    print(f"y shape: {y.shape}")
    # Shuffle the data
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    # Split the data
    split_idx = int(len(X)*0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Scale the data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    X_test = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Create the model
    model = get_lstm_model(X_train.shape[1:])
    model.summary()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, early_stopping_callback], batch_size=128)
    
    # Unscale y values
    y_pred = model.predict(X_test).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    # calculate mse and mae
    mse = np.mean((y_pred-y_test)**2)
    mae = np.mean(np.abs(y_pred-y_test))
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {1-mse/np.var(y_test):.2f}")
    print(f"MAE%: {mae/np.mean(y_test) * 100:.2f}%")