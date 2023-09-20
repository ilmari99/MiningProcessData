""" Train an autoencoder model, that takes in an hour of measurements

"""
from sklearn.model_selection import train_test_split
from load_data import load_to_dataframe
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from check_data import compute_outlier_mask
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf

class CustomMSE(tf.keras.losses.Loss):
    """ Only calculate loss != -1 VALUES"""
    def call(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return tf.keras.losses.mse(y_true, y_pred)

def get_model(input_shape, latent_dim=128):
    """ Return a tensorflow autoencoder model with an encoder-decoder architecture,
    that encode the input to a latent space of size latent_dim, and then decodes it back to the input.

    The input size is roughly 3500
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_shape[0], activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=CustomMSE())
    return model

if __name__ == "__main__":
    df = pd.read_csv("miningdata/data_flattened_hourly.csv", parse_dates=['date'])
    df = df.drop(columns=['date'])
    outlier_mask = compute_outlier_mask(df, outlier_z_thresh=3.5)
    print(f"Number of outliers:\n{outlier_mask.sum()}")

    # Split the data into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=False)
    outlier_mask = compute_outlier_mask(X_train, outlier_z_thresh=3.5)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Set outliers to -1
    X_train[outlier_mask] = -1

    Y_train = X_train.copy()
    Y_test = X_test.copy()

    model = get_model(input_shape=(X_train.shape[1],))
    model.summary()
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))
    model.save("miningdata/autoencoder_model.h5")


    



