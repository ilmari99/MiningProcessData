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
    """ Only calculate loss for values that are not -1
    """
    def call(self, y_true, y_pred):
        mask = tf.math.not_equal(y_true, -1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return tf.keras.losses.MSE(y_true, y_pred)

def get_model(input_shape, latent_dim=128):
    """ Return a tensorflow autoencoder model with an encoder-decoder architecture,
    that encode the input to a latent space of size latent_dim, and then decodes it back to the input.

    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2*latent_dim, activation='relu',kernel_regularizer="l2")(inputs)
    x = tf.keras.layers.Dense(2*latent_dim, activation='relu')(x)
    
    x = tf.keras.layers.Dense(latent_dim, activation='relu',name='latent_space')(x)
    
    x = tf.keras.layers.Dense(2*latent_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(2*latent_dim, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(input_shape[0], activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=CustomMSE(), metrics=['mse', 'mae'])
    return model

if __name__ == "__main__":
    df = pd.read_csv("miningdata/data_flattened_12hourly_multishift_train.csv", parse_dates=['date'])
    cols_to_drop = ["date", "% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"]
    # Get all columns, with any of the strings in cols_to_drop
    cols_to_drop = df.columns[df.columns.str.contains('|'.join(cols_to_drop))]
    df = df.drop(columns=cols_to_drop)

    # Split the data into train and test sets
    X_train, X_test = train_test_split(df, test_size=0.1, shuffle=True)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Set outliers to -1
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    xtrain_outlier_mask = compute_outlier_mask(X_train, outlier_z_thresh=3.5)
    print(f"Number of outliers in train set:\n{sum(xtrain_outlier_mask.sum())}")
    #X_train[xtrain_outlier_mask] = -1
    #X_test[xtest_outlier_mask] = -1

    Y_train = X_train.copy()
    Y_test = X_test.copy()
    Y_train[xtrain_outlier_mask] = -1

    model = get_model(input_shape=(X_train.shape[1],), latent_dim=256)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tblogs", histogram_freq=1)
    
    
    model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stop, tensorboard])
    
    encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer('latent_space').output)
    encoder.summary()
    encoder.save("models/encoder_hourly_256.h5")
    
    decoder = tf.keras.Model(inputs=model.get_layer('latent_space').output, outputs=model.output)
    decoder.summary()
    decoder.save("models/decoder_hourly_256.h5")

    



