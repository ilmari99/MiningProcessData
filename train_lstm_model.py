""" 
In this file we train an LSTM model to predict the next flotation phase vector
So, an LSTM which takes as input the flotation phase vector at time t (while remembering the previous flotation phase vectors)
and outputs the flotation phase vector at time t+1
"""

import tensorflow as tf
from load_data import load_to_dataframe
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

class CustomMSE(tf.keras.losses.Loss):
    """ Custom loss function, that ignores some columns at times, since the data has malfunctioning values
    """
    def __init__(self, data, window_size=50, name="custom_mse"):
        super().__init__(name=name)
        self.window_size = window_size
        self.adapt(data)
        print(f"Value thresholds:\n{self.value_thresholds}")
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        # Calculate the loss for each column
        loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
        # Ignore the loss for the columns that are not within the threshold
        loss = tf.where(tf.logical_and(y_true > self.value_thresholds[:,0], y_true < self.value_thresholds[:,1]), loss, tf.zeros_like(loss))
        # Return the mean of the loss
        return tf.reduce_mean(loss)
    
    def adapt(self, dataframe):
        """ Adapt the loss function to the data
        by calculating thresholds for each column, where if a value is not within the threshold, it is ignored
        """
        # Calculate the mean and std for each column
        mean = dataframe.mean(axis=0)
        std = dataframe.std(axis=0)
        # Calculate the threshold for each column
        upper_threshold = mean + 3*std
        lower_threshold = mean - 3*std
        # Save the threshold
        self.value_thresholds = np.stack((lower_threshold, upper_threshold), axis=1)
        return
    

def get_model(input_shape, output_length, loss):
    """ Create the model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(output_length))
    model.compile(optimizer='adam', loss=loss)
    return model

def make_dataset(data, window_size):
    """ Make the dataset
    """
    X = []
    y = []
    for i in range(0,len(data) - window_size - 1,window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Load the data
data = load_to_dataframe(remove_first_days=True)

train_data = data[data['date'] < '2017-08-15'].drop(columns=['date'])
test_data = data[data['date'] >= '2017-08-15'].drop(columns=['date'])

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


# Make the dataset
window_size = 300
custom_loss = CustomMSE(train_data, window_size=window_size)
X_train, y_train = make_dataset(train_data, window_size)
X_test, y_test = make_dataset(test_data, window_size)
# Shuffle the data
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = get_model((X_train.shape[1], X_train.shape[2]), len(data.columns)-1, custom_loss)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='min')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
print(model.summary())

history = model.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping, tensorboard])
#model.save('lstm_model.h5')
#model = tf.keras.models.load_model('lstm_model.h5')

# Calculate mean change in each column
mae = np.mean(np.abs(y_test[1:] - y_test[:-1]), axis=0)
mse = np.mean(np.square(y_test[1:] - y_test[:-1]), axis=0)
print(f"MAE when using current state as next state: {mae}")
print(f"MSE when using current state as next state: {mse}")

# Predict the test data
y_pred = model.predict(X_test)
print(f"y_pred shape: {y_pred.shape}")
# Plot predicted and true columns 0:2 and 7:9
cols_to_predict = [0,1,2,7,8,9]
for i in cols_to_predict:
    fig, ax = plt.subplots()
    ax.plot(y_test[:,i], label='true')
    ax.plot(y_pred[:,i], label='pred')
    ax.set_title(data.columns[i+1])
    ax.legend()
# Calculate MAE for each column
mae = np.mean(np.abs(y_pred - y_test), axis=0)
mse = np.mean(np.square(y_pred - y_test), axis=0)
print(f"MAE on predictions: {mae}")
print(f"MSE on predictions: {mse}")
plt.show()

# Start from the first X_test value, predict the next value, append it to X_test, and repeat for 100 times
X_values = X_test[0]
for i in range(window_size):
    pred = model.predict(np.expand_dims(X_values, axis=0),verbose=0)
    X_values = np.concatenate((X_values[1:], pred), axis=0)

# Calculate MAE for each column
mae = np.mean(np.abs(X_values - y_test[:window_size]), axis=0)
mse = np.mean(np.square(X_values - y_test[:window_size]), axis=0)
print(f"MAE on recursive predictions: {mae}")
print(f"MSE on recursive predictions: {mse}")

# Plot the predicted values
for i in cols_to_predict:
    fig, ax = plt.subplots()
    ax.plot(X_values[1:,i], label='pred')
    ax.plot(y_test[:window_size,i], label='true')
    ax.set_title(data.columns[i+1])
    ax.legend()
plt.show()












