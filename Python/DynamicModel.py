import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from load_data import load_to_dataframe
import tensorflow as tf
from abc import ABC, abstractmethod
import warnings
# Ignore FutureWarnings from Pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

class DynamicModel:
    """
    A class for a dynamic model that predicts lab measurements based on sensor data.
    
    Attributes:
    - NHOURS (int): The number of hours to predict ahead.
    - lab_columns (list): The names of the columns in the lab data.
    - n_components (int): The number of components to use in the PLS regression model.
    - verbose (int): The level of verbosity of the model.
    - update_freq_h (int): The frequency of model updates in hours.
    - is_fitted (bool): Whether the model has been fitted or not.
    - __prev_Y_size (int): The size of the previous Y_received dataframe.
    - max_samples (int): The maximum number of samples to keep in the X and Y dataframes.
    - X_received (pd.DataFrame): The 20 second data received from the sensors.
    - Y_received (pd.DataFrame): The hourly lab measures.
    - X (pd.DataFrame): The features used to predict the next hours lab measures.
    - Y (pd.DataFrame): The labels for X, i.e. the next hours lab measures.
    - Y_pred_latest (None or float): The latest prediction.
    - model (PLSRegression): The PLS regression model used for prediction.
    
    Methods:
    - __init__(self, NHOURS=12, update_freg_h = 24, n_components=2,verbose=0, max_samples=8000): Initializes the model.
    - __getattribute__(self, __name: str): Overrides the default __getattribute__ method.
    - pre_fit(self, sensor_data, lab_data, from_precomputed=False, hour_skips=0): Fits the model on some initial data.
    - _set_predictive_dataframes(self): Finds what are the columns of X.
    - add_data_to_XY(self, sensor_data, lab_data): Adds data to X and Y.
    - _data_to_predictive_data(self, sensor_data, lab_data, include_labels = False,hour_skips=0): Converts two dataframes of sensor and lab data to the format used by the predictive model.
    """
class DynamicModel(ABC):
    def __init__(self, NHOURS=12, update_freg_h = 24, verbose=0, max_samples=8000, model=None):
        self.NHOURS = NHOURS
        self.lab_columns = ['% Silica Concentrate', "% Iron Concentrate", "% Iron Feed", "% Silica Feed"]
        self.verbose = verbose
        self.update_freq_h = update_freg_h
        self.is_fitted = False
        self.__prev_Y_size = 0
        self.max_samples = max_samples
        self.X_normalizer = StandardScaler()
        self.Y_normalizer = StandardScaler()
        
        # The 20 second data received from the sensors
        self.X_received = pd.DataFrame()
        # The hourly lab measures
        self.Y_received = pd.DataFrame()
        
        # The features used to predict the next hours lab measures
        self.X = pd.DataFrame()
        # The labels for X, i.e. the next hours lab measures
        self.Y = pd.DataFrame()
        # The latest prediction
        self.Y_pred_latest = None
        self.model = model

    @abstractmethod
    def _fit_model(self, X, Y):
        """
        Fit model on self.X and self.Y.
        """
        pass

    @abstractmethod
    def _predict(self, X):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        pass

    @abstractmethod
    def _update_model(self, X, Y):
        """ Update the model with new data.
        This is optional, but recommended to implement
        """
        pass

    def _dimension_reduction(self, X):
        """ Reduce the dimensionality of X.
        This is optional. If you want to use dimensionality reduction, then override this method, and use it in the abstract methods.
        """
        return X
        
    def __getattribute__(self, __name: str):
        """
        Override the default behavior of attribute access for the class.
        This adds a shortcut to the latest NHOURS of data.
        """
        if __name == "X_window":
            return self.X_received.iloc[-180*self.NHOURS:, :]
        if __name == "Y_window":
            return self.Y_received.iloc[-self.NHOURS:, :]
        return super().__getattribute__(__name)
        
    def pre_fit(self, sensor_data, lab_data, hour_skips=0):
        """ Fit the model on some initial data, for example the first 2 months of data.
        hour_skips: int, the number of hours to skip between each X and Y pair.
        """
        self.X_received = sensor_data
        self.Y_received = lab_data
        self._set_predictive_dataframes()
        X, Y = self._data_to_predictive_data(self.X_received, self.Y_received, include_labels=True, hour_skips=hour_skips)
        print(f"Prefit X shape: {X.shape}")
        print(f"Prefit Y shape: {Y.shape}")
        self.X = X
        self.Y = Y
        self.fit_model()
        
    def _set_predictive_dataframes(self):
        """ Set the columns of X and Y.
        """
        nrows = 180 * self.NHOURS
        sensor_columns = []
        for lag in range(nrows):
            for col in self.X_received.columns:
                sensor_columns.append(f"{col}_lag{nrows-lag}")
        
        lab_columns = []
        for lag in range(self.NHOURS):
            for col in self.Y_received.columns:
                lab_columns.append(f"{col}_lag{self.NHOURS-lag}")
        #print(f"X columns: {xcolumns}")
        #print(f"Y columns: {ycolumns}")
        
        self.X = pd.DataFrame(columns=[lab_columns + sensor_columns]) 
        self.Y = pd.DataFrame(columns=self.lab_columns[0:2])
        
    
    def add_data_to_XY(self, sensor_data, lab_data):
        """
        This is triggered everytime lab measurements are received.
        Flatten the latest NHOURS hours of sensor and lab data.
        Label the data with the latest lab measurement.
        X contains the latest window of NHOURS hours of sensor data
        X also contains the latest NHOURS of lab data, such that the latest lab measurement occurred one hour ago.
        Note, that the sensor measurements are 20 seconds apart, and the latest sensor measurement is 20 seconds ago.
        Y contains the new (future) lab measurement.
        
        NOTE, that on inference, when the latest lab measurement was taken is not known, and it might ONLY just have occurred.
        
        This is unfortunate, but if the time of the latest lab measurement is known,
        then the model will learn that if the latest lab measurement is just taken, then the next lab measurement will be the same.
        
        """
        
        if self.X.shape[0] == 0:
            self._set_predictive_dataframes()
        
        X_new, Y_new = self._data_to_predictive_data(sensor_data, lab_data, include_labels=True)
        if self.verbose >= 1:
            print(f"Adding data to X and Y with {X_new.shape} X and {Y_new.shape} Y")
        
        # Add the flattened data to Y and X
        # Concat and keep self.lab_columns order
        self.Y = pd.concat([self.Y, Y_new], axis=0)
        self.X = pd.concat([self.X, X_new], axis=0)
        if self.X.shape[0] > self.max_samples:
            self.X = self.X.iloc[-self.max_samples:, :]
            self.Y = self.Y.iloc[-self.max_samples:, :]
        
        
    def _data_to_predictive_data(self, sensor_data, lab_data, include_labels = False,hour_skips=0):
        """ Convert two dataframes of sensor and lab data to the format used by the predictive model.
        """
        pred_df = pd.DataFrame(columns=self.X.columns)
        target_df = pd.DataFrame(columns=self.lab_columns[0:2])
        # Loop over sensor_data in windows of 180*NHOURS rows and flatten the data
        # Take a window of NHOURS of the corresponding lab_data and flatten it
        # Concatenate the two, and the flattened row to a new dataframe, with self.X.columns
        assert len(sensor_data) >= 180*self.NHOURS, f"Wrong number of rows: {len(sensor_data)}"
        assert len(lab_data) >= self.NHOURS+1 if include_labels else self.NHOURS, f"Wrong number of rows: {len(lab_data)}"
        #if include_labels:
        #    assert len(sensor_data) == 181*len(lab_data), f"Wrong number of rows: sensor_data: {len(sensor_data)}, lab_data: {len(lab_data)}"
        total_windows = len(sensor_data) - 180*self.NHOURS + 1
        # If sensor_data is exactly 180*(NHOURS-1) rows, then we take all the data in a single window
        for i in range(0,len(sensor_data) - 180*self.NHOURS + 1,max(hour_skips*180,1)):
            # Print progress every 10% of the way
            if total_windows > 10 and i % 10000 == 0:
                print(f"Window {i} of {total_windows}")
            # Take a window of NHOURS*180 rows of sensor data
            past_sensor_data = sensor_data.iloc[i:i+180*self.NHOURS, :]
            # Take the corresponding NHOURS rows of lab data
            nth_hour = i // 180
            past_lab_data = lab_data.iloc[nth_hour:nth_hour+self.NHOURS, :]
            # Label if include_labels is True
            if include_labels:
                next_hour_lab_data = lab_data.iloc[nth_hour+self.NHOURS : nth_hour+self.NHOURS + 1, 0:2]
                #next_hour_lab_data = pd.DataFrame(next_hour_lab_data.values.reshape(1, -1), columns=self.lab_columns)
                target_df = pd.concat([target_df, next_hour_lab_data], axis=0, ignore_index=True)
            
            past_lab_flattened = past_lab_data.values.flatten()
            past_sensor_data_flattened = past_sensor_data.values.flatten()
            
            # So we have NHOURS of sensor data, but only NHOURS-1 of lab data
            past_nhours = np.concatenate([past_lab_flattened, past_sensor_data_flattened], axis=0)
            past_nhours = pd.DataFrame(past_nhours.reshape(1, -1), columns=self.X.columns)
            pred_df = pd.concat([pred_df, past_nhours], axis=0, ignore_index=True)
        if include_labels:
            return pred_df, target_df
        return pred_df
        
        
    def predict_on_latest_data(self):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        sensor_window = self.X_received.iloc[-180*self.NHOURS:, :]
        lab_window = self.Y_received.iloc[-self.NHOURS:, :]
        if (not self.is_fitted) or lab_window.shape[0] < self.NHOURS:
            return None
        # Flatten the data
        predictive_data = self._data_to_predictive_data(sensor_window, lab_window)
        predictive_data = self.X_normalizer.transform(predictive_data)
        # Predict
        Y_preds = self._predict(predictive_data)
        Y_preds = self.Y_normalizer.inverse_transform(Y_preds)
        # Return the prediction
        return Y_preds
    
    
    def _update_received(self, sensor_data, lab_data):
        """ Update the X_received and Y_received with the latest sensor and lab data.
        """
        # If X_received is empty, set columns
        if self.X_received.shape[0] == 0:
            X_received_cols = sensor_data.columns if isinstance(sensor_data, pd.DataFrame) else sensor_data.index
            self.X_received = pd.DataFrame(columns=X_received_cols)
            self.Y_received = pd.DataFrame(columns=self.lab_columns)
        # Add the sensor data
        self.X_received = pd.concat([self.X_received, sensor_data], axis=0)
        # If we received lab data, then add it
        if lab_data is not None:
            assert lab_data.shape[1] == len(self.lab_columns) == self.Y_received.shape[1], f"Wrong number of columns: lab_data: {lab_data.shape[1]}, self.lab_columns: {len(self.lab_columns)}, self.Y_received: {self.Y_received.shape[1]}"
            self.Y_received = pd.concat([self.Y_received, lab_data], axis=0,ignore_index=True)
        
        print(f"X received shape: {self.X_received.shape}")
        print(f"Y received shape: {self.Y_received.shape}")
            
    
    def observe(self, sensor_data, lab_data):
        """ Add the latest sensor and lab data to the model. If lab_data is not None, then also update the model.
        """
        if not isinstance(sensor_data, pd.DataFrame):
            # Set column labels as index
            sensor_data = pd.DataFrame(sensor_data).T
            #print(f"Casting sensor data to dataframe: {sensor_data}")
        if lab_data is not None and not isinstance(lab_data, pd.DataFrame):
            lab_data = pd.DataFrame(lab_data).T
            #print(f"Casting lab data to dataframe: {lab_data}")
        
        # Add the sensor data to X_received, Y_received and update X_window, Y_window
        self._update_received(sensor_data, lab_data)


        # Everytime we receive lab data, we want a window of NHOURS of sensor data, and the following hour of lab data,
        # to be added to X and Y and later used for updating a model
        if lab_data is not None:
            if self.verbose > 1:
                print(f"Received lab data with {lab_data.shape[0]} rows")
            
            # If we have enough data, then update X and Y
            if self.X_window.shape[0] >= 180*self.NHOURS and self.Y_received.shape[0] >= self.NHOURS + 1:
                # The input data has already been added to X_received and Y_received.

                # We want to convert the received data, to the predictive data format X -> Y
                # The Y is the next hours measurement, and X contains the past NHOURS of sensor data and lab measures
                X_window = self.X_window
                # Shhift the Y_window by one hour, so that the latest lab measurement is one hour ago
                Y_window = self.Y_received.iloc[-self.NHOURS - 2:-1, :]

                print(f"Adding data to X and Y with {X_window.shape} X and {Y_window.shape} Y")
                self.add_data_to_XY(X_window, Y_window)
        
        print(f"Prev Y size: {self.__prev_Y_size}")
        print(f"Y size: {self.Y.shape[0]}")
        # if we have reached the update frequency, then train a model
        if (self.Y.shape[0] >= 2) and (self.Y.shape[0] - self.__prev_Y_size > self.update_freq_h) and (lab_data is not None):
            print(f"Updating model with {self.X_received.shape} X and {self.Y_received.shape} Y")
            if self.is_fitted:
                self.update_model()
            else:
                self.fit_model()
        
        
    def fit_model(self):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        print(f"Fitting model with {self.X.shape} X and {self.Y.shape} Y")
        self.is_fitted = True

        self.X_normalizer.fit(self.X)
        self.X = pd.DataFrame(self.X_normalizer.transform(self.X), columns=self.X.columns)
        self.Y_normalizer.fit(self.Y.iloc[:,0:2])
        self.Y = pd.DataFrame(self.Y_normalizer.transform(self.Y), columns=self.lab_columns[0:2])

        self._fit_model(self.X, self.Y)
        self.__prev_Y_size = self.Y.shape[0]
        # Empty X and Y
        #self.X = pd.DataFrame(columns=self.X.columns)
        #self.Y = pd.DataFrame(columns=self.lab_columns)

    def update_model(self):
        """ Update the model with new data.
        This is optional, but recommended to implement
        """
        # Rescale X and Y
        #self.X_normalizer.fit(self.X)
        #self.Y_normalizer.fit(self.Y)
        self.X = pd.DataFrame(self.X_normalizer.transform(self.X), columns=self.X.columns)
        self.Y = pd.DataFrame(self.Y_normalizer.transform(self.Y), columns=self.lab_columns[0:2])
        print(f"Sizes after transform X,Y: {self.X.shape}, {self.Y.shape}")
        print(f"__prev_Y_size: {self.__prev_Y_size}")
        # Update the model
        X = self.X.iloc[self.__prev_Y_size:, :]
        Y = self.Y.iloc[self.__prev_Y_size:, :]
        print(f"Updating model with {X.shape} X and {Y.shape} Y")
        self.__prev_Y_size = self.Y.shape[0]
        self._update_model(X, Y)

class OnlySilicaMSELoss(tf.keras.losses.Loss):
    def __init__(self, name="only_silica_mse_loss"):
        super().__init__(name=name)
    def call(self, y_true, y_pred):
        return tf.keras.losses.MSE(y_true[:, 0], y_pred[:, 0])

class DynamicNeuralNetModel(DynamicModel):
    def __init__(self, NHOURS=12, update_freg_h = 24,verbose=0, max_samples=8000):
        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        super().__init__(NHOURS, update_freg_h, verbose, max_samples, self.model)
        
    def _fit_model(self, X, Y):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        self.X_val = X.iloc[-50:, :]
        self.Y_val = Y.iloc[-50:, :]
        X = X.iloc[:-50, :]
        Y = Y.iloc[:-50, :]
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X, Y, epochs=100, verbose=1, validation_data=(self.X_val, self.Y_val), callbacks=[early_stop])

    def _predict(self, X):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        return self.model.predict(X, verbose=0).reshape(-1, 2)
    
    def _update_model(self, X, Y):
        """ Update the model with new data.
        This is optional, but recommended to implement
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(X, Y, epochs=5, verbose=1, validation_data=(self.X_val, self.Y_val))

class DynamicPLSModel(DynamicModel):
    def __init__(self, NHOURS=12, update_freg_h = 24, n_components=20,verbose=0, max_samples=8000):
        self.model = PLSRegression(n_components=n_components)
        self.n_components = n_components
        super().__init__(NHOURS, update_freg_h, verbose, max_samples, self.model)
        
    def _fit_model(self, X, Y):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        self.model.fit(X, Y)

    def _predict(self, X):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        return self.model.predict(X).reshape(-1, 2)
    
    def _update_model(self, X, Y):
        """ Calculate a PLS on the full X and Y
        """
        X = self.X
        Y = self.Y
        self._fit_model(X, Y)



if __name__ == "__main__":
    df = load_to_dataframe("miningdata/data.csv", remove_first_days=True)
    #dm = DynamicNeuralNetModel(NHOURS=6, update_freg_h=24, verbose=1,max_samples=6000)
    dm = DynamicPLSModel(NHOURS=3, update_freg_h=24, verbose=1,max_samples=6000, n_components=12)
    
    prefit_df = df.iloc[:180*24*120, :]
    pre_fit_lab_data = prefit_df[dm.lab_columns][::180]
    pre_fit_sensor_data = prefit_df.drop(labels=dm.lab_columns + ["date"], axis=1)
    assert 180*pre_fit_lab_data.shape[0] == pre_fit_sensor_data.shape[0], f"Wrong number of rows: pre_fit_lab_data: {pre_fit_lab_data.shape[0]}, pre_fit_sensor_data: {pre_fit_sensor_data.shape[0]}"

    dm.pre_fit(pre_fit_sensor_data, pre_fit_lab_data, hour_skips=5)

    df = df.iloc[180*24*120:, :]
    
    print(f"Pre-fitting done, dynamically fitting model with {df.shape} rows")
    # Lab data is measured every hour, so we only need every 180th row
    lab_data_frame = df[dm.lab_columns][::180]
    print(f"Lab data shape: {lab_data_frame.shape}")
    sensor_data_frame = df.drop(labels=dm.lab_columns + ["date"], axis=1)
    print(f"Sensor data shape: {sensor_data_frame.shape}")
    
    
    silica_mae_list = []
    total_mae_list = []
    updated_bools = []
    
    # When a DynamicModel is used in it's actual setting, K=1
    # When we want to test the model, we can use a larger K (multiple of 180), to go through the data faster
    K = 180*6
    STEP_SIZE = K

    total_windows = len(df) - K + 1
    prev_nth_hour = -1
    print(len(df))
    lab_data = None
    # i is the initial row of the window
    for i in range(0, len(df) - K, STEP_SIZE):
        if total_windows > 1 and i % (total_windows // 10000) == 0:
            print(f"Window {i} of {total_windows}")
        sensor_data = sensor_data_frame.iloc[i:i+K, :]

        # Lets give lab data every 180 rows
        nth_hour = i // 180
        #print(f"nth_hour: {nth_hour}, i: {i}")
        if nth_hour != prev_nth_hour:
            print(f"nth_hour: {nth_hour}, prev_nth_hour: {prev_nth_hour}")
            # If K > 180, then we need to give more hours of lab data at once
            # For example if K = 180, then we need to give 2 hours of lab data, if K = 360, then 3 hours of lab data
            end_nth_hour = nth_hour + max(K // 180,1)
            #print(f"end_nth_hour: {end_nth_hour}")
            lab_data = lab_data_frame.iloc[nth_hour:end_nth_hour, :]
            #print(lab_data)
            prev_nth_hour = nth_hour
        updated_bools.append(True if i % (dm.update_freq_h * 180) == 0 else False)
        
        # Show the data to the DynamicModel
        dm.observe(sensor_data, lab_data)
        # Get a prediction, that is based on the last NHOURS of data
        Y_pred = dm.predict_on_latest_data()
        if Y_pred is not None and lab_data is not None:
            # Get the next hours lab measures
            Y_true = lab_data_frame.iloc[nth_hour + max(K // 180,1) : nth_hour + max(K // 180,1) + 1, 0:2].values
            silica_mae = mean_absolute_error(Y_true[:, 0], Y_pred[:, 0])
            silica_mae_list.append(silica_mae)
            total_mae = mean_absolute_error(Y_true, Y_pred)
            total_mae_list.append(total_mae)
            print(f"Silica MAE: {silica_mae}")
            print(f"Total MAE: {total_mae}")
        lab_data = None
    
    
    print(f"Silica MAE over full run: {np.mean(silica_mae_list)}")
    print(f"Total MAE over full run: {np.mean(total_mae_list)}")
    print(f"Total silica MAE over last 50 measures: {np.mean(silica_mae_list[-50:])}")
    print(f"Total MAE over last 50 measures: {np.mean(total_mae_list[-50:])}")
    
    fig, ax = plt.subplots()
    ax.plot(silica_mae_list)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error in silica prediction")
    # Plot the points, where the model was updated
    for i, updated in enumerate(updated_bools):
        if updated:
            ax.scatter(i, silica_mae_list[i], color="red")
    
    fig, ax = plt.subplots()
    ax.plot(total_mae_list)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error in total prediction of all lab measures")
    # Plot the points, where the model was updated
    for i, updated in enumerate(updated_bools):
        if updated:
            ax.scatter(i, total_mae_list[i], color="red")
    
    plt.show()
    
        

