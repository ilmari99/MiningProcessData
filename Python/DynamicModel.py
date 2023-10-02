import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
from load_data import load_to_dataframe

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
    - __prev_Y_received_size (int): The size of the previous Y_received dataframe.
    - max_samples (int): The maximum number of samples to keep in the X and Y dataframes.
    - X_received (pd.DataFrame): The 20 second data received from the sensors.
    - Y_received (pd.DataFrame): The hourly lab measures.
    - X (pd.DataFrame): The features used to predict the next hours lab measures.
    - Y (pd.DataFrame): The labels for X, i.e. the next hours lab measures.
    - Y_pred_latest (None or float): The latest prediction.
    - predictive_model (PLSRegression): The PLS regression model used for prediction.
    
    Methods:
    - __init__(self, NHOURS=12, update_freg_h = 24, n_components=2,verbose=0, max_samples=8000): Initializes the model.
    - __getattribute__(self, __name: str): Overrides the default __getattribute__ method.
    - pre_fit(self, sensor_data, lab_data, from_precomputed=False, hour_skips=0): Fits the model on some initial data.
    - _set_predictive_dataframes(self): Finds what are the columns of X.
    - add_data_to_XY(self, sensor_data, lab_data): Adds data to X and Y.
    - _data_to_predictive_data(self, sensor_data, lab_data, include_labels = False,hour_skips=0): Converts two dataframes of sensor and lab data to the format used by the predictive model.
    """
class DynamicModel:
    def __init__(self, NHOURS=12, update_freg_h = 24, n_components=2,verbose=0, max_samples=8000):
        self.NHOURS = NHOURS
        self.lab_columns = ['% Silica Concentrate', "% Iron Concentrate", "% Iron Feed", "% Silica Feed"]
        self.n_components = n_components
        self.verbose = verbose
        self.update_freq_h = update_freg_h
        self.is_fitted = False
        self.__prev_Y_received_size = 0
        self.max_samples = max_samples
        
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
        self.predictive_model = PLSRegression(n_components=self.n_components, scale = False)
        
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
        self.Y = pd.DataFrame(columns=self.lab_columns)
        
    
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
        
        # Add the flattened data to Y and X
        # Concat and keep self.lab_columns order
        self.Y = pd.concat([self.Y, Y_new], axis=0)
        self.X = pd.concat([self.X, X_new], axis=0)
        if self.X.shape[0] > self.max_samples:
            self.X = self.X.iloc[-self.max_samples:, :]
            self.Y = self.Y.iloc[-self.max_samples:, :]
        if self.verbose > 1:
            print(f"X shape: {self.X.shape}")
            print(f"Y shape: {self.Y.shape}")
        
        
    def _data_to_predictive_data(self, sensor_data, lab_data, include_labels = False,hour_skips=0):
        """ Convert two dataframes of sensor and lab data to the format used by the predictive model.
        """
        pred_df = pd.DataFrame(columns=self.X.columns)
        target_df = pd.DataFrame(columns=self.lab_columns)
        # Loop over sensor_data in windows of 180*NHOURS rows and flatten the data
        # Take a window of NHOURS of the corresponding lab_data and flatten it
        # Concatenate the two, and the flattened row to a new dataframe, with self.X.columns
        assert len(sensor_data) >= 180*self.NHOURS, f"Wrong number of rows: {len(sensor_data)}"
        assert len(lab_data) >= self.NHOURS, f"Wrong number of rows: {len(lab_data)}"
        #if include_labels:
        #    assert len(sensor_data) == 181*len(lab_data), f"Wrong number of rows: sensor_data: {len(sensor_data)}, lab_data: {len(lab_data)}"
        total_windows = len(sensor_data) - 180*self.NHOURS + 1
        # If sensor_data is exactly 180*(NHOURS-1) rows, then we take all the data in a single window
        for i in range(0,len(sensor_data) - 180*self.NHOURS + 1,max(hour_skips*180,1)):
            if total_windows > 10 and i % (total_windows // 10) == 0:
                print(f"Window {i} of {total_windows}")
            # Take a window of NHOURS*180 rows of sensor data
            past_sensor_data = sensor_data.iloc[i:i+180*self.NHOURS, :]
            # Take the corresponding NHOURS rows of lab data
            nth_hour = i // 180
            past_lab_data = lab_data.iloc[nth_hour:nth_hour+self.NHOURS, :]
            # Label if include_labels is True
            if include_labels:
                next_hour_lab_data = lab_data.iloc[nth_hour+self.NHOURS : nth_hour+self.NHOURS + 1, :]
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
        # Predict
        Y_preds = self.predictive_model.predict(predictive_data)
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
        
        #print(f"X received shape: {self.X_received.shape}")
        #print(f"Y received shape: {self.Y_received.shape}")
            
    
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
        
        if lab_data is not None:
            if self.verbose > 1:
                print(f"Received lab data with {lab_data.shape[0]} rows")
            if lab_data.shape[0] >= self.NHOURS:
                # If the input data contains more than NHOURS of data, then we can directly update X and Y from that data.
                # All we need to do, is make sure the data is correct, and remove the last 180 rows of sensor data.
                if sensor_data.shape[0] != 180*lab_data.shape[0]:
                    print(f"Wrong number of rows: sensor_data: {sensor_data.shape[0]}, lab_data: {lab_data.shape[0]}")
                else:
                    sensor_data = sensor_data.iloc[:-180, :]
                    print(f"Lab data shape: {lab_data.shape}")
                    print(f"Sensor data shape: {sensor_data.shape}")
                    self.add_data_to_XY(sensor_data, lab_data)
                
            elif self.X_window.shape[0] >= 180*self.NHOURS and self.Y_received.shape[0] >= self.NHOURS + 1:
                # If the input contains less than NHOURS of data, then we can't update X and Y,
                # unless we have received lab data.
                # So the input data has already been added to X_received and Y_received.
                # And if we received lab data, then we can update X and Y by taking this new data, and some of the old data.
                Y_window = self.Y_received.iloc[-self.NHOURS-1:, :]
                self.add_data_to_XY(self.X_window, Y_window)
        
        # if we have reached the update frequency, then train a model
        if (self.Y.shape[0] >= 2) and (self.Y_received.shape[0] - self.__prev_Y_received_size > self.update_freq_h) and (lab_data is not None):
            print(f"Updating model with {self.X_received.shape} X and {self.Y_received.shape} Y")
            self.fit_model()
            self.__prev_Y_received_size = self.Y_received.shape[0]
        
        
    def fit_model(self):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        print(f"Fitting model with {self.X.shape} X and {self.Y.shape} Y")
        self.is_fitted = True
        # Fit the model
        self.predictive_model.fit(self.X, self.Y)
        
import tensorflow as tf
class DynamicNeuralNetModel(DynamicModel):
    def __init__(self, NHOURS=12, update_freg_h = 24, n_components=2,verbose=0, max_samples=8000):
        super().__init__(NHOURS, update_freg_h, n_components, verbose, max_samples)
        self.model = None
        self.__prev_Y_received_size = 0
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu')
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        
    def fit_model(self):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        
        print(f"Fitting model with {self.X.shape} X and {self.Y.shape} Y")
        # If the model is already fitted, then we only need to fit it for the new data
        if self.is_fitted:
            X = self.X.iloc[self.__prev_Y_received_size:, :]
            Y = self.Y.iloc[self.__prev_Y_received_size:, :]
            self.model.fit(X, Y, epochs=500, verbose=0)
        else:
            self.model.fit(self.X, self.Y, epochs=50, verbose=0)
        self.is_fitted = True
        
    def predict_on_latest_data(self):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        sensor_window = self.X_received.iloc[-180*self.NHOURS:, :]
        lab_window = self.Y_received.iloc[-self.NHOURS:, :]
        if (not self.is_fitted) or lab_window.shape[0] < self.NHOURS:
            return None
        # Flatten the data
        predictive_data = self._data_to_predictive_data(sensor_window, lab_window)
        # Predict
        Y_preds = self.model.predict(predictive_data, verbose=0)
        # Return the prediction
        return Y_preds[-1, :].reshape(1, -1)






if __name__ == "__main__":
    df = load_to_dataframe("miningdata/data.csv", remove_first_days=True)
    dm = DynamicNeuralNetModel(NHOURS=6, update_freg_h=24, n_components=8, verbose=1,max_samples=6000)
    
    prefit_df = df.iloc[:180*24*30, :]
    pre_fit_lab_data = prefit_df[dm.lab_columns]
    pre_fit_sensor_data = prefit_df.drop(labels=dm.lab_columns + ["date"], axis=1)
    dm.pre_fit(pre_fit_sensor_data, pre_fit_lab_data, from_precomputed=False, hour_skips = 3)

    df = df.iloc[180*24*30:, :]
    
    print(f"Pre-fitting done, dynamically fitting model with {df.shape} rows")
    # Lab data is measured every hour, so we only need every 180th row
    lab_data_frame = df[dm.lab_columns][::180]
    print(f"Lab data shape: {lab_data_frame.shape}")
    sensor_data_frame = df.drop(labels=dm.lab_columns + ["date"], axis=1)
    print(f"Sensor data shape: {sensor_data_frame.shape}")
    
    
    silica_mae_list = []
    total_mae_list = []
    
    # When a DynamicModel is used in it's actual setting, K=1
    # When we want to test the model, we can use a larger K (multiple of 180), to go through the data faster
    K = 17*180
    total_windows = len(df) - K + 1
    for i in range(0, len(df)):
        if total_windows > 1 and i % (total_windows // 10000) == 0:
            print(f"Window {i} of {total_windows}")
        sensor_data = sensor_data_frame.iloc[i:i+K, :]
        lab_data = lab_data_frame.iloc[i:i+K:180, :]
        # Show the data to the DynamicModel
        dm.observe(sensor_data, lab_data)
        # Get a prediction, that is based on the last NHOURS of data
        Y_pred = dm.predict_on_latest_data()
        if Y_pred is not None:
            Y_true = lab_data.iloc[-1, :].to_numpy().reshape(1, -1)
            silica_mae = mean_absolute_error(Y_true[:, 0], Y_pred[:, 0])
            silica_mae_list.append(silica_mae)
            total_mae = mean_absolute_error(Y_true, Y_pred)
            total_mae_list.append(total_mae)
            print(f"Silica MAE: {silica_mae}")
            print(f"Total MAE: {total_mae}")
    
    
    print(f"Silica MAE: {np.mean(silica_mae_list)}")
    print(f"Total MAE: {np.mean(total_mae_list)}")
    
    fig, ax = plt.subplots()
    ax.plot(silica_mae_list)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error in silica prediction")
    
    fig, ax = plt.subplots()
    ax.plot(total_mae_list)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute error in total prediction of all lab measures")
    
    plt.show()
    
        

