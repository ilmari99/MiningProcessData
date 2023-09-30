"""
DynamicModel that takes in a a stream of sensor data (measured every 20 seconds), and maintains
a prediction of the next hour's silica concentrate.

Model weights are updated every time a new silica measurement (ground truth) is received.

Model also remembers the latest ground truth silica measurement and uses this in the prediction.

The model uses a fixed window of the last NHOURS of sensor data (19*180*NHOURS datarows) to make the prediction.


"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from load_data import load_to_dataframe
from check_data import flatten_n_hours_data

def flatten_n_hours_with_shifts(df, nhours=1, shifts=0,lab_columns=['% Silica Concentrate', "% Iron Concentrate", "% Iron Feed", "% Silica Feed"]):
    """ Flatten the data to one row for N hours, and shift the data by 1 hour shifts times.
    if directly_to_file has a value, then save the flattened data to that file directly after flattening for each shift
    """

    combined_hourly = flatten_n_hours_data(df, exclude_columns=lab_columns, nhours=nhours)
    print(f"Combined shape: {combined_hourly.shape}")
    # Do combinations with different shifts. So first, no shift, just group by 12 hours
    # Then shift by 1 hour, and group by 12 hours
    # Then shift by 2 hours, and group by 12 hours etc.
    for measure_shift in range(shifts):
        # One hour shift forward; we remove the first hour of data, and then group by 12 hours
        df = df.iloc[1:]
        shifted_combined = flatten_n_hours_data(df, exclude_columns=lab_columns, nhours=nhours)
        print(f"Shifted shape: {shifted_combined.shape}")
        combined_hourly = pd.concat([combined_hourly, shifted_combined], axis=0)
        print(f"Combined shape: {combined_hourly.shape}")
        assert combined_hourly.shape[1] == 19*nhours*180 + nhours*4 + 1, "Wrong number of columns"
        # Assert no duplicates
        assert not combined_hourly.columns.duplicated().any(), f"Found duplicates in columns: {combined_hourly.columns[combined_hourly.columns.duplicated()]}"
    return combined_hourly

class DynamicModel:
    """
    Attributes:
    - X_received : Dataframe : All data received from the sensors
    - Y_received : Dataframe : All ground truth lab measures
    
    - X_window : Dataframe : The last NHOURS hours of sensor data
    - Y_window : Dataframe : The last NHOURS hours of ground truth lab measures
    
    - X : Dataframe : The last NHOURS hours of sensor data, flattened in windows of NHOURS hours
    - Y : Dataframe : The labels for X, i.e. the next hours lab measures
    
    - Y_latest : float : The latest ground truth silica concentrate
    - predictive_model : PLSRegression : The model used to make predictions
    
    Methods:
    - fit_model() -> None : Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
    The received data is flattened in windows and regressed over the next hours lab measures.
    
    - predict(X,Y) -> float : Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
    X is flattened
    
    - observe(sensor_data, lab_data) -> None : Add the latest sensor and lab data to the model. If lab_data is not None, then also update the model.
    
    - _update_window(sensor_data, lab_data) -> None : Update the X_window and Y_window with the latest sensor and lab data.
    
    - add_latest_data_to_XY(sensor_data, lab_data) -> None : Flatten the latest NHOURS hours of sensor and lab data, and add to X and Y.
    
    - _received_data_to_predictive_data(sensor_data, lab_data) -> X : Convert the received data to the format used by the predictive model.
    The dataframe X also contains 'start_date' and 'end_date' columns, which are the start and end dates of the window.
    """
    def __init__(self, NHOURS=24, n_components=2):
        self.NHOURS = NHOURS
        self.lab_columns = ['% Silica Concentrate', "% Iron Concentrate", "% Iron Feed", "% Silica Feed"]
        self.n_components = n_components
        
        # The 20 second data received from the sensors
        self.X_received = pd.DataFrame()
        # The hourly lab measures
        self.Y_received = pd.DataFrame()
        
        # The last 180*NHOURS rows of sensor data so data from the last NHOURS hours
        self.X_window = pd.DataFrame()
        
        # The last NHOURS hours of lab measures
        self.Y_window = pd.DataFrame()
        
        # The features used to predict the next hours lab measures
        self.X = pd.DataFrame()
        # The labels for X, i.e. the next hours lab measures
        self.Y = pd.DataFrame()
        # The latest prediction
        self.Y_pred_latest = None
        self.predictive_model = PLSRegression(n_components=self.n_components)
        
    def _set_predictive_dataframes(self):
        """ Find what are the columns of X.
        """
        nrows = 180 * self.NHOURS
        sensor_columns = []
        for lag in range(nrows):
            for col in self.X_received.columns:
                sensor_columns.append(f"{col}_lag{nrows-lag}")
        
        lab_columns = []
        for lag in range(self.NHOURS - 1):
            for col in self.Y_received.columns:
                lab_columns.append(f"{col}_lag{self.NHOURS-lag}")
        #print(f"X columns: {xcolumns}")
        #print(f"Y columns: {ycolumns}")
        
        self.X = pd.DataFrame(columns=[lab_columns + sensor_columns]) 
        self.Y = pd.DataFrame(columns=self.lab_columns)
        
    
    def add_latest_data_to_XY(self):
        """
        This is triggered everytime lab measurements are received.
        Flatten the latest NHOURS hours of sensor and lab data.
        Label the data with the latest lab measurement.
        """
        past_lab_data = self.Y_window.iloc[:-1, :]
        new_lab_data = self.Y_window.iloc[-1, :]
        past_sensor_data = self.X_window
        print(f"Past lab data shape: {past_lab_data.shape}")
        print(f"New lab data shape: {new_lab_data.shape}")
        print(f"Past sensor data shape: {past_sensor_data.shape}")
        
        # If X is empty
        if self.X.shape[0] == 0:
            self._set_predictive_dataframes()
        
        # Flatten the data
        past_nhours = np.concatenate([past_lab_data.values.flatten(),
                                      past_sensor_data.values.flatten(),
                                      ], axis=0)
        
        new_lab_data = pd.DataFrame(new_lab_data.values.reshape(1, -1), columns=self.lab_columns)
        past_nhours = pd.DataFrame(past_nhours.reshape(1, -1), columns=self.X.columns)
        
        # Add the flattened data to Y and X
        # Concat and keep self.lab_columns order
        self.Y = pd.concat([self.Y, new_lab_data], axis=0)
        self.X = pd.concat([self.X, past_nhours], axis=0)
        
    def _data_to_predictive_data(self, sensor_data, lab_data):
        """ Convert two dataframes of sensor and lab data to the format used by the predictive model.
        """
        pred_df = pd.DataFrame(columns=self.X.columns)
        # Loop over sensor_data in windows of 180*NHOURS rows and flatten the data
        # Take a window of NHOURS of the corresponding lab_data and flatten it
        # Concatenate the two, and the flattened row to a new dataframe, with self.X.columns
        assert len(sensor_data) >= 180*self.NHOURS, "Wrong number of rows"
        for i in range(len(sensor_data) - 180*self.NHOURS + 1):
            past_sensor_data = sensor_data.iloc[i:i+180*self.NHOURS, :]
            past_lab_data = lab_data.iloc[i:i+self.NHOURS-1, :]
            past_lab_flattened = past_lab_data.values.flatten()
            past_sensor_data_flattened = past_sensor_data.values.flatten()
            past_nhours = np.concatenate([past_lab_flattened, past_sensor_data_flattened], axis=0)
            past_nhours = pd.DataFrame(past_nhours.reshape(1, -1), columns=self.X.columns)
            pred_df = pd.concat([pred_df, past_nhours], axis=0)
        return pred_df
        
        
    def predict_on_latest_data(self):
        """ Predict the next hours lab measures from the last NHOURS hours of data (including lab measures).
        """
        # Flatten the data
        predictive_data = self._data_to_predictive_data(self.X_window, self.Y_window)
        # Predict
        Y_preds = self.predictive_model.predict(predictive_data)
        # Return the prediction
        return Y_preds
    
    def _update_window(self, sensor_data, lab_data):
        """ Update the X_window and Y_window with the latest sensor and lab data.
        """
        # Add the sensor data
        self.X_window = pd.concat([self.X_window, sensor_data], axis=0)
        self.X_window = self.X_window.iloc[-self.NHOURS*180:, :]
        # If we received lab data, then add it
        if lab_data is not None:
            if isinstance(lab_data, pd.Series):
                lab_data = lab_data.values.reshape(1, -1)
                lab_data = pd.DataFrame(lab_data, columns=self.lab_columns)
            self.Y_window = pd.concat([self.Y_window, lab_data], axis=0)
            self.Y_window = self.Y_window.iloc[-self.NHOURS:, :]
        #print(f"X window shape: {self.X_window.shape}")
        #print(f"Y window shape: {self.Y_window.shape}")
            
    def _update_received(self, sensor_data, lab_data):
        """ Update the X_received and Y_received with the latest sensor and lab data.
        """
        # If X_received is empty, set columns
        if self.X_received.shape[0] == 0:
            self.X_received = pd.DataFrame(columns=sensor_data.index)
            self.Y_received = pd.DataFrame(columns=self.lab_columns)
        # Add the sensor data
        self.X_received = pd.concat([self.X_received, sensor_data], axis=0)
        # If we received lab data, then add it
        if lab_data is not None:
            self.Y_received = pd.concat([self.Y_received, lab_data], axis=0)
        #print(f"X received shape: {self.X_received.shape}")
        #print(f"Y received shape: {self.Y_received.shape}")
            
    
    def observe(self, sensor_data, lab_data):
        """ Add the latest sensor and lab data to the model. If lab_data is not None, then also update the model.
        """
        # Add the sensor data
        self._update_received(sensor_data, lab_data)
        self._update_window(sensor_data, lab_data)
        print(f"Received Y data shape: {self.Y_received.shape}")
        print(f"Received X data shape: {self.X_received.shape}")
        if self.Y_received.shape[0] % 24 == 0:
            self.add_latest_data_to_XY()
            if self.Y.shape[0] > 1:
                print(f"Y shape: {self.Y.shape}")
                if (self.Y.shape[0]-2) % 12 == 0:
                    print(f"Updated model...\n\n")
                    # Fit the model
                    self.fit_model()
                self.Y_pred_latest = self.predict_on_latest_data()[0]
        
        
    def fit_model(self):
        """ Fit a model from the last NHOURS hours of data (including lab measures) to the next hours lab measures.
        The X_received and Y_received are flattened in windows and regressed over the next hours lab measures.
        """
        # Fit the model
        self.predictive_model.fit(self.X, self.Y)

if __name__ == "__main__":
    df = load_to_dataframe("miningdata/data.csv", remove_first_days=True)
    dm = DynamicModel(NHOURS=24, n_components=2)
    lab_data_frame = df[dm.lab_columns]
    print(f"Lab data shape: {lab_data_frame.shape}")
    sensor_data_frame = df.drop(labels=dm.lab_columns + ["date"], axis=1)
    print(f"Sensor data shape: {sensor_data_frame.shape}")
    for i, row in df.iterrows():
        if i % 10 == 0:
            print(f"Row {i}")
        lab_data = row[dm.lab_columns]
        #print(lab_data)
        sensor_data = row.drop(labels=dm.lab_columns + ["date"])
        if i % 180 != 0:
            lab_data = None
        dm.observe(sensor_data, lab_data)
        pred = dm.Y_pred_latest
        if pred is not None:
            mean_abs = np.abs(pred - lab_data.iloc[-1, :]).mean()
            silica_abs_error = np.abs(pred[0] - lab_data.iloc[-1, 0])
            print(f"Mean absolute error: {mean_abs}")
            print(f"Silica absolute error: {silica_abs_error}")
            

    exit()
    
    ### This one works. It feeds the data in batches, rather than one by one
    ### It is faster, but not really realistic
    
    # Loop over the data; take 180*NHOURS rows of sensor data, and NHOURS rows of lab data (every 180th row)
    for i in range(0, len(sensor_data_frame) - 180*dm.NHOURS, 180*dm.NHOURS):
        sensor_data = sensor_data_frame.iloc[i:i+180*dm.NHOURS, :]
        lab_indices = np.arange(i, i+180*dm.NHOURS, 180)
        lab_data = lab_data_frame.iloc[lab_indices, :]
        assert len(sensor_data) == 180*dm.NHOURS, "Wrong number of rows"
        assert len(lab_data) == dm.NHOURS, "Wrong number of rows"
        dm.observe(sensor_data, lab_data)
        pred = dm.Y_pred_latest
        print(f"Predicted: {pred}")
        print(f"Ground truth: {lab_data.iloc[-1, :].values}")
        if pred is not None:
            mean_abs = np.abs(pred - lab_data.iloc[-1, :]).mean()
            silica_abs_error = np.abs(pred[0] - lab_data.iloc[-1, 0])
            print(f"Mean absolute error: {mean_abs}")
            print(f"Silica absolute error: {silica_abs_error}")
    
        

