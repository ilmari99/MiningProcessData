""" Visualize the timeseries data in 'miningdata/data.csv' 
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Disable sns deprecation warning
import warnings

from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from load_data import load_to_dataframe, train_test_validation_split_sequence
from scipy import stats
import pandas as pd

STATIONARY_COLUMNS = ["date", "% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"]

def compute_outlier_mask(df, outlier_z_thresh = 3, return_abs_z_scores=False):
    """ Return a dataframe with True values at indices of outliers, and False else where
    """
    has_date = 'date' in df.columns
    if has_date:
        df = df.drop(columns=['date'])
    print(f"Shape of df: {df.shape}")
    print(f"dtypes of df: {df.dtypes}")
    # Compute the z-scores for each feature
    z_scores = stats.zscore(df)
    # Compute the absolute z-scores
    abs_z_scores = np.abs(z_scores)
    # Create a mask, same ssize as df, with True values at indices of outliers, and False else where
    outlier_mask = abs_z_scores > outlier_z_thresh
    nan_mask = np.isnan(df)
    # Combine the outlier mask and nan mask
    outlier_mask = np.logical_or(outlier_mask, nan_mask)
    if has_date:
        # insert the date column (only False values)
        outlier_mask = np.insert(outlier_mask, 0, False, axis=1)
    if return_abs_z_scores:
        return outlier_mask, abs_z_scores
    return outlier_mask


def basic_check_data(df):
    """ Check the data for missing values
    """
    print(f"Shape of the dataframe: {df.shape}")
    print(f"Data from {df['date'].min()} to {df['date'].max()}")
    print(f"Null values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Columns:\n{df.columns}")
    print(f"Describe:\n{df.describe()}")
    print(f"Z-score max silica:\n{max(stats.zscore(df['% Silica Concentrate']))}")
    print(f"Z-score min silica:\n{min(stats.zscore(df['% Silica Concentrate']))}")
    return
    
def check_data_timeseries(df):
    basic_check_data(df)
    # Check the count of each unique date
    unique_date_counts = df['date'].value_counts()
    print(unique_date_counts)
    return
    
def basic_visualize(df, histograms=False, scatterplots=False, save=True):
    """ Plot images of the timeseries data
    """
    figs_and_axes = []
    if histograms:
        fig,ax = plt.subplots(5,5)
        print(f"Subplot size: {(5,5)}")
        # Plot the histogram for each column
        for col_idx, col in enumerate(df.columns):
            ax_ = ax[col_idx//5, col_idx%5]
            sns.histplot(df[col], ax=ax_,legend=col)
    plt.show()
    
def visualize_timeseries(df, save=True, outlier_indices = [], window_size = 180):
    """ Plot the timeseries data.
    If outlier_indices is not empty, then plot the outliers in red.
    """
    # Plot the timeseries for each column
    fig,ax = plt.subplots(5,5)
    # Divide the data into windows of size window_size
    for col_idx, col in enumerate(df.columns):
        if col == 'date':
            continue
        ax_ : plt.Axes = ax[col_idx//5, col_idx%5]
        ax_.plot(df[col])
        ax_.set_title(col)
        # Plot the outliers in red
        for outlier_idx in outlier_indices:
            ax_.axvspan(outlier_idx*window_size, (outlier_idx+1)*window_size, color='red', alpha=0.5)
    plt.show()


def combine_to_hourly(df : pd.DataFrame, nhours=1, method = lambda group : np.mean(group, axis=0), exclude_columns=['% Silica Concentrate']):
    """ Combine N hours of data to one row using 'method(group : np.arr) -> np.arr'
    """
    # Remove the columns we don't want to apply the method to, except date
    # We later combine these on the combined dates
    removed_cols = df[exclude_columns + ['date']]
    df = df.drop(columns=exclude_columns)
    # Group the data by date
    grouped = df.groupby(pd.Grouper(key='date', freq=f'{nhours}H'))
    # Apply the method to each group
    combined = grouped.apply(method)
    # Add the removed columns back
    # We use the first date in each group as the date for the combined row
    combined = pd.concat([combined, removed_cols.groupby(pd.Grouper(key='date', freq=f'{nhours}H')).first()], axis=1)
    # Remove the NaN values
    combined = combined.dropna()
    # Reset the index
    combined = combined.reset_index(drop=True)
    if "date" not in combined.columns:
        # Add the date column
        combined.insert(0, "date", pd.date_range(start=df['date'].min(), periods=len(combined), freq=f'{nhours}H'))
    # Round the date to the nearest hour
    combined['date'] = combined['date'].dt.round('H')
    # Remove the first row if Nhours is >1
    if nhours > 1:
        combined = combined.iloc[1:]
    return combined

def flatten_n_hours_data(df, exclude_columns = ["% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"], nhours=1):
    """ Flatten all observations from 1 hours to one row, excluding the columns measured only once per hour
    """
    nrows = nhours*180
    exclude_columns = ["date"] + exclude_columns if 'date' not in exclude_columns else exclude_columns
    # The dataframe will have 180 columns for each non-excluded column + the excluded columns
    columns = []
    for i in range(nrows):
        for col in df.columns:
            if col not in exclude_columns:
                columns.append(f"{col}_{i}")
    
    stationary_columns = exclude_columns
    # If more than one hour, then we must maintain lagged values for excluded columns
    if nhours > 1:
        stationary_columns = []
        for i in range(0, nhours):
            for col in exclude_columns:
                if col != 'date':
                    stationary_columns.append(f"{col}_{i}")
        stationary_columns = ["date"] + stationary_columns

    
    assert len(columns) == nhours*180*(len(df.columns) - len(exclude_columns)), f"Excpected {180*len(df.columns) - len(exclude_columns)} columns, got {len(columns)}"
    assert len(stationary_columns) == (len(exclude_columns) - 1) * nhours + 1, f"Expected {(len(exclude_columns) - 1) * nhours + 1} stationary columns, got {len(stationary_columns)}"
    
    all_columns = stationary_columns + columns
    new_columns = columns
    
    combined = pd.DataFrame(columns=all_columns)

    # For each hour (180 consecutive rows)
    total_number_of_rows = len(df)//nrows
    for i in range(total_number_of_rows):
        
        # Get the n*180 rows
        rows = df.iloc[i*nrows:(i+1)*nrows]
        
        # Separate the stationary values, that only change once every 180 rows
        stationary_values = rows[exclude_columns]
        date = stationary_values['date'].iloc[0]
        stationary_values = stationary_values.drop(columns=['date'])
        
        # Only keep every 180th row of the stationary values
        stationary_values = stationary_values.iloc[::180]
        rows = rows.drop(columns=exclude_columns)
        row_values = rows.values.flatten().tolist()
        stationary_values = [date] + stationary_values.values.flatten().tolist()
        
        # Add the values to combined
        observation = stationary_values + row_values

        # Add a row to the dataframe
        combined = pd.concat([combined, pd.DataFrame([observation], columns=all_columns)], axis=0)
        
        # Print progress
        if i % 10 == 0:
            print(f"Progress: {i}/{total_number_of_rows}")
            print(f"Shape of combined: {combined.shape}")
        
    # Reset the index
    combined = combined.reset_index(drop=True)
    # Convert all columns to float except date
    combined = combined.astype({col:float for col in combined.columns if col != 'date'})
    return combined

def combine_hourly_to_latent_space(df, model, exclude_columns = ["date", "% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"], nhours=1, shifts=0):
    """ Combine the hourly data to a latent space using the autoencoder model
    """
    if isinstance(df,pd.DataFrame):
        flattened_data = flatten_n_hours_with_shifts(df, nhours=nhours, shifts=shifts,directly_to_file=False)
    else:
        flattened_data = pd.read_csv(df, parse_dates=['date'])
    print(flattened_data.shape)
    
    stationary_columns = exclude_columns
    # If more than one hour, then we must maintain lagged values for excluded columns
    if nhours > 1:
        stationary_columns = []
        for i in range(0, nhours):
            for col in exclude_columns:
                if col != 'date':
                    stationary_columns.append(f"{col}_{i}")
        stationary_columns = ["date"] + stationary_columns
    
    #Drop excluded columns
    excluded_data = flattened_data[stationary_columns]
    flattened_data = flattened_data.drop(columns=stationary_columns)
    
    # Scale the data
    scaler = StandardScaler()
    flattened_data = scaler.fit_transform(flattened_data)
    
    # Predict the latent space
    latent_space = model.predict(flattened_data)
    cols = [f"latent_{i}" for i in range(latent_space.shape[1])]
    latent_space = pd.DataFrame(latent_space, columns=cols)
    latent_space.reset_index(drop=True, inplace=True)
    excluded_data.reset_index(drop=True, inplace=True)
    print(latent_space.shape)
    
    # Add the date column back
    latent_space = pd.concat([excluded_data, latent_space], axis=1)
    return latent_space, scaler

def flatten_n_hours_with_shifts(df, nhours=1, shifts=0,directly_to_file="default"):
    """ Flatten the data to one row for N hours, and shift the data by 1 hour shifts times.
    if directly_to_file has a value, then save the flattened data to that file directly after flattening for each shift
    """
    
    if directly_to_file == "default":
        directly_to_file = f"miningdata/data_flattened_{nhours}hourly_" + ("multishift_" if shifts > 0 else "") + "train.csv"
    combined_hourly = flatten_n_hours_data(df, exclude_columns=STATIONARY_COLUMNS, nhours=nhours)
    # Write the results to a file
    if directly_to_file:
        combined_hourly.to_csv(directly_to_file, index=False,mode="w",header=True)
        # Then we can empty the dataframe
        combined_hourly = pd.DataFrame()
    print(f"Combined shape: {combined_hourly.shape}")
    # Do combinations with different shifts. So first, no shift, just group by 12 hours
    # Then shift by 1 hour, and group by 12 hours
    # Then shift by 2 hours, and group by 12 hours etc.
    for hour_shift in range(shifts):
        # One hour shift forward; we remove the first hour of data, and then group by 12 hours
        df = df.iloc[180:]
        shifted_combined = flatten_n_hours_data(df, exclude_columns=STATIONARY_COLUMNS, nhours=nhours)
        print(f"Shifted shape: {shifted_combined.shape}")
        combined_hourly = pd.concat([combined_hourly, shifted_combined], axis=0)
        print(f"Combined shape: {combined_hourly.shape}")
        assert combined_hourly.shape[1] == 19*nhours*180 + nhours*4 + 1, "Wrong number of columns"
        # Assert no duplicates
        assert not combined_hourly.columns.duplicated().any(), f"Found duplicates in columns: {combined_hourly.columns[combined_hourly.columns.duplicated()]}"
        if directly_to_file:
            combined_hourly.to_csv(directly_to_file, index=False,mode="a",header=False)
            # Then we can empty the dataframe
            combined_hourly = pd.DataFrame()
    return combined_hourly

def combine_heuristically(df):
    """ Combine hour of data to one row using the methods in methods
    """
    hourly_means = combine_to_hourly(df, method=lambda group : np.mean(group, axis=0))
    hourly_means.rename(columns=lambda col : f"{col}_mean" if col not in STATIONARY_COLUMNS else col, inplace=True)
    hourly_maxs = combine_to_hourly(df, method=lambda group : np.max(group, axis=0))
    hourly_maxs.rename(columns=lambda col : f"{col}_max" if col not in STATIONARY_COLUMNS else col, inplace=True)
    hourly_mins = combine_to_hourly(df, method=lambda group : np.min(group, axis=0))
    hourly_mins.rename(columns=lambda col : f"{col}_min" if col not in STATIONARY_COLUMNS else col, inplace=True)
    hourly_std = combine_to_hourly(df, method=lambda group : np.std(group, axis=0))
    hourly_std.rename(columns=lambda col : f"{col}_std" if col not in STATIONARY_COLUMNS else col, inplace=True)
    
    combined_hourly = pd.concat([hourly_means, hourly_maxs, hourly_mins, hourly_std], axis=1)
    combined_hourly = combined_hourly.loc[:,~combined_hourly.columns.duplicated()]
    # Put the stationary columns first
    combined_hourly = combined_hourly[STATIONARY_COLUMNS + [col for col in combined_hourly.columns if col not in STATIONARY_COLUMNS]]
    
    return combined_hourly

def combine_to_latent_space(df, nhours=1, shifts=0):
    model = tf.keras.models.load_model("models/encoder_1hourly_32.h5")
    combined_hourly,scaler = combine_hourly_to_latent_space(df, model,nhours=nhours, exclude_columns=STATIONARY_COLUMNS, shifts=shifts)
    return combined_hourly

def print_df_info(df):
    print(f"Shape of df: {df.shape}")
    print(f"Columns of df: {df.columns}")
    print(f"Data types of df: {df.dtypes}")
    print(f"Describe of df: {df.describe()}")
    print(f"Head of df: {df.head()}")
    print(f"Tail of df: {df.tail()}")
    return

def combine_heuristically():
    df, test_df, validation_df = train_test_validation_split_sequence("miningdata/data.csv",keep_stagnant=True)
    
    train_combined_heuristically = combine_heuristically(df)
    test_combined_heuristically = combine_heuristically(test_df)
    validation_combined_heuristically = combine_heuristically(validation_df)
    
    # Save the dataframes to csv files
    train_combined_heuristically.to_csv(f"miningdata/1hourly_heuristic_train.csv", index=False)
    test_combined_heuristically.to_csv(f"miningdata/1hourly_heuristic_test.csv", index=False)
    validation_combined_heuristically.to_csv(f"miningdata/1hourly_heuristic_validation.csv", index=False)
    
def flatten_nhours():
    df, test_df, validation_df = train_test_validation_split_sequence("miningdata/data.csv",keep_stagnant=True)

    nhours = 12
    shifts = nhours -1
    
    train_flattened = flatten_n_hours_with_shifts(df, nhours=nhours, shifts=shifts, directly_to_file=f"miningdata/{nhours}hourly_flattened_multishift_train.csv")
    test_flattened = flatten_n_hours_with_shifts(test_df, nhours=nhours, shifts=shifts, directly_to_file=f"miningdata/{nhours}hourly_flattened_multishift_test.csv")
    validation_flattened = flatten_n_hours_with_shifts(validation_df, nhours=nhours, shifts=shifts, directly_to_file=f"miningdata/{nhours}hourly_flattened_multishift_validation.csv")


if __name__ == "__main__":
    #df, test_df, validation_df = train_test_validation_split_sequence("miningdata/data.csv",keep_stagnant=True)
    #df = pd.read_csv("miningdata/1hourly_flattened_multishift_train.csv", parse_dates=['date'])
    #test_df = pd.read_csv("miningdata/1hourly_flattened_multishift_test.csv", parse_dates=['date'])
    #validation_df = pd.read_csv("miningdata/1hourly_flattened_multishift_validation.csv", parse_dates=['date'])

    
    df_combined_latent_space = combine_to_latent_space("miningdata/1hourly_flattened_multishift_train.csv", nhours=1, shifts=0)
    df_test_combined_latent_space = combine_to_latent_space("miningdata/1hourly_flattened_multishift_test.csv", nhours=1, shifts=0)
    df_validation_combined_latent_space = combine_to_latent_space("miningdata/1hourly_flattened_multishift_validation.csv", nhours=1, shifts=0)
    
    df_combined_latent_space.to_csv("miningdata/1hourly_latent32_train.csv", index=False)
    df_test_combined_latent_space.to_csv("miningdata/1hourly_latent32_test.csv", index=False)
    df_validation_combined_latent_space.to_csv("miningdata/1hourly_latent32_validation.csv", index=False)
    
    
    #assert train_flattened.shape[1] == 19*nhours*180 + nhours*4 + 1, f"Expected {19*nhours*180 + nhours*4 + 1} columns, got {train_flattened.shape[1]}"
    
    #combined_hourly = combine_to_latent_space("miningdata/data_flattened_12hourly_multishift_train.csv", nhours=nhours, shifts=0)
    
    #combined_hourly = flatten_n_hours_with_shifts(df, nhours=nhours, shifts=nhours-1)
    
    #combined_hourly.to_csv('miningdata/data_combined_12hourly_multishift_latent256_test.csv', index=False)