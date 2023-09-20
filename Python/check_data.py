""" Visualize the timeseries data in 'miningdata/data.csv' 
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Disable sns deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from load_data import load_to_dataframe
from scipy import stats
import pandas as pd

def find_outlier_sequences(df, window=180,outlier_mask=None):
    """ For each window of time calculate each feature's
    - standard deviation (std(values)) (float)
    - mean (mean(values)) (float)
    
    We then have a matrix of shape (len(df)/window, 2*len(df.columns))
    where each row is a window of time and the columns contain the standard deviations and means of each feature
    
    Then for each feature
    - Calculate the global standard deviation
    - Calculate the global mean
    
    Then for each windowed feature (w_i, x_i)):
    - Check if the standard deviation or mean is greater than the global standard deviation or global mean of the feature
    """
    df = df.drop(columns=['date'])
    # Create a matrix of shape (len(df)/window, 2*len(df.columns))
    # where each row is a window of time and the columns contain the standard deviations and means of each feature
    windowed_features = np.zeros((len(df)//window, 2*len(df.columns)))
    for i in range(len(df)//window):
        # Get the window of time
        window_df = df.iloc[i*window:(i+1)*window]
        # Get the standard deviations and means of each feature
        stds = window_df.std()
        means = window_df.mean()
        # Add the standard deviations and means to the windowed_features matrix
        windowed_features[i,:len(df.columns)] = stds
        windowed_features[i,len(df.columns):] = means
    # Calculate the global standard deviation and mean for each feature
    global_stds = df.std()
    global_stds_std = global_stds.std()
    global_stds_mean = global_stds.mean()
    global_means = df.mean()
    global_means_std = global_means.std()
    global_means_mean = global_means.mean()
    # Check if the standard deviation or mean of a window is more than 2 standard deviations away from the global standard deviation or mean
    # If so, then we consider the window an outlier
    window_stats_std = []
    window_stats_mean = []
    for i in range(len(windowed_features)):
        mean_z_scores = []
        std_z_scores = []
        for j in range(len(df.columns)):
            std = windowed_features[i,j]
            mean = windowed_features[i,j+len(df.columns)]
            z_mean = (mean - global_means_mean)/global_means_std
            z_std = (std - global_stds_mean)/global_stds_std
            mean_z_scores.append(z_mean)
            std_z_scores.append(z_std)
        # Save the z scores for each window and feature
        window_stats_mean.append(mean_z_scores)
        window_stats_std.append(std_z_scores)
    # Convert to numpy arrays
    window_stats_mean = np.array(window_stats_mean)
    window_stats_std = np.array(window_stats_std)
    # Check each row from window_stats_mean and window_stats_std and classify in to outliers using Mahalanobis distance
    outlier_windows = []
    for window_idx in range(len(window_stats_mean)):
        # Get the z-scores for each feature
        z_scores_mean = window_stats_mean[window_idx]
        z_scores_std = window_stats_std[window_idx]
        # Calculate the Mahalanobis distance
        mean_mahalanobis_distance = np.sqrt(np.sum(z_scores_mean**2))
        std_mahalanobis_distance = np.sqrt(np.sum(z_scores_std**2))
        # If the Mahalanobis distance is greater than 3, then we consider the window an outlier
        if mean_mahalanobis_distance > 6 or std_mahalanobis_distance > 6:
            outlier_windows.append(window_idx)
    # Print the results
    print(f"Number of outlier windows: {len(outlier_windows)}")
    print(f"Number of windows: {len(windowed_features)}")
    print(f"Percentage of outlier windows: {len(outlier_windows)/len(windowed_features)*100:.2f}%")
    plt.show()
    return outlier_windows

def compute_outlier_mask(df, outlier_z_thresh = 3, return_abs_z_scores=False):
    """ Return a dataframe with True values at indices of outliers, and False else where
    """
    has_date = 'date' in df.columns
    if has_date:
        df = df.drop(columns=['date'])
    # Compute the z-scores for each feature
    z_scores = stats.zscore(df)
    # Compute the absolute z-scores
    abs_z_scores = np.abs(z_scores)
    # Create a mask, same ssize as df, with True values at indices of outliers, and False else where
    outlier_mask = abs_z_scores > outlier_z_thresh
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

def flatten_n_hours_data(df, exclude_columns = ["% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"]):
    """ Flatten all observations from 1 hours to one row, excluding the columns measured only once per hour
    """
    exclude_columns = exclude_columns + ['date']
    # The dataframe will have 180 columns for each non-excluded column + the excluded columns
    columns = []
    for i in range(180):
        for col in df.columns:
            if col not in exclude_columns:
                columns.append(f"{col}_{i}")
    columns = columns + exclude_columns
    # Create the dataframe
    combined = pd.DataFrame(columns=columns)
    print(combined.shape)
    print(combined.columns)

    # For each hour (180 consecutive rows)
    for i in range(len(df)//180):
        # Get the 180 rows from non-excluded columns
        rows = df.iloc[i*180:(i+1)*180]
        excluded_values = df.iloc[i*180][exclude_columns]
        rows = rows.drop(columns=exclude_columns)
        # Flatten the rows to one row
        flattened_df = pd.DataFrame(rows.values.flatten()).T
        # Add the excluded values to the row
        flattened_df = pd.concat([flattened_df, excluded_values], columns=exclude_columns, axis=1)
        # Add the row to the dataframe
        combined = pd.concat([combined,flattened_df], axis=0)
    return combined



if __name__ == "__main__":
    # ["% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"]
    df = load_to_dataframe(remove_first_days=True)
    no_change_cols = ['% Silica Concentrate', '% Iron Concentrate', '% Iron Feed', '% Silica Feed']
    combined_hourly_means = flatten_n_hours_data(df, exclude_columns=no_change_cols)
    #combined_hourly_means.rename(columns=lambda col : f"{col}_mean" if col not in no_change_cols else col, inplace=True)
    #combined_hourly_maxs = combine_to_hourly(df,nhours=1, method=lambda group : np.max(group, axis=0), exclude_columns=[])
    #combined_hourly_maxs.rename(columns=lambda col : f"{col}_max" if col not in no_change_cols else col, inplace=True)
    #combined_hourly_mins = combine_to_hourly(df,nhours=1, method=lambda group : np.min(group, axis=0), exclude_columns=[])
    #combined_hourly_mins.rename(columns=lambda col : f"{col}_min" if col not in no_change_cols else col, inplace=True)
    #combined_hourly_std = combine_to_hourly(df,nhours=1, method=lambda group : np.std(group, axis=0), exclude_columns=[])
    #combined_hourly_std.rename(columns=lambda col : f"{col}_std" if col not in no_change_cols else col, inplace=True)

    combined_hourly = pd.concat([combined_hourly_means], axis=1)
    # Remove duplicate columns, but keep 1
    combined_hourly = combined_hourly.loc[:,~combined_hourly.columns.duplicated()]
    # Round the values to 3 decimals
    combined_hourly = combined_hourly.round(3)
    print(combined_hourly.describe())
    print(combined_hourly.head())
    print(combined_hourly.tail())
    outlier_mask = compute_outlier_mask(combined_hourly, outlier_z_thresh=3)
    print(outlier_mask)
    print(outlier_mask.shape)
    print(outlier_mask.sum())
    combined_hourly.to_csv('miningdata/data_flattened_hourly.csv', index=False)