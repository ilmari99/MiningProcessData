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

class OutlierDetector:
    def __init__(self, df, X,y):
        """ X contains a sequence of vectors (181 measurements in one hour)
        Each X_i is a 180x24 matrix
        X_i is considered an outlier if y_i is an outlier
        """
        self.df = df
        self.X = X
        self.y = y
        #self.similarity_matrix = self.calc_sequence_similarity_matrix()
        #print(f"Shape of similarity matrix: {self.similarity_matrix.shape}")
        # Save as excel file
        #pd.DataFrame(self.similarity_matrix).to_excel("similarity_matrix.xlsx")
        
    
    def calc_sequence_similarity_matrix(self):
        """ Calculate the similarity matrix for the sequences
        """
        self.similarity_matrix = np.zeros((len(self.X), len(self.X)))
        todo = len(self.X)**2
        print(f"Calculating similarity matrix for {todo} pairs of sequences")
        for i in range(len(self.X)):
            # Calculate the similarity between X_i and X_j
            for j in range(len(self.X)):
                if j % 100 == 0:
                    print(f"Progress: {i*len(self.X)+j}/{todo}")
                # Calculate the similarity between X_i and X_j
                similarity = self.calc_sequence_similarity(self.X[i], self.X[j])
                # Add the similarity to the similarity matrix
                self.similarity_matrix[i,j] = similarity
    
    def calc_sequence_similarity(self, X_i, X_j):
        """ Calculate the similarity between two matrices consisiting of rows using Kullback-Leibler divergence
        """
        
        # Calculate the similarity between X_i and X_j
        similarity = 0
        for row_i in X_i:
            for row_j in X_j:
                similarity += stats.entropy(row_i, row_j)
        return similarity
    
    def is_sequence_outlier(self, i):
        """ Return True if the matrix (sequence) is an outlier.
        """
        # For now we just do this by considering the z-score of the silica concentration
        # If the z-score is > 3, then we consider the sequence an outlier
        z_score = stats.zscore(self.y)
        return z_score[i] > 2
    

def find_outlier_sequences(df, window=180):
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
            # Check if the standard deviation or mean is more than 2 standard deviations away from the global standard deviation or mean
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
    # Check each row, and if the average Z-score is > 2, then we consider the window an outlier
    outlier_windows = []
    for i in range(len(window_stats_mean)):
        z_scores = window_stats_mean[i]
        mean_z_score = np.mean(z_scores)
        if max(z_scores) > 4 or mean_z_score > 0.75:
            outlier_windows.append(i)
    # Print the results
    print(f"Number of outlier windows: {len(outlier_windows)}")
    print(f"Number of windows: {len(windowed_features)}")
    print(f"Percentage of outlier windows: {len(outlier_windows)/len(windowed_features)*100:.2f}%")
    plt.show()
    return outlier_windows
        
        
            
        

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
            ax_.axvspan(outlier_idx*window_size, (outlier_idx+1)*window_size, facecolor='r', alpha=0.5)
    plt.show()

def convolve_series_confidence_interval(series, window_size, return_series=False):
    """ Convolve a time series with a window_size convolution kernel.
    Return the confidence interval of the convoluted series.
    """
    # Convolve the series with a window_size convolution kernel
    convolved_series = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    # Calculate the confidence interval of the convolved series
    convolved_series_std = np.std(convolved_series)
    convolved_series_confidence_interval = 1.96 * convolved_series_std / np.sqrt(len(convolved_series))
    if return_series:
        return convolved_series, convolved_series_confidence_interval
    return convolved_series_confidence_interval

def visualize_convolved_df(df, window_size, save=True):
    """ Plot the timeseries data
    """
    # Plot the timeseries for each column
    fig,ax = plt.subplots(5,5)
    print(f"Subplot size: {(5,5)}")
    for col_idx, col in enumerate(df.columns):
        if col == 'date':
            continue
        vals, CI = convolve_series_confidence_interval(df[col], window_size, return_series=True)
        print(f"{col} (CI={CI:.2f})")
        ax_ : plt.Axes = ax[col_idx//5, col_idx%5]
        ax_.plot(vals)
        ucb = vals+CI
        lcb = vals-CI
        ax_.fill_between(np.arange(len(vals)), lcb, ucb)
        ax_.set_title(f"{col} (CI={CI:.2f})")
    plt.show()
    return

def remove_dates(df, ret_good_indices=False):
    """ Remove time periods, where the '% Iron Feed' and '% Silica Feed' values stay the same
    """
    print(f"Shape of df before removing dates: {df.shape}")
    # Remove the periods of time, where the '% Iron Feed' and '% Silica Feed' values stay the same
    iron_diffs = df['% Iron Feed'].diff()
    silica_diffs = df['% Silica Feed'].diff()
    # Get the indices of the rows, where both '% Iron Feed' and '% Silica Feed' have 0 difference
    indices = np.where((iron_diffs != 0) & (silica_diffs != 0))[0]
    print(indices)
    # Remove the rows with the indices
    df = df.iloc[indices]
    print(f"Shape of df after removing dates: {df.shape}")
    if ret_good_indices:
        return df, indices
    return df

if __name__ == "__main__":
    df = load_to_dataframe(remove_first_days=True)
    #df = remove_dates(df)
    outlier_indxs = find_outlier_sequences(df, window = 1800)
    basic_check_data(df)
    #check_data_timeseries(df)
    #basic_visualize(df, histograms=True, scatterplots=False, save=True)
    visualize_timeseries(df, save=True, outlier_indices=outlier_indxs, window_size=1800)
    #visualize_convolved_df(df, 1000, save=True)
    
    