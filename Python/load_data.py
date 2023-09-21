""" Different ways to load the data from 'miningdata/data.csv'
date,% Iron Feed,% Silica Feed,Starch Flow,Amina Flow,Ore Pulp Flow,Ore Pulp pH,Ore Pulp Density,Flotation Column 01 Air Flow,...
2017-03-10 01:00:00,"55,2","16,98","3019,53","557,434","395,713","10,0664","1,74","249,214","253,235","250,576","295,096" ...
"""

import pandas as pd
import os

def load_to_dataframe(path, remove_first_days=False):
    """ Load the data to a pandas dataframe
    Convert floats with a comma to floats with a dot
    """
    df = pd.read_csv(path, parse_dates=['date'], decimal=',')
    if remove_first_days:
        df = df[df['date'] > '2017-04-11']
        # reset index
        df = df.reset_index(drop=True)
    return df

def separate_test_set(path, sz_percent=0.12):
    """ Take the last sz_percent of the data as test set.
    Copy the original data to a new file, and remove the test set from the original data.
    """
    df = pd.read_csv(path, parse_dates=['date'], decimal=',')
    # Get the index of the last sz_percent of the data
    idx = int(len(df)*(1-sz_percent))
    # Copy the last sz_percent of the data to a new dataframe
    df_test = df.iloc[idx:, :]
    # Remove the last sz_percent of the data from the original dataframe
    df = df.iloc[:idx, :]
    # Save the dataframes to csv files
    df.to_csv(path[:-4] + "_train.csv", index=False)
    df_test.to_csv(path[:-4] + "_test.csv", index=False)
    
    
def remove_sus_period(df):
    """ Remove data from 2017-05-13 to 2017-06-15
    """
    df = df[df['date'] < '2017-05-13' or df['date'] > '2017-06-15']

if __name__ == "__main__":
    separate_test_set("miningdata/data.csv", sz_percent=0.12)