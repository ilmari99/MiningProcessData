""" Different ways to load the data from 'miningdata/data.csv'
date,% Iron Feed,% Silica Feed,Starch Flow,Amina Flow,Ore Pulp Flow,Ore Pulp pH,Ore Pulp Density,Flotation Column 01 Air Flow,...
2017-03-10 01:00:00,"55,2","16,98","3019,53","557,434","395,713","10,0664","1,74","249,214","253,235","250,576","295,096" ...
"""

import pandas as pd

def load_to_dataframe(remove_first_days=False):
    """ Load the data to a pandas dataframe
    Convert floats with a comma to floats with a dot
    """
    df = pd.read_csv('miningdata/data.csv', parse_dates=['date'], decimal=',')
    if remove_first_days:
        df = df[df['date'] > '2017-04-11']
        # reset index
        df = df.reset_index(drop=True)
    return df

def filter_dataframe(df):
    """ Remove outliers"""


if __name__ == "__main__":
    df = load_to_dataframe()
    print(df.head())
    print(df.dtypes)
    