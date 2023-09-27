""" Visualize data from the synchronized mining process data.

"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import random
warnings.filterwarnings("ignore")

def plot_distributions(df,cols="all", show = False, exclude_cols=[]):
    """ Plot the values distributions for each column in df.
    """
    figs, axs = [], []
    if cols == "all":
        cols = df.columns
    for col in cols:
        if col in exclude_cols:
            continue
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        sns.distplot(df[col],ax=ax,label=col)
        ax.legend()
        ax.set_title(f"Distribution of '" + col + f"' in training data when compressing 12 hours to 128 latent variables")
        ax.grid(True)
        ax.set_facecolor((0.9, 0.9, 0.9))
        figs.append(fig)
        axs.append(ax)
    if show:
        plt.show()
    return figs, axs

def plot_distributions_grid(df, cols=4, show = False):
    """ Plot a grid with the distributions of each column in cols.
    If cols = 4, the select 8 at random.
    """
    if cols == 4:
        cols = random.sample(list(df.columns), 4)
        while any("latent" not in col for col in cols):
            cols = random.sample(list(df.columns), 4)
    fig, axs = plt.subplots(2,2,figsize=(20,10))
    fig.suptitle("Distribution of values in training data when compressing 12 hours to 64 latent variables")
    for i, col in enumerate(cols):
        sns.distplot(df[col],ax=axs[i//2][i%2],label=col)
        axs[i//2][i%2].legend()
        axs[i//2][i%2].set_title(f"Distribution of '" + col)
        axs[i//2][i%2].grid(True)
        axs[i//2][i%2].set_facecolor((0.9, 0.9, 0.9))
    if show:
        plt.show()
    return fig, axs

def plot_major_distribution_grids(df, cols, show = False):
    """ Plot 2x2 grids of each heuristic for each major.
    """
    majors = [col.split("_")[0] for col in cols]
    cols = [major + "_" + col for major in majors for col in ["mean", "max", "min", "std"]]
    figs, axs = [], []
    for major in majors:
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        for i, col in enumerate(["mean", "max", "min", "std"]):
            sns.distplot(df[major + "_" + col], ax=ax[i//2][i%2], label=col)
            ax[i//2][i%2].legend()
            ax[i//2][i%2].set_title(f"Distribution of '" + major + "_" + col + f"' in training data")
            ax[i//2][i%2].grid(True)
            ax[i//2][i%2].set_facecolor((0.9, 0.9, 0.9))
        figs.append(fig)
        axs.append(ax)
    if show:
        plt.show()
    return figs, axs

if __name__ == "__main__":
    df = pd.read_csv("miningdata/12hourly_latent64_train.csv", parse_dates=['date'])
    # Take 5 random columns, and take the majors (split at _)
    #cols = random.sample(list(df.columns), 4)
    #while any("_" not in col for col in cols):
    #    cols = random.sample(list(df.columns), 5)
    plot_distributions_grid(df, cols=4, show=True)
    

        