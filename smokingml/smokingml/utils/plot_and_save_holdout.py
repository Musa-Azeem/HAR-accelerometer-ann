import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_and_save_holdout(
    df: pd.DataFrame, 
    raw_dir: str,
    index: int, 
    filename: str
) -> None:
    """
        Plots and saves true labels and predictions on the xyz signal of a 
        holdout set. Only plots the smoking session within the holdout session.

    Args:
        df (pd.DataFrame): Dataframe of holdout session to plot
        raw_dir (str): dir containing raw data and json labels
        index (int): index of holdout session
        date (str): timestamp for directories
    """
    # Get start and stop of smoking session from json annotations
    start = 0
    stop = 0
    with open(f'{raw_dir}/{index}/{index}_data.json', 'r') as f:
        annot = json.load(f)
        start = annot['start']
        stop = annot['end']

    # plot signals
    df_plot = df.iloc[start:stop].reset_index(drop=True)
    df_plot.columns = ['x acc', 'y acc', 'z acc', 'true label', 'pred label']
    fig, ax = plt.subplots(1, figsize=(16,9))
    ax.plot(df_plot, label=df_plot.columns)
    ax.legend()
    plt.savefig(filename, dpi=400)
    plt.close()