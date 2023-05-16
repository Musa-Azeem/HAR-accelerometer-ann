import matplotlib.pyplot as plt
import torch
from typing import List

def plot_and_save_losses(
    train_losses: List[float], 
    test_losses: List[float], 
    n_epochs: int,
    filename: str
) -> None:
    """
        Plots a loss curve for given train and test losses.

    Args:
        train_losses (list[float]): train losses
        test_losses (list[float]): test losses
        n_epochs (int): number of epochs that generated these results
        filename (str): file name and path to save image as
    """

    fig, ax = plt.subplots(1)
    ax.plot(torch.tensor(train_losses), label='Train Loss')
    ax.plot(torch.tensor(test_losses), label='Test Loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel('Epochs')
    ax.legend()
    ax.set_title(f'Train and Test Loss over {n_epochs} Epochs')
    fig.set_size_inches(16, 9)
    plt.savefig(filename, dpi=400)
    plt.close()