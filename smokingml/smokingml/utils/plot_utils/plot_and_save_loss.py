import matplotlib.pyplot as plt
import torch

def plot_and_save_loss(
    losses: list[float], 
    n_epochs: int,
    filename: str
) -> None:
    """
        Plots a loss curve for given losses.

    Args:
        losses (list[float]): losses
        n_epochs (int): number of epochs that generated these results
        filename (str): file name and path to save image as
    """

    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(torch.tensor(losses), label='Train Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.legend(loc='lower left')
    plt.title(f'Loss over {n_epochs} Epochs')
    plt.savefig(filename, dpi=400)
    plt.close()