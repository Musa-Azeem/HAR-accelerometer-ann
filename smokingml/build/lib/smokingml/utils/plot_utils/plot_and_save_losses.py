import matplotlib.pyplot as plt
import torch

def plot_and_save_losses(
    train_losses: list[float], 
    test_losses: list[float], 
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

    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.plot(torch.tensor(train_losses), label='Train Loss')
    plt.plot(torch.tensor(test_losses), label='Test Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epochs')
    plt.legend(loc='lower left')
    plt.title(f'Train and Test Loss over {n_epochs} Epochs')
    plt.savefig(filename, dpi=400)
    plt.close()