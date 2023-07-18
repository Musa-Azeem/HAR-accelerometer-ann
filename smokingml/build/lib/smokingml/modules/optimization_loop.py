import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from . import inner_train_loop, inner_evaluate_loop
from ..utils import plot_and_save_losses, print_on_start_and_end

@print_on_start_and_end
def optimization_loop(
    model: nn.Module,
    trainloader: DataLoader,
    devloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: str,
    outdir: Path = None,
):
    if outdir:
        model_outdir = outdir / 'model'
        model_outdir.mkdir(parents=True)

    train_loss = []
    dev_loss = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        # Train Loop
        train_lossi = inner_train_loop(model, trainloader, criterion, optimizer, device)
        train_loss.append(sum(train_lossi) / len(trainloader))

        # Dev Loop
        y_true, y_pred, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
        dev_loss.append(sum(dev_lossi) / len(devloader))

        pbar.set_description(f'Epoch {epoch}: Train Loss: {train_loss[-1]:.5}: Dev Loss: {dev_loss[-1]:.5}')

        # Plot loss
        plt.plot(train_loss)
        plt.plot(dev_loss)
        plt.savefig('running_loss.jpg')

        if outdir:
            torch.save(model.state_dict(), model_outdir / f'{epoch}.pt')
            plot_and_save_losses(train_loss, dev_loss, epochs, str(outdir / 'loss.jpg'))
        plt.close()