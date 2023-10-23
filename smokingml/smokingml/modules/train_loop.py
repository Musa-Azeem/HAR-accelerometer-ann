import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from . import inner_train_loop
from ..utils import plot_and_save_loss, print_on_start_and_end

@print_on_start_and_end
def train_loop(
    model: nn.Module,
    trainloader: DataLoader,
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

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        # Train Loop
        train_lossi = inner_train_loop(model, trainloader, criterion, optimizer, device)
        train_loss.append(sum(train_lossi) / len(trainloader))

        pbar.set_description(f'Epoch {epoch}: Train Loss: {train_loss[-1]:.5}')

        # Plot loss
        plt.plot(train_loss)
        plt.savefig('running_loss.jpg')

        if outdir:
            torch.save(model.state_dict(), model_outdir / f'{epoch}.pt')
            plot_and_save_loss(train_loss, epochs, str(outdir / 'loss.jpg'))
        plt.close()