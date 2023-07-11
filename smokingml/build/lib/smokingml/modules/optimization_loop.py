import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def optimization_loop(
    model: any,
    trainloader: DataLoader,
    devloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    device: str
):
    train_loss = []
    dev_loss = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        # Train Loop
        model.train()
        train_loss_sum = 0
        for Xtr,ytr in trainloader:
            Xtr,ytr = Xtr.to(device),ytr.to(device)

            # Forward pass
            logits = model(Xtr)
            loss = criterion(logits, ytr)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
        
        train_loss.append(train_loss_sum / len(trainloader))

        # Dev Loop
        model.eval()
        dev_loss_sum = 0
        for Xdev,ydev in devloader:
            Xdev,ydev = Xdev.to(device),ydev.to(device)

            # Forward pass
            logits = model(Xdev)
            dev_loss_sum += criterion(logits, ydev).item()

        dev_loss.append(dev_loss_sum / len(devloader))

        pbar.set_description(f'Epoch {epoch}: Train Loss: {train_loss[-1]:.5}: Dev Loss: {dev_loss[-1]:.5}')

        # Plot loss
        plt.plot(train_loss)
        plt.plot(dev_loss)
        plt.savefig('running_loss.jpg')
        plt.close()