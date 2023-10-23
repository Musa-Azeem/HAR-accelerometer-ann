from torch.utils.data import DataLoader
from torch import nn
import torch

def inner_train_loop(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> list[float]:

    model.train()
    lossi = []
    for Xtr,ytr in trainloader:
        Xtr,ytr = Xtr.to(device),ytr.to(device)

        # Forward pass
        logits = model(Xtr)
        loss = criterion(logits, ytr)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossi.append(loss.item())
    
    return lossi