import torch
from torch.utils.data import DataLoader
from torch import nn
from ..utils import print_on_start_and_end

@print_on_start_and_end
def evaluate_loop(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: str) -> tuple[torch.tensor, torch.tensor]:
    y_preds = []
    y_true = []
    loss_total = 0

    model.eval()
    for X,y in loader:
        y_true.append(y)
        X,y = X.to(device), y.to(device)
        logits = model(X)
        loss_total += criterion(logits, y).item()
        pred = torch.round(nn.Sigmoid()(logits)).detach().to('cpu')
        y_preds.append(pred)


    return (torch.cat(y_true), torch.cat(y_preds), loss_total / len(loader))