import torch
from torch import nn
from torch.utils.data import DataLoader

def inner_evaluate_loop(
    model: nn.Module,
    devloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

    y_preds = []
    y_true = []
    dev_lossi = []

    model.eval()
    for X,y in devloader:
        y_true.append(y)
        X,y = X.to(device), y.to(device)
        logits = model(X)
        dev_lossi.append(criterion(logits, y).item())
        pred = torch.round(nn.Sigmoid()(logits)).detach().to('cpu')
        y_preds.append(pred)

    return (torch.cat(y_true), torch.cat(y_preds), dev_lossi)
