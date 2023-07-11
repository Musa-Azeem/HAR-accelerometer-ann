import torch
from torch.utils.data import DataLoader
from torch import nn

def evaluate_loop(model: nn.Module, loader: DataLoader, device: str) -> torch.tensor:
    y_preds = []

    model.eval()
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        logits = model(X)
        pred = torch.round(nn.Sigmoid()(logits)).detach().to('cpu')
        y_preds.append(pred)

    return torch.cat(y_preds)