import torch
from torch.utils.data import DataLoader
from torch import nn

def predict(model: nn.Module, loader: DataLoader, device: str) -> torch.tensor:
    # loader should have X and y tensors
    y_preds = []
    model.eval()
    for X,_ in loader:
        X = X.to(device)

        logits = model(X)
        pred = nn.Sigmoid()(logits)
        y_preds.append(torch.round(pred.detach().to('cpu')))

    return torch.cat(y_preds)