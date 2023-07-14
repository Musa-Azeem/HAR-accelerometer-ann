import torch
from torch.utils.data import DataLoader
from torch import nn
from ..utils import print_on_start_and_end

@print_on_start_and_end
def evaluate_loop(model: nn.Module, loader: DataLoader, device: str) -> tuple[torch.tensor, torch.tensor]:
    y_preds = []
    y_true = []

    model.eval()
    for X,y in loader:
        y_true.append(y)
        X = X.to(device)
        logits = model(X)
        pred = torch.round(nn.Sigmoid()(logits)).detach().to('cpu')
        y_preds.append(pred)

    return (torch.cat(y_true), torch.cat(y_preds))