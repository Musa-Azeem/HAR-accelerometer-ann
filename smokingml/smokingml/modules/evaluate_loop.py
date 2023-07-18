import torch
from torch.utils.data import DataLoader
from torch import nn
from . import inner_evaluate_loop
from ..utils import print_on_start_and_end

@print_on_start_and_end
def evaluate_loop(
    model: nn.Module, 
    criterion: nn.Module, 
    devloader: DataLoader, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    y_true, y_pred, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
    dev_loss = sum(dev_lossi) / len(devloader)

    return y_true, y_pred, dev_loss