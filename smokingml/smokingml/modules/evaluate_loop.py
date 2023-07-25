import torch
from torch.utils.data import DataLoader
from torch import nn
from . import inner_evaluate_loop
from ..utils import print_on_start_and_end, plot_and_save_cm
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

@print_on_start_and_end
def evaluate_loop(
    model: nn.Module, 
    criterion: nn.Module, 
    devloader: DataLoader, 
    device: str,
    metrics: bool = False,
    outdir: Path = None,

) -> any:

    y_true, y_pred, dev_lossi = inner_evaluate_loop(model, devloader, criterion, device)
    dev_loss = sum(dev_lossi) / len(devloader)

    if outdir:
        plot_and_save_cm(y_true, y_pred, outdir)

    if metrics:
        prec, recall, f1score, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division='warn', average='macro'
        )

        return y_true, y_pred, dev_loss, prec, recall, f1score
    
    return y_true, y_pred, dev_loss