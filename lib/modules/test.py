import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from lib.utils import plot_and_save_cm
from typing import List


def test(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    y_true: List[int],
    device: str,
    criterion: nn.Module,
    batch_size: int,
    date: str,
    project: str
):
    """
        Tests the given model on the given dataset. Generates a confusion 
        matrix with the results.

    Args:
        model (nn.Module): model to test
        dataset (torch.utils.data.Dataset): pytorch test dataset
        y_true (list[int]): true y labels to compare results to
        device (str): gpu or cpu device
        criterion (nn.Module): loss function
        batch_size (int): batch size (for memory purposes)
        date (str): timestamp for directories
        project (str): project directory
    """

    os.system(f'mkdir -p {project}/results/test/')

    length = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    preds = []
    n_correct = 0
    loss = 0

    # Test loop
    for X_test, y_test in tqdm(dataloader):
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        logits = model(X_test)
        pred = torch.round(nn.Sigmoid()(logits))
        loss += criterion(logits, y_test).item()
        n_correct += sum(y_test == pred)
        preds += pred.flatten().tolist()

    accuracy = (n_correct / length).item()
    loss = loss / len(dataloader)
    print(f'Accuracy: {100*accuracy:.4}%')
    print(f'Loss: {loss:.5}')

    # Save Test metrics
    pd.DataFrame(
        {
            'Accuracy': [accuracy], 
            'Loss': [loss]
        }
    ).to_csv(f'{project}/results/test/test-metrics.csv')


    # Generate and save confusion matrix for test dataset
    y_pred = np.array(preds).reshape(-1,1)
    plot_and_save_cm(y_true, y_pred, f'{project}/results/test/test-cm.jpg')