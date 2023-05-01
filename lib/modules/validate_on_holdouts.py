import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from utils import plot_and_save_holdout
from utils import prepare_labels_for_plot
from utils import plot_and_save_cm
import pandas as pd
import numpy as np

def validate_on_holdouts(
    model: torch.nn,
    holdout_dir: str,
    date: str,
    criterion: function,
    batch_size: int,
    win_size: int,
    device: str
):

    os.system(f'mkdir -p results/{date}/evaluation/holdouts')
    model.eval()

    for file in os.listdir(holdout_dir):
        index = file.split('-')[0]
        os.system(f'mkdir results/{date}/evaluation/holdouts/{index}')
        cur_dir = f'results/{date}/evaluation/holdouts/{index}'

        # Read in holdout session dataset
        dataset = torch.load(f'{holdout_dir}/{file}')
        length = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        y_true = dataset[:][1]

        # Read in holdout session dataframe
        df = pd.read_csv(f'pipeline/1_xyz/{index}.csv')
        df['y_true'] = prepare_labels_for_plot(y_true, win_size)

        # Keep track of metrics
        preds = []
        n_correct = 0
        loss = 0

        # Test loop
        for X, y in dataloader:
            X = X.to(device) 
            y = y.to(device)

            logits = model(X)
            pred = torch.round(nn.Sigmoid()(logits))

            n_correct += sum(y == pred)
            preds += pred.flatten().tolist()
            loss += criterion(logits, y).item()

        y_pred = np.array(preds).reshape(-1,1)
        accuracy = (n_correct / length).item()
        loss = loss/len(dataloader)
        print(f'Accuracy: {100*accuracy:.4}%')
        print(f'Loss: {loss:.4}')

        df['y_pred'] = prepare_labels_for_plot(y_pred, win_size)

        # Save Predictions
        np.save(f'{cur_dir}/y_pred.npy', y_pred)
        np.save(f'{cur_dir}/y_true.npy', y_true)

        # Save metrics
        pd.DataFrame({
            'Accuracy': [accuracy], 
            'Loss': [loss]
        }).to_csv(f'{cur_dir}/holdout-{index}-metrics.csv')


        # Save Figures
        plot_and_save_holdout(df, index, f'{cur_dir}/holdout-{index}.jpg')
        plot_and_save_cm(y_true, y_pred, f'{cur_dir}/holdout-{index}-cm.jpg')