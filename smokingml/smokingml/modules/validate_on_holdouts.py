import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from smokingml.utils import (
    plot_and_save_holdout,
    prepare_labels_for_plot,
    plot_and_save_cm,
    print_on_start_and_end
)

@print_on_start_and_end
def validate_on_holdouts(
    model: nn.Module,
    holdout_dir: str,
    df_dir: str,
    raw_dir: str,
    date: str,
    criterion: nn.Module,
    batch_size: int,
    win_size: int,
    device: str,
    project: str,
    dm_factor: int,
    cnn: bool=False
):
    """
        Validates model on holdout sets in dataset

    Args:
        model (nn.Module): trained model
        holdout_dir (str): directory containing holdout sets (in pytorch dataset archives)
        df_dir (str): directory containing holdout set as pandas dataframes (not windowed)
        raw_dir (str): directory containing raw data and json labels
        date (str): date of current session
        criterion (nn.Module): loss function for testing model
        batch_size (int): batch size for memory concerns
        win_size (int): window size of data examples
        device (str): device to run computations on
        project (str): directory to save results
        dm_factor (int): factor of dessimation
        cnn (bool): true if model uses convolution
    """

    os.system(f'mkdir -p {project}/results/holdouts')
    model.eval()

    accs = {}
    losses = {}

    for file in os.listdir(holdout_dir):
        index = file.split('-')[0]
        os.system(f'mkdir {project}/results/holdouts/{index}')
        cur_dir = f'{project}/results/holdouts/{index}'

        # Read in holdout session dataset
        dataset = torch.load(f'{holdout_dir}/{file}')
        length = len(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        y_true = dataset[:][1]

        # Read in holdout session dataframe
        df = pd.read_csv(f'{df_dir}/{index}.csv')
        df['y_true'] = prepare_labels_for_plot(y_true, win_size)

        # Keep track of metrics
        preds = []
        n_correct = 0
        loss = 0

        # Test loop
        for X, y in dataloader:
            X = X.to(device) 
            y = y.to(device)

            if cnn:
                X = X.reshape([-1, 3, win_size])

            logits = model(X)
            pred = torch.round(nn.Sigmoid()(logits))

            n_correct += sum(y == pred)
            preds += pred.flatten().tolist()
            loss += criterion(logits, y).item()

        y_pred = np.array(preds).reshape(-1,1)
        accs[int(index)] = (n_correct / length).item()
        losses[int(index)] = loss/len(dataloader)

        # print(f'Accuracy: {100*accuracy:.4}%')
        # print(f'Loss: {loss:.4}')

        df['y_pred'] = prepare_labels_for_plot(y_pred, win_size)

        # Save Predictions
        np.save(f'{cur_dir}/y_pred.npy', y_pred)
        np.save(f'{cur_dir}/y_true.npy', y_true)

        # Save metrics
        pd.DataFrame({
            'Accuracy': [accs[int(index)]], 
            'Loss': [losses[int(index)]]
        }).to_csv(f'{cur_dir}/holdout-{index}-metrics.csv')


        # Save Figures
        plot_and_save_holdout(df, raw_dir, index, f'{cur_dir}/holdout-{index}.jpg', dm_factor)
        plot_and_save_cm(y_true, y_pred, f'{cur_dir}/holdout-{index}-cm.jpg')

    print(tabulate(
        [
            ['Accuracy'] + [accs[index] for index in sorted(accs.keys())],
            ['Loss'] + [losses[index] for index in sorted(losses.keys())]
        ],
        headers=[index for index in sorted(accs.keys())],
        tablefmt='simple_grid'
    ))        