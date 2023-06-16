from pathlib import Path
import torch
import pandas as pd
import json
from typing import Tuple

def load_sessions_by_idxs(
    data_dir: str, 
    idxs: list,
    dm_factor: int = 5,
    win_len: int = 101
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir)
    
    X = torch.Tensor()
    y = torch.Tensor()

    for i in idxs:
        print(i)
        fp = Path(data_dir / str(i))
        labels_fp = Path(fp / f'{i}_data.json')
        raw_fp = Path(fp / 'raw_data.csv')

        if not (fp.is_dir() and labels_fp.is_file() and raw_fp.is_file()):
            print(f"Error: data missing from {str(fp)} directory - skipping")
            continue

        # Load labels
        with labels_fp.open('r') as f:
            try:
                json_labels = json.load(f)
                if not json_labels.get('puffs'):
                    raise ValueError('Json document missing "puffs" field')
            except ValueError as e:
                print(f'Error: {str(labels_fp)} - {e} - skipping')
                continue

        # Load raw and decimate - save length
        df = pd.read_csv(raw_fp, header=None, usecols=[2,3,4], names=['x','y','z'])[::dm_factor]
        len_df = len(df)
        
        # Get labels from json
        start = json_labels['start'] // dm_factor
        end = json_labels['end'] // dm_factor
        l = torch.zeros(len_df - (win_len - 1))
        for j in range(start, end):
            for puff in json_labels['puffs']:
                # If the midpoint of window j is within a puff, label the window as a puff
                startp = puff['start']//dm_factor
                endp = puff['end']//dm_factor
                if j+win_len//2 >= startp and j+win_len//2 <= endp:
                    l[j] = 1
        l = l.reshape([-1,1])
        y = torch.cat([y,l], axis=0)

        # Window Data
        raw = torch.from_numpy(df.to_numpy())
        x = raw[:,0].unsqueeze(1)
        y = raw[:,1].unsqueeze(1)
        z = raw[:,2].unsqueeze(1)

        w = win_len-1

        xs = [x[:-w]]
        ys = [y[:-w]]
        zs = [z[:-w]]

        for j in range(1,w):
            xs.append(x[j:j-w])
            ys.append(y[j:j-w])
            zs.append(z[j:j-w])

        xs.append(x[w:])
        ys.append(y[w:])
        zs.append(z[w:])

        xs = torch.cat(xs,axis=1).float()
        ys = torch.cat(ys,axis=1).float()
        zs = torch.cat(zs,axis=1).float()

        windowed = torch.cat([xs,ys,zs],axis=1)
        X = torch.cat([X, windowed], axis=0)

    return X,y