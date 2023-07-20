import pandas as pd
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm
import json

def produce_nursingv1_dataset_from_raw(
    data_dir: str, 
    labels_dir: str, 
    nursingv1_outdir: str,
    win_size: int = 101,
    dm_factor: int = 5
):
    data_dir = Path(data_dir)
    labels_dir = Path(labels_dir)
    nursingv1_outdir = Path(nursingv1_outdir)

    nursingv1_outdir.mkdir()
    # ---------------------- Read Labels -----------------------------
    # Read json labels and get list of which sessions are labelled

    # labels for each labelled session
    json_labels = {}

    # Go through all json labels
    for file in labels_dir.iterdir():

        # Get session index from label filename
        session_idx = int(file.name.split('_')[0])

        # Read json and save it if session is labelled
        with file.open() as f:
            doc = json.load(f)

            # If the "puffs" key exists in the json document, 
            # the session has been labelled
            if doc.get('puffs'):
                json_labels[session_idx] = doc
    
    # ----------------------- Read Data ------------------------------

    # Go through all sessions that labels were found for
    pbar = tqdm(json_labels.items())
    for session_idx, y_json in pbar:

        ## Get X from csv
        # Read x,y,z data for this session and decimate from 100 Hz to 
        # 100/dm_factor Hz
        data_df = pd.read_csv(
            data_dir / f'{session_idx}' / 'raw_data.csv', 
            header=None,
            usecols=[2,3,4]
        )[::dm_factor]

        # Convert data to torch tensor, with x,y,z as rows and each 
        # datapoint as columns
        X = torch.tensor(data_df.values)  # Shape: len(session) by 3

        # Pad session so that there will be len(session) windows after 
        # windowing
        X = nn.functional.pad(X, (0,0,win_size//2,win_size//2))

        ## Get y from json labels
        y = torch.zeros(len(X) - (win_size - 1), 1)
        for puff in y_json['puffs']:

            # Get start and stop of puff, in same frequency as data
            puff_start = puff['start'] // dm_factor
            puff_end = puff['end'] // dm_factor

            # All windows whose center is within puff get y of 1
            # If index of this window is within range(start, end), then its 
            # center point is within the original puff (before padding)

            puff_start_idx = max(puff_start, 0)
            puff_end_idx = max(puff_end, 0)
            y[puff_start_idx:puff_end_idx] = 1
        
        ## Save X and y in dataset
        session_outdir = nursingv1_outdir / f'{session_idx}'
        Path(session_outdir).mkdir()
        
        # Save X and y
        torch.save(X, session_outdir / 'X.pt')
        torch.save(y, session_outdir / 'y.pt')