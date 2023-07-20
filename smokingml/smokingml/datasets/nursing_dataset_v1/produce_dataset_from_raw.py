import pandas as pd
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm
import json

def produce_nursingv1_dataset_from_raw(
    data_dir: Path, 
    labels_dir: Path, 
    nursingv1_outdir: Path,
    win_size: int = 101,
    dm_factor: int = 5
):
    # data_dir = Path(working_dir / 'nursing_raw')
    # labels_dir = Path(working_dir / 'nursing_labels_musa')
    # nursingv1_outdir = Path(working_dir / 'nursingv1_dataset')
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
        X = torch.tensor(data_df.values).T  # Shape: 3 by len(session)

        # Pad session so that there will be len(session) windows after 
        # windowing
        X = nn.functional.pad(X, (win_size//2, win_size//2), 'constant', 0)

        ## Get y from json labels
        y = torch.zeros(len(X[0]) - WINSIZE + 1).reshape([-1,1])   
        for puff in y_json['puffs']:

            # Get start and stop of puff, in same frequency as data
            puff_start = puff['start'] // DM_FACTOR
            puff_end = puff['end'] // DM_FACTOR

            # All windows whose center is within puff get y of 1
            # All windows `WINSIZE//2` before start and end have a center within puff
            puff_start_idx = max(puff_start - WINSIZE//2, 0)
            puff_end_idx = max(puff_end - WINSIZE//2, 0)
            y[puff_start_idx:puff_end_idx] = 1
        
        
        ## Save X and y in dataset
        session_outdir = nursingv1_outdir / f'{session_idx}'
        Path(session_outdir).mkdir()
        
        # Save X and y
        torch.save(X, session_outdir / 'X.pt')
        torch.save(y, session_outdir / 'y.pt')
        
        # Save size of session
        torch.save(X.shape, session_outdir / 'Xshape.pt')

        # Save X of each session in files of size 5000 for less file overhead
        # for i in range(0, len(X[0], 5000)):
        #     end_idx = min(len(X[0:], )
        #     torch.save(
        #         X[]
        #     )
        # torch.save(
        #     TensorDataset(X,y),
        #     f'{len(X[0])}.pt'       # use length of session as filename
        # )