import torch
from pathlib import Path
import numpy as np
from .. import WINSIZE
from ..utils import load_labels_from_fs, load_session_from_fs

def load_one_session(dir: Path, session_id: int)  -> tuple[torch.Tensor, torch.Tensor]:
    # Get one unwindowed session from fs and its labels (labels are padded)

    # Load session and labels
    X = load_session_from_fs(dir, session_id)
    y = load_labels_from_fs(dir, session_id)

    # Pad labels with half of window size at beginning and end to match length of X
    y = np.pad(
        y.flatten(), 
        (WINSIZE//2, WINSIZE//2), 
        mode='constant',
        constant_values=0
    )

    return (X,y)