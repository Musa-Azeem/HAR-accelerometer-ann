import torch
from pathlib import Path
import numpy as np
from .. import WINSIZE
from ..utils import load_labels_from_fs, load_session_from_fs

def load_one_session(dir: Path, session_id: int)  -> tuple[torch.Tensor, torch.Tensor]:
    # Get one unwindowed session (padded) from fs and its labels

    # Load session and labels
    X = load_session_from_fs(dir, session_id)
    y = load_labels_from_fs(dir, session_id)

    return (X,y)