from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from .. import WINSIZE
from ..utils import window_session, window_session_for_conv, load_labels_from_fs, load_session_from_fs

def load_one_windowed_session(dir: Path, session_id: int, for_conv: bool = False) -> TensorDataset:
    # Get one session from dataset, window it, and turn it into a TensorDataset with its labels
    session = load_session_from_fs(dir, session_id)
    y = load_labels_from_fs(dir, session_id)

    # Window Session
    if for_conv:
        X = window_session_for_conv(session)
    else:
        X = window_session(session)

    # Return X and y as TensorDataset
    return TensorDataset(X, y)