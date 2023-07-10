from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from .. import WINSIZE
from ..utils import window_session, load_labels_from_fs, load_session_from_fs

def load_one_windowed_session(dir: Path, session_id: int) -> TensorDataset:
    # Get one session from dataset, window it, and turn it into a TensorDataset with its labels
    # session = torch.load(dir / f'{session_id}' / 'X.pt')
    # labels = torch.load(dir / f'{session_id}' / 'y.pt')
    session = load_session_from_fs(dir, session_id)
    labels = load_labels_from_fs(dir, session_id)

    # Window Session
    X = window_session(session)

    # Return X and y as TensorDataset
    return TensorDataset(X, labels)