from pathlib import Path
from torch.utils.data import TensorDataset
from ..utils import window_session, load_labels_from_fs, load_session_from_fs

def load_one_windowed_session(dir: Path, session_id: int, winsize: int) -> TensorDataset:
    # Get one session from dataset, window it, and turn it into a TensorDataset with its labels
    session = load_session_from_fs(dir, session_id)
    y = load_labels_from_fs(dir, session_id)

    # Window Session
    X = window_session(session, winsize)

    # Return X and y as TensorDataset
    return TensorDataset(X, y)