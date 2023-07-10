from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from ..utils import get_all_session_ids
import load_one_windowed_session

def load_windowed_sessions(dir: Path, session_ids: list[int] = None, shuffle: bool = False):
    # return concatonated tensor of windowed sessions in list
    # if no list is provided, all the sessions are returned (might cause memory issues)

    if not session_ids:
        session_ids = get_all_session_ids(dir)
    
    sessions = []
    all_labels = []
    for session_id in session_ids:
        dataset = load_one_windowed_session(dir, session_id)
        sessions.append(dataset.tensors[0])
        all_labels.append(dataset.tensors[1])
    
    return TensorDataset(torch.cat(sessions), torch.cat(all_labels))