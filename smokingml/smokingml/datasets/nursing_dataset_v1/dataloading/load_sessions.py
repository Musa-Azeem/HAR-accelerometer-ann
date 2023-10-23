from pathlib import Path
import torch
from . import load_one_session

def load_sessions(dir: Path, session_ids: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # get unwindowed sessions (padded) listed in param and their labels

    sessions = []
    for session_id in session_ids:
        sessions.append(load_one_session(dir, int(session_id)))

    return sessions