from pathlib import Path
import torch
import load_one_session

def load_sessions(dir: Path, session_ids: list[int]) -> list[tuple[torch.tensor, torch.tensor]]:
    # get unwindowed sessions listed in param and their labels (labels are padded)

    sessions = []
    for session_id in session_ids:
        sessions.append(load_one_session(dir, int(session_id)))

    return sessions