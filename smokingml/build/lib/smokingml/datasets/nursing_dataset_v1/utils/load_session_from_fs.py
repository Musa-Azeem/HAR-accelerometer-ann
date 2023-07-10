from pathlib import Path
import torch

def load_session_from_fs(dir: Path, session_id: int) -> torch.tensor:
    return torch.load(dir / f'{session_id}' / 'X.pt')