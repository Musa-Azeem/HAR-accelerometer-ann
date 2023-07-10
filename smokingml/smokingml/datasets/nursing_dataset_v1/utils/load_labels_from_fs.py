from pathlib import Path
import torch

def load_labels_from_fs(dir: Path, session_id: int) -> torch.tensor:
    return torch.load(dir / f'{session_id}' / 'y.pt')