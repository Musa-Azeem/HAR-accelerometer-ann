import torch
from torch.utils.data import Dataset, TensorDataset
from pathlib import Path
import numpy as np
from . import WINSIZE
from . import dataloading

# Future idea - keep as many sessions in memory as possible

class NursingDatasetV1(Dataset):
    """
        Dataset class to handle the nursingv1_dataset
    """

    def __init__(
            self, 
            dir: Path, 
            session_ids: list[int], 
            shuffle: bool = False, 
        ) -> None:
        super().__init__()

        # Public attributes
        self.dir = dir
        self.session_ids = session_ids

        # Private attributes
        self._shuffle = shuffle
        
        ## Get info from session sizes
        
        # save sum of length of each session in dataset
        self.length = 0
        self.sessions = {}
        
        # Save mapping from each possible index to the session that window is in
        self._idx_to_session = []

        for session_id in self.session_ids:
            # Read files for this session (shape, data, labels)
            X = torch.load(dir / f'{session_id}' / 'X.pt')
            y = torch.load(dir / f'{session_id}' / 'y.pt')

            # Save session/labels pair
            self.sessions[session_id] = X,y
            
            # Save number of windows, which is session length - winsize + 1
            lengthi = X.shape[0] - WINSIZE + 1
            self.length += lengthi

            # Save which indices should map to this session as tuple (<session id>, <idx of window in that session>)
            self._idx_to_session += zip([session_id]*lengthi, list(range(lengthi)))

        # Save random mapping of internal window indices to external indices (for shuffling)
        self._idxs = list(range(self.length))
        if shuffle:
            np.random.shuffle(self._idxs)
        

    def __getitem__(self, index: int) -> torch.Tensor:
        # Return one single window from one of the sessions and its label
        # return data in shape for convolution rather than linear input for now

        # For now, only support postive integer indices
        if not isinstance(index, int) or index < 0:
            print("Error: Unsupported index type")
            return None
        

        ## Get session to choose window from based on index
        # Use random mapping to choose random index
        idx = self._idxs[index]     # Will catch index out of bounds
        x,y = self._get_one_window_and_label(idx)

        # Flatten windows
        x,y = x.T.flatten(),y.flatten()
        
        return (x.float(),y.float())

    def _get_one_window_and_label(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        
        # Get the session that this idx is in and the idx within that session
        session_id, window_idx = self._idx_to_session[idx]

        # Window session starting at window_idx
        window = self.sessions[session_id][0][window_idx:window_idx+WINSIZE]
        label = self.sessions[session_id][1][window_idx]

        return (window, label)
    
    def __len__(self) -> int:
        # Total number of windows in every session is length of dataset
        return self.length

    def load_one_session(self, session_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get one unwindowed session (padded) from session_ids and its labels
        # Only return session if it is a part of this dataset

        if session_id not in self.session_ids:
            print("Error: Session id not a part of this dataset")
            return None
        
        return dataloading.load_one_session(self.dir, session_id)

    def load_all_sessions(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # return list of all unwindowed sessions (padded) and their labels in this dataset
        return dataloading.load_sessions(self.dir, self.session_ids)

    def load_one_windowed_session(self, session_id) -> TensorDataset:
        # Return one windowed session and its labels as tensor dataset
        if session_id not in self.session_ids:
            print("Error: Session id not a part of this dataset")
            return None

        return dataloading.load_one_windowed_session(self.dir, session_id)

    def load_all_windowed_sessions(self) -> list[TensorDataset]:
        # Return all windowed sessions and their labels as list of tensor datasets
        return dataloading.load_windowed_sessions(self.dir, self.session_ids)