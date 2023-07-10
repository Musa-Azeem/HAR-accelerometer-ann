import torch
from torch.utils.data import Dataset, TensorDataset
from pathlib import Path
import numpy as np
from . import WINSIZE
from . import dataloading

class NursingDatasetV1(Dataset):
    """
        Dataset class to handle the nursingv1_dataset
    """

    def __init__(self, dir: Path, session_ids: list[int], shuffle: bool = False) -> None:
        super().__init__()

        # Public attributes
        self.dir = dir
        self.session_ids = session_ids

        # Private attributes
        self._shuffle = shuffle
        
        ## Get info from session sizes
        
        # save length of each session in dataset - TODO might be able to replace this with just a sum - dont need to save lengths and rn its useless
        self._lengths = []
        
        # Save mapping from each possible index to the session that window is in
        self._idx_to_session = []

        for session_id in self.session_ids:
            # Get shape of session from dataset
            session_shape = torch.load(dir / f'{session_id}' / 'Xshape.pt')
            
            # Save number of windows, which is session length - winsize + 1
            self._lengths.append(session_shape[1] - WINSIZE + 1)

            # Save which indices should map to this session as tuple (<session id>, <idx of window in that session>)
            self._idx_to_session += zip([session_id]*self._lengths[-1], list(range(self._lengths[-1])))
            # print(session_id, ':', self._idx_to_session[-1], '---', self._lengths[-1])


        # Save random mapping of internal window indices to external indices (for shuffling)
        self._idxs = list(range(sum(self._lengths)))
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
        return (x,y)

    def _get_one_window_and_label(self, idx: int) -> tuple[torch.Tensor]:
        
        # Get the session that this idx is in and the idx within that session
        session_id, window_idx = self._idx_to_session[idx]

        # Read whole session and label files
        X = torch.load(self.dir / f'{session_id}' / 'X.pt')
        y = torch.load(self.dir / f'{session_id}' / 'y.pt')
        # print(session_id, window_idx, X.shape[1] - WINSIZE +1)

        # Window session starting at window_idx
        window = X[:, window_idx:window_idx+WINSIZE]
        label = y[window_idx]

        return (window, label)

    def __len__(self) -> int:
        # Total number of windows in every session is length of dataset
        return sum(self._lengths)

    def load_one_session(self, session_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get one unwindowed session from session_ids and its labels (labels are padded)
        # Only return session if it is a part of this dataset

        if session_id not in self.session_ids:
            print("Error: Session id not a part of this dataset")
            return None
        
        return dataloading.load_one_session(self.dir, session_id)

    def load_all_sessions(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # return list of all unwindowed sessions and their labels in this dataset
        return dataloading.load_sessions(self.dir, self.session_ids)

    def load_one_windowed_session(self, id) -> TensorDataset:
        # Return one windowed session and its labels as tensor dataset
        return dataloading.load_one_windowed_session(self.dir, id)

    def load_all_windowed_sessions(self) -> list[TensorDataset]:
        # Return all windowed sessions and their labels as list of tensor datasets
        pass