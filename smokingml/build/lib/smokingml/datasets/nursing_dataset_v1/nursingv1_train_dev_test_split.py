from pathlib import Path
import numpy as np
from utils import get_all_session_ids
from . import NursingDatasetV1

def nursingv1_train_dev_test_split(
        dir: Path, 
        train_size: float, 
        dev_size: float, 
        test_size: float,
        shuffle: bool = False,
        session_ids: list[int] = None
    ) -> tuple:
        """
            Creates and returns three NursingDatasetV1 objects for train,
                dev, and test purposes. Each of the three objects are given
                a subset of the total sessions in the dataset. The number
                of sessions given to each dataset is set with train, dev,
                and test size parameters, which each represent a percentage 
                of the total number of sessions.
        Args:
            dir (Path): filepath to nursingv1 dataset in filesystem
            train_size (float): percent of sessions for train dataset
            dev_size (float): percent of sessions for dev dataset
            test_size (float): percent of sessions for test dataset
            shuffle (bool, optional): shuffle dataset before split. Defaults to False.

        Returns:
            tuple: Three NursingDatasetV1 objects (train, dev, test)
        """

        ## Check parameters:
        if not dir.is_dir():
            print("Error: directory does not exist")
            return None
        
        if sum([train_size, dev_size, test_size]) != 1:
            print("Error: train_size + dev_size + test_size != 1")
            return None

        ## Get list of all session ids in dataset or use provided ids
        if not session_ids:
            session_ids = get_all_session_ids(dir)
        
        ## Split sessions into train, dev, and test
        # Shuffle first if desired
        if shuffle:
            np.random.shuffle(session_ids)

        # Get size of partitions
        n_train_sessions = round(train_size * len(session_ids))
        n_dev_sessions = round(dev_size * len(session_ids))

        # Split sessions into three parts
        train_ids, dev_ids, test_ids = np.split(
            session_ids,
            [n_train_sessions, n_train_sessions + n_dev_sessions]
        )

        return (
            NursingDatasetV1(dir, train_ids, shuffle),
            NursingDatasetV1(dir, dev_ids, shuffle),
            NursingDatasetV1(dir, test_ids, shuffle)
        )