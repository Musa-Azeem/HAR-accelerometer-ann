from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

def train_test_split_windows(dataset: TensorDataset, test_size: float) -> tuple[TensorDataset, TensorDataset]:
    Xtr,Xte,ytr,yte = train_test_split(dataset.tensors[0], dataset.tensors[1], test_size=test_size)
    return (
        TensorDataset(Xtr,ytr),
        TensorDataset(Xte,yte)
    )