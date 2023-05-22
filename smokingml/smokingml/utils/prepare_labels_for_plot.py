import numpy as np

def prepare_labels_for_plot(y: np.ndarray, win_size: int) -> np.ndarray:
    """
        Appends 0s on both sides of array of labels to prepare it for plotting.
        Also multiplies labels by 10 to make them more apparent on plot

    Args:
        y (np.ndarray): labels to prepare
        win_size (int): size of windows for current model

    Returns:
        np.ndarray: labels multiplied by 10 and padded with 0s on both sides
    """
    if win_size%2==0:
        return np.pad(
            y.flatten()*10, 
            (win_size//2-1, win_size//2), 
            mode='constant', 
            constant_values=0
        )
    else:
        return np.pad(
            y.flatten()*10, 
            (win_size//2, win_size//2), 
            mode='constant', 
            constant_values=0
        )