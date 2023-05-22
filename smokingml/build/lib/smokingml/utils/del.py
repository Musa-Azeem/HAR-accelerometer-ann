import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from typing import List

def plot_and_save_cm(
    y_true: List[int], 
    y_pred: List[int], 
    filename: str
) -> None:
    """ 
        Plot and save confusion matrix (recall, precision, and total) for 
        given true labels and predictions. Saves plot to image with given 
        filename

    Args:
        y_true (list[int]): True labels, 0 or 1 for each example
        y_pred (list[int]): Predictions - same length as y_true
        filename (str): file name and path to save image as
    """

    fig,axes = plt.subplots(1,3,sharey=True,figsize=(10,5))

    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='true'),annot=True,ax=axes[0],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred,normalize='pred'),annot=True,ax=axes[1],cbar=False,fmt='.2f')
    sns.heatmap(confusion_matrix(y_true=y_true,y_pred=y_pred),annot=True,ax=axes[2],cbar=False,fmt='.2f')

    axes[0].set_title('Recall')
    axes[1].set_title('Precision')
    axes[2].set_title('Count')
    fig.set_size_inches(16, 9)
    
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()

