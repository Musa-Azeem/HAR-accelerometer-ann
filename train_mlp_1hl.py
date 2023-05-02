import torch
import argparse
import os
from lib.models.mlp_1hl import MLP_1hl

if __name__=='__main__':

    win_size = 500
    n_hl = 10

    model = MLP_1hl(n_hl=n_hl, n_features=win_size*3)

