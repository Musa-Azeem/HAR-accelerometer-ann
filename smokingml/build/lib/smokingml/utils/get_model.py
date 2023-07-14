from torch import nn
import torch
import argparse
from smokingml.models import MLP_1hl


def get_model(args: argparse.Namespace, device) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module, bool]:
    """
        Returns the model corrosponding with the given name
        Also returns whether or not this model requires input
            data in cnn format (multidimensial windows)

    Args:
        args (str): argparse namespace object
            contains model name and other model parameters
        device (str): device to train model on (cpu or cuda)

    Returns:
        nn.Module: instance of model
        bool: whether or not this model is a cnn
    """

    if args.model == 'mlp_1hl':
        # try:
        model = MLP_1hl(n_hl=10, n_features=303).to(device)
        # except:
        #     return (None, None, None, None)
        
        optimizer = MLP_1hl.get_optimizer(model)
        criterion = MLP_1hl.get_criterion()
        is_cnn = False

        return (model, optimizer, criterion, is_cnn)

    else:
        return (None, None, None, None)