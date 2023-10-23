from torch import nn
import torch
import argparse
from ..models import MLP_1hl, CNNLSTM, ResNetLSTM


def get_model(
    args: argparse.Namespace, 
    device
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
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
        torch.optim.Optimizer: optimizer to use with this model
        nn.Module: criterion to use with this model
    """
    if args.model == 'mlp_1hl':
        try:
            model = MLP_1hl(n_hl=10, n_features=303).to(device)
        except Exception as e:
            print(e)
            return tuple([None]*3)
        
        optimizer = MLP_1hl.get_optimizer(model)
        criterion = MLP_1hl.get_criterion()

        return (model, optimizer, criterion)

    elif args.model == 'cnnlstm':
        try:
            model = CNNLSTM(winsize=args.winsize).to(device)
        except Exception as e:
            print(e)
            return tuple([None]*3)
        
        optimizer = CNNLSTM.get_optimizer(model)
        criterion = CNNLSTM.get_criterion()

        return (model, optimizer, criterion)

    elif args.model == 'resnetlstm':
        try:
            model = ResNetLSTM(winsize=args.winsize).to(device)
        except Exception as e:
            print(e)
            return tuple([None]*3)
        
        optimizer = ResNetLSTM.get_optimizer(model)
        criterion = ResNetLSTM.get_criterion()

        return (model, optimizer, criterion)

    else:
        return tuple([None]*3)