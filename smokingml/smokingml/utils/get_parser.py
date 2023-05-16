import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog='main_mlp_1hl',
        description='Python script to train smoking neural networks'
    )

    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Size of batches for training')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('-i', '--hidden', default=10, help='Number of hidden layer neurons')
    parser.add_argument('-w', '--winsize', default=500, help='Window size of this dataset')
    parser.add_argument('-u', '--dataset', default='.', help='Data directory')
    parser.add_argument('-m', '--mem-batches', type=int, default=10000, help='Batch size for memory concerns during testing')
    parser.add_argument('-d', '--device', type=int, default=0, help='Cuda device index to use')
    parser.add_argument('-p', '--project', type=str, default=f'projects', help='Project directory to save results')
    parser.add_argument('-t', '--test', action='store_true', help='Set to true if only testing a model, not training')
    parser.add_argument('--testmodel', type=str, default=None, help='Path to model to test, ignored if --test is False, required if --test is True')

    return parser