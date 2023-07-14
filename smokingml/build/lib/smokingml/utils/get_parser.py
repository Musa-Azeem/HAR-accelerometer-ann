import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog='main',
        description='Python script to train smoking neural networks'
    )

    parser.add_argument('-m', '--model', default='mlp_1hl', type=str, help='Model to train')
    parser.add_argument('-P', '--dataset-path', default='data/nursingv1_dataset', type=str, help='Path to nursing v1 dataset')
    parser.add_argument('-i', '--n-sessions', default=10, type=int, help='Number of sessions from dataset to use')
    parser.add_argument('-S', '--shuffle', action='store_true', help='Set to shuffle dataset')
    parser.add_argument('-s', '--dev-size', default='0.3', type=float, help='Amount of dataset to use as dev set')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Size of batches for training and testing')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('-d', '--device', type=int, default=0, help='Cuda device index to use')
    parser.add_argument('-p', '--project', type=str, default=f'nursingv1_projects', help='Project directory to save results')
    parser.add_argument('-t', '--test', action='store_true', help='Set to true if only testing a model, not training')
    parser.add_argument('-T', '--testmodel', type=str, default=None, help='Path to model to test, ignored if --test is False, required if --test is True')

    return parser