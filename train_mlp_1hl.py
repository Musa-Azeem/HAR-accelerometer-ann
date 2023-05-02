import torch
from datetime import datetime
import os
import json
from lib.utils import get_parser
from lib.models.mlp_1hl import MLP_1hl
from lib.datasets import SmokingDataset_500ws
from lib.modules import train, test, validate_on_holdouts

if __name__=='__main__':

    # Command line arguments and current session date
    date = datetime.now().strftime("%m.%d.%y-%H-%M-%S")
    args = get_parser(date).parse_args()

    # Save config information for this session
    config = {
        'date': date,
        'dataset': args.dataset,
        'model': 'mlp_1hl',
        'hidden-layers': args.hidden,
        'epochs': args.epochs,
        'batch-size': args.batch,
        'learning-rate': args.lr
    }

    # Get device for training (cpu or gpu)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Instantiate model
    model = MLP_1hl(n_hl=args.hidden, n_features=args.winsize*3).to(device)
    optimizer = MLP_1hl.get_optimizer(model)
    criterion = MLP_1hl.get_criterion()

    # Instantiate datasets
    train_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/train')
    test_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/test')
    y_true = test_dataset[:][1] # do this once, it takes long -_-

    # Create directory for results
    dir = f'{args.project}/{date}'
    os.system(f'mkdir -p {dir}')
    json.dump(config, open(f'{dir}/config.json', 'w'), indent=4)


    # Train model
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch,
        test_batch_size=args.mem_batches,
        optimizer=optimizer,
        criterion=criterion,
        date=date,
        device=device,
        project=dir
    )

    # Test Model
    test(
        model=model,
        dataset=test_dataset,
        y_true=y_true,
        device=device,
        criterion=criterion,
        batch_size=args.mem_batches,
        date=date,
        project=dir
    )

    # Validate model on holdout sets
    validate_on_holdouts(
        model=model,
        holdout_dir=f'{args.dataset}/holdouts',
        df_dir=f'{args.dataset}/1_xyz',
        date=date,
        criterion=criterion,
        batch_size=args.mem_batches,
        win_size=args.winsize,
        device=device,
        project=dir
    )