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
    args = get_parser().parse_args()
    date = datetime.now().strftime("%m.%d.%y-%H-%M-%S")

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

    # Instantiate datasets
    train_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/train')
    test_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/test')

    # Create directory for results
    os.system(f'mkdir {args.project}')
    json.dump(config, open(f'{args.project}/config.json', 'w'))

    # Train model
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch,
        test_batch_size=args.mem_batches,
        optimizer=MLP_1hl.get_optimizer(model),
        criterion=MLP_1hl.get_criterion(),
        date=date,
        device=device,
        project=args.project
    )





