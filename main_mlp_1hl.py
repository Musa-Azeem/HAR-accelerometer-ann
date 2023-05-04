import torch
from datetime import datetime
import os
import json
import time
from lib.utils import get_parser, Colors
from lib.models.mlp_1hl import MLP_1hl
from lib.datasets import SmokingDataset_500ws
from lib.modules import train, test, validate_on_holdouts

if __name__=='__main__':

    # Command line arguments and current session date
    args = get_parser().parse_args()
    date = datetime.now().strftime("%m.%d.%y-%H-%M-%S")
    start = time.time()

    # Check Args
    if args.test and not args.testmodel:
        print('--testmodel is required if --test is set to True')
        exit(1)

    # Save config information for this session
    config = {
        'date': date,
        'dataset': args.dataset,
        'model': 'mlp_1hl',
        'hidden-layers': args.hidden,
        'epochs': args.epochs,
        'batch-size': args.batch,
        'learning-rate': args.lr,
        'test': args.test, 
        'tested-model': args.testmodel
    }

    # Get device for training (cpu or gpu)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Instantiate model
    model = MLP_1hl(n_hl=args.hidden, n_features=args.winsize*3).to(device)
    optimizer = MLP_1hl.get_optimizer(model)
    criterion = MLP_1hl.get_criterion()

    if args.test:
        try:
            model.load_state_dict(torch.load(args.testmodel))
        except:
            print('Error in path to model')
            exit(1)

    # Instantiate datasets
    train_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/train')
    test_dataset = SmokingDataset_500ws(f'{args.dataset}/4_all/test')

    # If the dataset has y_true saved as pytorch file, use that
    if os.path.isfile(f'{args.dataset}/y_true/y_test.pt'):
        y_true = torch.load(f'{args.dataset}/y_true/y_test.pt')
    else:
        # otherwise, iterate through test dataset and get each label
        y_true = test_dataset[:][1]

    # Create directory for results
    dir = f'{args.project}/{date}'
    os.system(f'mkdir -p {dir}')
    json.dump(config, open(f'{dir}/config.json', 'w'), indent=4)


    # Train model
    if not args.test:
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
        raw_dir=f'{args.dataset}/0_raw',
        date=date,
        criterion=criterion,
        batch_size=args.mem_batches,
        win_size=args.winsize,
        device=device,
        project=dir
    )

    end = time.time()
    print(f'{Colors.OKGREEN}Complete. Elapsed Time: {end-start:.3f}{Colors.ENDC}')