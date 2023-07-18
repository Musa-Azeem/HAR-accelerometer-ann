#!/usr/bin/python3

import torch
from datetime import datetime
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader
from smokingml.utils import get_parser, Colors, get_model, plot_and_save_cm
from smokingml.datasets.nursing_dataset_v1 import (
    WINSIZE,
    NursingDatasetV1,
    nursingv1_train_dev_test_split,    
    utils
)
from smokingml.modules import (
    optimization_loop,
    evaluate_loop
)
import sys

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
        'run-command': str(sys.argv),
        'date': date,
        'model': args.model,
        'dataset': 'Nursing v1',
        'n_sessions': args.n_sessions,
        'shuffle': args.shuffle,
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
    model,optimizer,criterion = get_model(args, device)
    if not model:
        print(f'No model of name {args.model} or not all args provided')
        exit(1)

    if args.test:
        try:
            model.load_state_dict(torch.load(args.testmodel))
        except:
            print('Error in path to model')
            exit(1)

    # Create directory for results
    outdir = Path(args.project) / f'{date}'
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(f'{outdir}/config.json', 'w'), indent=4)

    # Get dataset
    nursingv1_dir = Path(args.dataset_path)

    try:
        if args.n_sessions > 0:
            session_ids = utils.get_all_session_ids(nursingv1_dir)[:args.n_sessions]
        else:
            session_ids = utils.get_all_session_ids(nursingv1_dir)
    except FileNotFoundError as e:
        print(f'Error in path to dataset directory - {e}')
        exit(1)
    
    # Using this api to dataset will split train and dev by session
    train_dataset, dev_dataset, _ = nursingv1_train_dev_test_split(
        nursingv1_dir,
        train_size=1-args.dev_size,
        dev_size=args.dev_size,
        test_size=0.0,
        shuffle=args.shuffle,
        session_ids=session_ids
    )
    trainloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=args.shuffle)
    devloader = DataLoader(dev_dataset, batch_size=args.batch, shuffle=args.shuffle)

    # Train model
    if not args.test:
        optimization_loop(
            model=model,
            trainloader=trainloader,
            devloader=devloader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=args.epochs,
            device=device,
            outdir=outdir
        )

    # Test Model
    y_true_dev, y_pred_dev, test_loss = evaluate_loop(
        model=model,
        criterion=criterion,
        loader=devloader,
        device=device
    )
    y_true_dev,y_pred_dev = y_true_dev.flatten(), y_pred_dev.flatten()
    plot_and_save_cm(y_true=y_true_dev, y_pred=y_pred_dev, filename=str(outdir / 'cm.jpg'))

    end = time.time()
    print(f'{Colors.OKGREEN}Complete. Elapsed Time: {end-start:.3f}{Colors.ENDC}')
