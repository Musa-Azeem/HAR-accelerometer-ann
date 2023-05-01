import torch
from tqdm import tqdm
from torch.utils import DataLoader
from torch import nn
import os

def train(
    train_dataset: torch.utils.data.Dataset, 
    test_dataset: torch.utils.data.Dataset, 
    model: torch.nn, 
    epochs: int, 
    batch_size: int, 
    optimizer: torch.optim.Optimizer, 
    criterion: function, 
    date: str, 
    device: str
) -> torch.nn:
    """_summary_

    Args:
        train_dataset (torch.utils.data.Dataset): _description_
        test_dataset (torch.utils.data.Dataset): _description_
        model (torch.nn): _description_
        epochs (int): _description_
        batch_size (int): _description_
        optimizer (torch.optim.Optimizer): _description_
        criterion (function): _description_
        date (str): _description_
        device (str): _description_

    Returns:
        torch.nn: _description_
    """

    # Directories to save results
    os.system(f'mkdir -p results/{date}/training')
    os.system(f'mkdir -p model/{date}')

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=10000)

    n_train_batches = len(train_dataloader)
    n_test_batches = len(test_dataloader)

    test_length = len(test_dataset)

    # Keep track of train and test loss
    train_losses = []
    test_losses = []



    for epoch in range(epochs):

        # Train Model over entire dataset in batches of batch_size
        print(f'Epoch {epoch} - Training')
        model.train()
        train_losses.append(0)

        # Train Loop
        for X_train, y_train in tqdm(train_dataloader):

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Forward Pass
            logits = model(X_train)
            loss = criterion(logits, y_train)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses[-1] += loss.item()

        # Get average loss over all batches for this epoch
        train_losses[-1] /= n_train_batches
        print(f'\tloss: {train_losses[-1]}')



        # Test model over entire train dataset
        print(f'Epoch {epoch} - Testing')
        model.eval()

        preds = []
        n_correct = 0
        test_losses.append(0)

        # Train loop
        for X_test, y_test in tqdm(test_dataloader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            # Forward pass - round sigmoid of logits to get prediction
            logits = model(X_test)
            pred = torch.round(nn.Sigmoid()(logits))

            n_correct += sum(y_test == pred)
            preds += pred.flatten().tolist()
            loss = criterion(logits, y_test)
            test_losses[-1] += loss.item()

        test_losses[-1] /= n_test_batches
        accuracy = (n_correct / test_length).item()
        print(f'\tTest Accuracy: {100*accuracy:.4}%')
        print(f'\tTest Loss: {test_losses[-1]}')

        torch.save(model.state_dict(), f'model/{date}/model-epoch-{epoch}.pt')

    torch.save(train_losses, f'results/{date}training/train_losses.pt')
    torch.save(test_losses, f'results/{date}training/test_losses.pt')