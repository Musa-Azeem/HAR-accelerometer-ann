import torch
from tqdm import tqdm
from torch.utils import DataLoader
from torch import nn
import os
from utils import plot_and_save_loss

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
    """
        Trains a model on the provided training dataset. Tests the model on the
        provided test dataset every epoch. Saves the model each epoch and live 
        updates 'loss.png' in the home directory as the model trains. On 
        completion, returns the model and saves the train and test loss of each
        epoch in the results folder

    Args:
        train_dataset (torch.utils.data.Dataset): dataset to train model
        test_dataset (torch.utils.data.Dataset): dataset to test model
        model (torch.nn): model to train
        epochs (int): number of epochs to train the model for
        batch_size (int): batch size for training
        optimizer (torch.optim.Optimizer): optimizer to use during training
        criterion (function): loss function for training
        date (str): timestamp for directories
        device (str): gpu or cpu device

    Returns:
        torch.nn: trained model
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

    # Train and test each epoch
    for i,epoch in enumerate(range(epochs)):

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
            loss = criterion(logits, y_test)


            # Each batch, save predictions, num of correct predictions, and the loss
            preds += pred.flatten().tolist()
            n_correct += sum(y_test == pred)
            test_losses[-1] += loss.item()

        # Get average loss over all batches and total percentage of correct predictions
        test_losses[-1] /= n_test_batches
        accuracy = (n_correct / test_length).item()

        print(f'\tTest Accuracy: {100*accuracy:.4}%')
        print(f'\tTest Loss: {test_losses[-1]}')

        # Each epcoh, save model in timestamped directory
        torch.save(model.state_dict(), f'model/{date}/model-epoch-{epoch}.pt')

        # Each epoch, update temprorary loss figure
        plot_and_save_loss(train_losses, test_losses, i, "loss.png")

    # Save all train and test losses for each epoch (for continuing training later)
    torch.save(train_losses, f'results/{date}training/train_losses.pt')
    torch.save(test_losses, f'results/{date}training/test_losses.pt')