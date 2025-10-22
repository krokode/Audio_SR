import torch
import torch.nn as nn
from tqdm import tqdm # Provides a progress bar

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        device (torch.device or str): The device (e.g., "cuda" or "cpu").

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0

    # Iterate over the training data with a progress bar
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        # Move data to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Add channel dimension: (batch_size, seq_len) -> (batch_size, 1, seq_len)
        # This is required by the Conv1d layers in the model.
        # if inputs.dim() == 2:
        #     inputs = inputs.unsqueeze(1)
        # if targets.dim() == 2:
        #     targets = targets.unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item()

    # Return the average loss for the epoch
    return total_loss / len(dataloader)

def test_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the test/validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The test/validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device or str): The device (e.g., "cuda" or "cpu").

    Returns:
        float: The average test/validation loss for the epoch.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    # Disable gradient calculations
    with torch.no_grad():
        # Iterate over the test data
        for inputs, targets in tqdm(dataloader, desc="Testing ", leave=False):
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Add channel dimension: (batch_size, seq_len) -> (batch_size, 1, seq_len)
            # if inputs.dim() == 2:
            #     inputs = inputs.unsqueeze(1)
            # if targets.dim() == 2:
            #     targets = targets.unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate the loss
            total_loss += loss.item()

    # Return the average loss for the epoch
    return total_loss / len(dataloader)