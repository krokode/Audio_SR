import torch
import torch.nn as nn
from tqdm import tqdm # Provides a progress bar


def fix_shape(inputs):
    if inputs.dim() == 3:
        # If middle dimension is larger than last
        if inputs.shape[1] > inputs.shape[2]:
            inputs = inputs.permute(0, 2, 1)
            # print(f"traintest fix_shape dim==3 permute {inputs.shape}")
    # else assume it's already (batch, channels, seq_len)
    elif inputs.dim() == 2:
        # print(f"traintest fix_shape dim==2unsqueeze(1) {inputs.shape}")
        inputs = inputs.unsqueeze(1)
        
    return inputs
    


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

        inputs = fix_shape(inputs)
        targets = fix_shape(targets)

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

            inputs = fix_shape(inputs)
            targets = fix_shape(targets)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate the loss
            total_loss += loss.item()

    # Return the average loss for the epoch
    return total_loss / len(dataloader)
