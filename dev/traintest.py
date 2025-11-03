import torch
import torch.nn as nn
from tqdm import tqdm # Provides a progress bar


def fix_shape(inputs, targets):
    """
    Ensure inputs and targets have correct shape for the model
    Model expects (batch, channels, length)
    """
    # If inputs are (batch, length), add channel dimension
    if len(inputs.shape) == 2:
        inputs = inputs.unsqueeze(1)  # (batch, 1, length)
        targets = targets.unsqueeze(1)  # (batch, 1, length)
    
    # If we somehow get transposed data, fix it
    if inputs.shape[1] != 1:
        inputs = inputs.transpose(1, 0)
        targets = targets.transpose(1, 0)
       
    return inputs, targets


def train_epoch(model, dataloader, batch_size, optimizer, criterion, device):
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
    batch_count = 0

    # Iterate over the training data with a progress bar
    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move data to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets = fix_shape(inputs=inputs, targets=targets)

        # Zero gradients at the start of each batch
        if i % batch_size == 0:
            optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()

        # Step optimizer only after accumulating batch_size gradients
        if (i + 1) % batch_size == 0:
            optimizer.step()
            batch_count += 1

        # Accumulate the loss
        total_loss += loss.item()

    # Return average loss per batch
    return total_loss / (batch_count if batch_count > 0 else 1)


def test_epoch(model, dataloader, batch_size, criterion, device):
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
    batch_count = 0

    # Disable gradient calculations
    with torch.no_grad():
        # Iterate over the test data
        for inputs, targets in tqdm(dataloader, desc="Testing ", leave=False):
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            inputs, targets = fix_shape(inputs, targets)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate the loss
            total_loss += loss.item()
            batch_count += 1

    return total_loss / batch_count