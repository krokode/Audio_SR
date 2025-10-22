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

        # Ensure tensors are (batch, channels, seq_len) for Conv1d.
        # Handle common dataset output shapes:
        # - (batch, seq_len) -> add channel dim -> (batch, 1, seq_len)
        # - (batch, seq_len, channels) where channels==1 -> permute to (batch, 1, seq_len)
        if inputs.dim() == 3:
            # If middle dimension is larger than last, assume (batch, seq_len, channels)
            if inputs.shape[1] > inputs.shape[2]:
                inputs = inputs.permute(0, 2, 1)
            # else assume it's already (batch, channels, seq_len)
        elif inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        if targets.dim() == 3:
            if targets.shape[1] > targets.shape[2]:
                targets = targets.permute(0, 2, 1)
        elif targets.dim() == 2:
            targets = targets.unsqueeze(1)

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

            # Ensure tensors are (batch, channels, seq_len) for Conv1d.
            if inputs.dim() == 3:
                if inputs.shape[1] > inputs.shape[2]:
                    inputs = inputs.permute(0, 2, 1)
            elif inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)

            if targets.dim() == 3:
                if targets.shape[1] > targets.shape[2]:
                    targets = targets.permute(0, 2, 1)
            elif targets.dim() == 2:
                targets = targets.unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate the loss
            total_loss += loss.item()

    # Return the average loss for the epoch
    return total_loss / len(dataloader)