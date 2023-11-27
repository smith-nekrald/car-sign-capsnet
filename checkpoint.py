""" Implements API for saving and loading checkpoints. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import Dict
from typing import Any

import logging

import torch
import torch.nn as nn
import torch.backends
from torch.optim import Optimizer


def save_checkpoint(checkpoint_path: str, epoch: int, 
                    model: nn.Module, optimizer: Optimizer,
                    test_loss: float, test_accuracy: float,
                    train_loss: float, train_accuracy: float) -> None:
    """ Saves current model state into a checkpoint. 

    Args:
        checkpoint_path: The path to save checkpoint.
        epoch: The number of current epoch.
        model: The current model. For saving, state_dict is extracted.
        optimizer: The current optimizer. For saving, state_dict is extracted.
        test_loss: The value of current test loss.
        test_accuracy: The value of current test accuracy.
        train_loss: The value of current train loss.
        train_accuracy: The value of current train accuracy.
    """
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy
    }, checkpoint_path)
    logging.info(f"Saved checkpoint for epoch {epoch + 1}.")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer: Optimizer, use_cuda: bool) -> int:
    """ Loads model parameters from checkpoint. 

    Args:
        checkpoint_path: The path to checkpoint.
        model: The network module for loading model parameters.
        optimizer: The optimizer module for loading optimizer parameters.
        use_cuda: Whether to use CUDA.

    Returns:
        The index of the epoch corresponding to loaded checkpoint.
    """
    device: str = 'cpu'
    if use_cuda:
        device: str = 'cuda'

    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_idx: int = checkpoint['epoch']
    test_loss: float = checkpoint['test_loss']
    test_accuracy: float = checkpoint['test_accuracy']
    train_loss: float = checkpoint['train_loss']
    train_accuracy: float = checkpoint['train_accuracy']

    logging.info(f"Loaded checkpoint after epoch: {epoch_idx}. ")
    logging.info(f"Checkpoint test loss: {test_loss}.")
    logging.info(f"Checkpoint test accuracy: {test_accuracy}.")
    logging.info(f"Checkpoint train loss: {train_loss}.")
    logging.info(f"Checkpoint train accuracy: {train_accuracy}.")

    return epoch_idx

