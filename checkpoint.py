from typing import Dict
from typing import Any

import logging

import torch
import torch.nn as nn
import torch.backends
from torch.optim import Optimizer


def save_checkpoint(checkpoint_path: str, epoch: int, model: nn.Module,
                    optimizer: Optimizer,
                    test_loss: float, test_accuracy: float) -> None:
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }, checkpoint_path)
    logging.info(f"Saved checkpoint for epoch {epoch + 1}.")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer: Optimizer, use_cuda: bool) -> int:
    device: str = 'cpu'
    if use_cuda:
        device: str = 'cuda'

    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_idx: int = checkpoint['epoch']
    test_accuracy: float = checkpoint['test_accuracy']
    test_loss: float = checkpoint['test_loss']

    logging.info(f"Loaded checkpoint after epoch: {epoch_idx}. ")
    logging.info(f"Checkpoint test accuracy: {test_accuracy}.")
    logging.info(f"Checkpoint test loss: {test_loss}.")

    return epoch_idx
