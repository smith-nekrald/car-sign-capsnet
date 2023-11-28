""" Implements API related to estimating some static properties from dataset before training. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional

import numpy as np

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as T

from dataset import DynamicDataset

TransformType = Union[None, Compose, Module]
NormalizationTyping = Union[Tuple[float, float], Tuple[List[float], List[float]]]
TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


def make_estimating_transformation_base(
        use_grayscale: bool, image_size: Tuple[int, int]) -> TransformType:
    """ Creates basic image transformation, that may be further 
    composed with other transformations. Essentially, converts 
    to tensor, resizes, and converts to grayscale if requested.

    Args:
        use_grayscale: Whether to convert to grayscale.
        image_size: Desired image size.

    Returns:
        Compiled and ready-to-use transformation.
    """
    estimate_list: List[TransformType] = [T.ToTensor(), T.Resize(size=image_size) ]
    if use_grayscale:
        estimate_list.append(T.Grayscale())
    estimate_transform: TransformType = T.Compose(estimate_list)
    return estimate_transform


def estimate_normalization(path_to_img_root: str, path_to_annotations: Optional[str],
                           DataSetType: Type[DynamicDataset],
                           use_grayscale: bool, image_size: Tuple[int, int],
                           estimation_length: int) -> NormalizationTyping:
    """ Estimates mean value and standard deviation by reading a batch of images. 

    Args:
        path_to_img_root: Path to the folder with images.
        path_to_annotations: Path to the file with annotations (if relevant).
        DataSetType: The relevant heir of DynamicDataset to handle dataset loading.
        use_grayscale: Whether to use grayscale transformation.
        image_size: Image shape (2-dimensional).
        estimation_length: The amount of images in a batch used for 
            estimating mean and standard deviation.

    Returns:
        Tuple with two entries. The first contains batch mean, the second contains
            batch standard deviation.
    """
    estimate_transform: TransformType = make_estimating_transformation_base(
        use_grayscale, image_size)
    dataset: DataSetType = DataSetType(path_to_img_root, path_to_annotations,
                          estimate_transform, estimate_transform)
    loader: DataLoader = DataLoader(
        dataset, batch_size=estimation_length, shuffle=True)

    batch: TypingFloatTensor; target: TypingIntTensor
    batch, target = next(iter(loader))
    batch_mean: Union[float, np.array] = batch.mean().data
    batch_std: Union[float, np.array] = batch.std().data
    return batch_mean, batch_std

