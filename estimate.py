from typing import List
from typing import Type
from typing import Tuple
from typing import Union

import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as T

from dataset import DynamicDataset

TransformType = Union[None, Compose, Module]
NormalizationTyping = Union[Tuple[float, float], Tuple[List[float], List[float]]]


def make_estimating_transformation(
        use_grayscale: bool, image_size: Tuple[int, int]) -> TransformType:
    estimate_list: List[TransformType] = [T.ToTensor(), T.Resize(size=image_size) ]
    if use_grayscale:
        estimate_list.append(T.Grayscale())
    estimate_transform: TransformType = T.Compose(estimate_list)
    return estimate_transform


def estimate_normalization(path_to_img_root: str, path_to_annotations: str,
                           DataSetType: Type[DynamicDataset],
                           use_grayscale: bool, image_size: Tuple[int, int],
                           estimation_length: int) -> NormalizationTyping:
    estimate_transform: TransformType = make_estimating_transformation(
        use_grayscale, image_size)
    dataset: DataSetType = DataSetType(path_to_img_root, path_to_annotations,
                          estimate_transform, estimate_transform)
    loader: DataLoader = DataLoader(
        dataset, batch_size=estimation_length, shuffle=True)

    batch, target = next(iter(loader))
    batch_mean: Union[float, np.array] = batch.mean().data
    batch_std: Union[float, np.array] = batch.std().data
    return batch_mean, batch_std
