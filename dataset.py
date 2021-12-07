from typing import List
from typing import Type
from typing import Tuple
from typing import Union

from abc import ABC
import os
import logging

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as T

from keys import TableColumns

TransformType = Union[None, Compose, Module]


class DynamicDataset(Dataset, ABC):
    def __init__(self, root_dir: str, annotations_path: str,
                 static_transform: TransformType,
                 random_transform: TransformType) -> None:
        super().__init__()
        self.static_transform: TransformType = static_transform
        self.random_transform: TransformType = random_transform
        if self.random_transform is None:
            self.random_transform = self.static_transform

        self.transform: TransformType = static_transform

        self.idx2class: List[int] = list()
        self.idx2name: List[str] = list()

        self.root_dir: str = root_dir
        self.annotations_path: str = annotations_path

    def static_mode(self):
        self.transform = self.static_transform

    def random_mode(self):
        self.transform = self.random_transform

    def __getitem__(self, idx: Union[torch.Tensor, int]) -> Tuple[Image.Image, int]:
        if torch.is_tensor(idx):
            idx = idx.item()
        assert isinstance(idx, int)

        image_path: str = os.path.join(self.root_dir, self.idx2name[idx])
        label: int = self.idx2class[idx]

        image: Image.Image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return (image, label)

    def __len__(self) -> int:
        return len(self.idx2class)


class ChineseDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: TransformType,
                 random_transform: TransformType) -> None:
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)

        column_names: List[str] = ['file_name', 'img_width', 'img_height', 'sign_x_top', 'sign_y_top',
                                   'sign_x_bottom', 'sign_y_bottom', 'label']
        self.annotation_df: pd.DataFrame = pd.read_csv(path_to_annotations, sep=';', header=None,
                                                       index_col=False, names=column_names)

        self.idx2class: List[int] = list(self.annotation_df['label'])
        self.idx2name: List[str] = list(self.annotation_df['file_name'])

        self.n_classes: int = max(self.idx2class) + 1


class GermanDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
        self.annotation_df: pd.DataFrame = pd.read_csv(
            path_to_annotations, sep=',', header=0, index_col=False)
        self.idx2class: List[int] = list(
            self.annotation_df[TableColumns.GERMAN_CLASS_COLUMN])
        self.idx2name: List[str] = list(
            self.annotation_df[TableColumns.GERMAN_PATH_COLUMN])

        self.n_classes: int = max(self.idx2class) + 1


class RussianDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def __getitem(self, idx):
        raise NotImplemented


class BelgianDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        super().__init__(path_to_img_dir, path_to_annotations, static_transform, random_transform)
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def __getitem(self, idx):
        raise NotImplemented
