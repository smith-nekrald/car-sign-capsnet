from typing import List
from typing import Optional
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
    def __init__(self, root_dir: str, annotations_path: Optional[str],
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

    def existence_tweak(self):
        verified_idx2class: List[int] = list()
        verified_idx2name: List[str] = list()
        for class_id, relative_path in zip(self.idx2class, self.idx2name):
            image_path: str = os.path.join(self.root_dir, relative_path)
            if os.path.exists(image_path):
                verified_idx2class.append(class_id)
                verified_idx2name.append(relative_path)
        self.idx2class = verified_idx2class
        self.idx2name = verified_idx2name

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

        column_names: List[str] = ['file_name', 'img_width', 'img_height',
                                   'sign_x_top', 'sign_y_top',
                                   'sign_x_bottom', 'sign_y_bottom', 'label']
        self.annotation_df: pd.DataFrame = pd.read_csv(path_to_annotations, sep=';', header=None,
                                                       index_col=False, names=column_names)

        self.idx2class: List[int] = list(self.annotation_df['label'])
        self.idx2name: List[str] = list(self.annotation_df['file_name'])

        self.n_classes: int = max(self.idx2class) + 1
        self.existence_tweak()


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
        self.existence_tweak()


class RussianDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
        self.annotation_df: pd.DataFrame = pd.read_csv(
            path_to_annotations, sep=',', header=0, index_col=False)
        self.idx2class: List[int] = list(
            self.annotation_df[TableColumns.RUSSIAN_CLASS_COLUMN])
        self.idx2name: List[str] = list(
            self.annotation_df[TableColumns.RUSSIAN_PATH_COLUMN])

        self.n_classes: int = max(self.idx2class) + 1
        self.existence_tweak()


class BelgiumDataset(DynamicDataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: Optional[str],
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        super().__init__(path_to_img_dir, path_to_annotations, static_transform, random_transform)
        self.n_classes = 0
        sub_folder: str
        for sub_folder in os.listdir(path_to_img_dir):
            candidate_directory: str = os.path.join(path_to_img_dir, sub_folder)
            if os.path.isdir(candidate_directory):
                image_name: str
                for image_name in os.listdir(candidate_directory):
                    if image_name.endswith(".ppm"):
                        image_class: int = int(sub_folder)
                        image_path: str = os.path.join(sub_folder, image_name)
                        self.idx2class.append(image_class)
                        self.idx2name.append(image_path)
                        self.n_classes = max(self.n_classes, image_class + 1)
