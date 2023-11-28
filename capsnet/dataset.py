""" Implements Dataset API - family of PyTorch-friendly benchmark-specific classes 
used for transforming image input and creating batches. 
"""

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from abc import ABC
import os

import pandas as pd
from PIL import Image

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from capsnet.keys import TableColumns

TransformType = Union[None, Compose, Module]


class DynamicDataset(Dataset, ABC):
    """ DynamicDataset is a PyTorch friendly proxy for reading data and creating image batches. 
    The parent class Dataset does the vast majority of the job, so this class only implements
    some methods to fill the gaps between pre-implemented and modified API.
    
    Attributes:
        static_transform: Static initial transformation version, used at evaluation.
        random_transform: Random initial transformation version, used at training.
        transform: The currently set initial tranformation.
        idx2class: Maps entry index to class.
        idx2name: Maps entry index to relative path (often simply image name).
        root_dir: Path to root directory with dataset images.
        annotations_path: Path to file with annotations, when relevant.
        n_classes: Number of classes.
    """
    def __init__(self, root_dir: str, annotations_path: Optional[str],
                 static_transform: TransformType,
                 random_transform: TransformType) -> None:
        """ Initializer method. 

        Args:
            root_dir: Path to root directory with dataset images.
            annotations_path: Path to annotation file, if relevant.
            static_transform: Static image transformation version (used at evaluation).
            random_transform: Dynamic image transformation version (used at training).
        """
        super().__init__()
        self.static_transform: TransformType = static_transform
        self.random_transform: TransformType = random_transform
        if self.random_transform is None:
            self.random_transform = self.static_transform

        self.transform: TransformType = static_transform

        self.idx2class: List[int] = list()
        self.idx2name: List[str] = list()

        self.root_dir: str = root_dir
        self.annotations_path: Optional[str] = annotations_path
        self.n_classes: Optional[int] = None

    def existence_tweak(self) -> None:
        """ Tweak to ensure consistency with the class-path structures. Some benchmark have
        some annotations linking to non-existent images, and this method filters for such
        entries.
        """
        verified_idx2class: List[int] = list()
        verified_idx2name: List[str] = list()

        class_id: int; relative_path: str
        for class_id, relative_path in zip(self.idx2class, self.idx2name):
            image_path: str = os.path.join(self.root_dir, relative_path)
            if os.path.exists(image_path):
                verified_idx2class.append(class_id)
                verified_idx2name.append(relative_path)
        self.idx2class = verified_idx2class
        self.idx2name = verified_idx2name

    def static_mode(self) -> None:
        """ Turns static mode on. Now static_transform is applied when transform requested. """
        self.transform = self.static_transform

    def random_mode(self) -> None:
        """ Turns dynamic mode on. Now random_transform is applied when transform requested. """
        self.transform = self.random_transform

    def __getitem__(self, idx: Union[torch.Tensor, int]) -> Tuple[Image.Image, int]:
        """ Getter by index. Method needed re-implementation to link prepared and desired APIs. 

        Args:
            idx: Index of the object to get.
            
        Returns:
            Tuple with two elements. The first element is the transformed
                image, the second element is the image class label.
        """
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
        """ Method to request data length. Needs re-implementation for 
        linking prepared and desired APIs. 
        
        Returns:
            The number of entries in the dataset.
        """
        return len(self.idx2class)


class ChineseDataset(DynamicDataset):
    """ Specifies DynamicDataset for Chinese benchmark. 

    Attributes:
        annotation_df: The Pandas DataFrame with annotations.
    """
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: TransformType, random_transform: TransformType) -> None:
        """ Initializer method. 

        Args:
            path_to_img_dir: Path to root directory with dataset images.
            path_to_annotations_file: Path to file with annotations.
            static_transform: Static image transformation version (used at evaluation).
            random_transform: Dynamic image transformation version (used at training).

        """
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)

        column_names: List[str] = [
            'file_name', 'img_width', 'img_height', 'sign_x_top', 
            'sign_y_top', 'sign_x_bottom', 'sign_y_bottom', 'label']
        self.annotation_df: pd.DataFrame = pd.read_csv(
            path_to_annotations, sep=';', header=None, 
            index_col=False, names=column_names)

        self.idx2class: List[int] = list(self.annotation_df['label'])
        self.idx2name: List[str] = list(self.annotation_df['file_name'])

        self.n_classes = max(self.idx2class) + 1
        self.existence_tweak()


class GermanDataset(DynamicDataset):
    """ Specifies DynamicDataset for German benchmark. 

    Attributes:
        annotation_df: The Pandas DataFrame with annotations.
    """
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        """ Initializer method. 

        Args:
            path_to_img_dir: Path to root directory with dataset images.
            path_to_annotations_file: Path to file with annotations.
            static_transform: Static image transformation version (used at evaluation).
            random_transform: Dynamic image transformation version (used at training).

        """
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
        self.annotation_df: pd.DataFrame = pd.read_csv(
            path_to_annotations, sep=',', header=0, index_col=False)
        self.idx2class = list(
            self.annotation_df[TableColumns.GERMAN_CLASS_COLUMN])
        self.idx2name = list(
            self.annotation_df[TableColumns.GERMAN_PATH_COLUMN])

        self.n_classes = max(self.idx2class) + 1
        self.existence_tweak()


class RussianDataset(DynamicDataset):
    """ Specifies DynamicDataset for Russian benchmark. 

    Attributes:
        annotation_df: The Pandas DataFrame with annotations.
    """
    def __init__(self, path_to_img_dir: str, path_to_annotations: str,
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        """ Initializer method. 

        Args:
            path_to_img_dir: Path to root directory with dataset images.
            path_to_annotations_file: Path to file with annotations.
            static_transform: Static image transformation version (used at evaluation).
            random_transform: Dynamic image transformation version (used at training).

        """
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
        self.annotation_df: pd.DataFrame = pd.read_csv(
            path_to_annotations, sep=',', header=0, index_col=False)
        self.idx2class = list(
            self.annotation_df[TableColumns.RUSSIAN_CLASS_COLUMN])
        self.idx2name = list(
            self.annotation_df[TableColumns.RUSSIAN_PATH_COLUMN])

        self.n_classes = max(self.idx2class) + 1
        self.existence_tweak()


class BelgiumDataset(DynamicDataset):
    """ Specifies DynamicDataset for Belgium benchmark. """

    def __init__(self, path_to_img_dir: str, path_to_annotations: Optional[str],
                 static_transform: Union[None, Compose, Module],
                 random_transform: Union[None, Compose, Module]) -> None:
        """ Initializer method. 

        Args:
            path_to_img_dir: Path to root directory with dataset images.
            path_to_annotations_file: Path to file with annotations. Irrelevant for 
                this dataset, since class labels are encoded in file names.
            static_transform: Static image transformation version (used at evaluation).
            random_transform: Dynamic image transformation version (used at training).

        """
        super().__init__(path_to_img_dir, path_to_annotations,
                         static_transform, random_transform)
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


