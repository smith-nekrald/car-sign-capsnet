""" Implements Benchmarks and related API. 
Benchmark is designed to process all job
related to loading and transforming images. 
"""

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import ChineseDataset
from dataset import GermanDataset
from dataset import BelgiumDataset
from dataset import RussianDataset
from dataset import TransformType
from dataset import DynamicDataset
from config import ConfigBenchmark
from keys import ColorSchema
from keys import BenchmarkName
from keys import FileFolderPaths
from estimate import estimate_normalization

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


class IBenchmark:
    """ Benchmark Interface. Benchmark is designed to process all job 
    related to loading and transforming images.  
    
    Attributes:
        normalize_transform: Normalization transformation. Standard scaling for images.
        recover_normalize: Inverse transformatino to normalize_transform.
        static_transform: Static transform version. Used at evaluation.
        random_transform: Random transform version. Used at training.
        train_dataset: The train dataset, benchmark-specific DynamicDataset for training part.
        test_dataset: The test dataset, benchmark-specific DynamicDataset for testing part.
        train_loader: Loader for training part.
        test_loader: Loader for testing part.
        use_cuda: Whether to use CUDA.
        num_workers: Number of parallel workers in loaders.
    """
    def __init__(self, config: ConfigBenchmark) -> None:
        """ Initializer method. Prepares static and random transformations.
        
        Args:
            config: Benchmark configuration.
        """
        static_list: List[TransformType] = list()
        random_list: List[TransformType] = list()

        transform_list: List[TransformType]
        for transform_list in [static_list, random_list]:
            transform_list += [T.ToTensor(), T.Resize(size=config.image_size)]

        static_core_transforms: List[TransformType] = [
            T.RandomAffine(degrees=15, shear=2, translate=(0.1, 0.1)),
            T.RandomVerticalFlip(p=1),
            T.RandomHorizontalFlip(p=1),
            T.RandomRotation(degrees=15),
            T.ColorJitter(contrast=(0.5, 2.0), saturation=(0.5, 2.0),
                          brightness=(0.5, 2.0), hue=0.4),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.RandomPerspective(distortion_scale=0.6, p=1.0),
            T.RandomResizedCrop(size=config.image_size),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(p=1),
        ]
        random_core_transforms: List[TransformType] = list()
        entry: TransformType
        for entry in static_core_transforms:
            random_core_transforms.append(
                T.RandomApply(transforms=[entry], p=config.random_entry_proba)
            )

        augment_transform: TransformType = T.RandomApply(
            transforms=random_core_transforms, p=config.augment_proba)
        random_list.append(augment_transform)

        mean: np.array = np.array(config.mean_normalize)
        std: np.array = np.array(config.std_normalize)
        entry_list: List[TransformType]
        for entry_list in [static_list, random_list]:
            if config.image_color == ColorSchema.GRAYSCALE:
                entry_list.append(T.Grayscale())
            entry_list.append(T.Normalize(mean.tolist(), std.tolist()))

        self.normalize_transform = T.Normalize(mean.tolist(), std.tolist())
        self.recover_normalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

        self.static_transform: TransformType = T.Compose(static_list)
        self.random_transform: TransformType = T.Compose(random_list)

        if not config.use_augmentation:
            self.random_transform = self.static_transform

        self.train_dataset: Optional[DynamicDataset] = None
        self.test_dataset: Optional[DynamicDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        self.use_cuda: bool = config.use_cuda
        self.num_workers: int = config.num_load_workers

    def set_random_mode(self) -> None:
        """ Sets train dataset to random transformation mode. """
        if self.train_dataset is not None:
            self.train_dataset.random_mode()

    def set_static_mode(self) -> None:
        """ Sets train dataset to static transformation mode. """
        if self.train_dataset is not None:
            self.train_dataset.static_mode()

    def init_loaders(self, config: ConfigBenchmark) -> None:
        """ Loader initialization. Sets train dataset to random mode, test dataset
        to static mode, and calls loader-specific methods.

        Args:
            config: Benchmark configuration.
        """
        self.train_dataset.random_mode()
        self.test_dataset.static_mode()

        self.reset_train_loader(config.batch_size)
        self.reset_test_loader(config.batch_size, False)

        self.n_classes: int = self.train_dataset.n_classes
        assert (self.test_dataset.n_classes == self.train_dataset.n_classes)

    def reset_test_loader(self, batch_size: int, shuffle_switch: bool) -> None:
        """ Resets/Initializes test data loader. 

        Args:
            batch_size: The size of batch.
            shuffle_switch: Whether to shuffle the data.
        """
        if self.test_loader is not None:
            del self.test_loader
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle_switch,
            pin_memory=self.use_cuda, num_workers=self.num_workers)

    def reset_train_loader(self, batch_size: int) -> None:
        """ Resets/Initializes train data loader. 

        Args:
            batch_size: The size of batch.
        """
        if self.train_loader is not None:
            del self.train_loader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=self.use_cuda, num_workers=self.num_workers)


def build_benchmark(config: ConfigBenchmark) -> IBenchmark:
    """ Benchmark builder method.  Creates corresponding benchmark according to 
    benchmark configuration. If needed, estimates normalization parameters before
    creating benchmark.

    Args:
        config: Benchmark configuration.

    Returns:
        Ready-to-use benchmark supporting IBenchmark interface.
    """
    DataSetTypeClass: Optional[Type[DynamicDataset]] = None
    BenchmarkTypeClass: Optional[Type[IBenchmark]] = None
    path_to_img_root: Optional[str] = None
    path_to_annotations: Optional[str] = None

    if config.benchmark == BenchmarkName.CHINESE:
        DataSetTypeClass = ChineseDataset
        BenchmarkTypeClass = ChineseTraffic
        path_to_img_root = FileFolderPaths.CHINESE_TRAIN_ROOT
        path_to_annotations = FileFolderPaths.CHINESE_TRAIN_ANNOTATIONS

    elif config.benchmark == BenchmarkName.GERMANY:
        DataSetTypeClass = GermanDataset
        BenchmarkTypeClass = GermanTraffic
        path_to_img_root = FileFolderPaths.GERMAN_TRAIN_ROOT
        path_to_annotations = FileFolderPaths.GERMAN_TRAIN_ANNOTATIONS

    elif config.benchmark == BenchmarkName.BELGIUM:
        DataSetTypeClass = BelgiumDataset
        BenchmarkTypeClass = BelgiumTraffic
        path_to_img_root = FileFolderPaths.BELGIUM_TRAIN_ROOT
        path_to_annotations = FileFolderPaths.BELGIUM_TRAIN_ANNOTATIONS

    elif config.benchmark == BenchmarkName.RUSSIAN:
        DataSetTypeClass = RussianDataset
        BenchmarkTypeClass = RussianTraffic
        path_to_img_root = FileFolderPaths.RUSSIAN_TRAIN_ROOT
        path_to_annotations = FileFolderPaths.RUSSIAN_TRAIN_ANNOTATIONS
    else:
        raise ValueError("Unknown benchmark name.")

    if config.estimate_normalization:
        mean_normalize: TypingFloatTensor; std_normalize: TypingFloatTensor
        mean_normalize, std_normalize = estimate_normalization(
            path_to_img_root, path_to_annotations,
            DataSetTypeClass, config.image_color == ColorSchema.GRAYSCALE,
            config.image_size, config.n_point_to_estimate
        )
        config.mean_normalize = mean_normalize.numpy()
        config.std_normalize = std_normalize.numpy()

    return BenchmarkTypeClass(config)


class ChineseTraffic(IBenchmark):
    """ Chinese Traffic Benchmark.  """

    def __init__(self, config: ConfigBenchmark) -> None:
        """ Initialization method. Sets datasets and initializes loaders. 
        
        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)

        self.train_dataset: ChineseDataset = ChineseDataset(
            path_to_img_dir=FileFolderPaths.CHINESE_TRAIN_ROOT,
            path_to_annotations=FileFolderPaths.CHINESE_TRAIN_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=self.random_transform
        )
        self.test_dataset: ChineseDataset = ChineseDataset(
            path_to_img_dir=FileFolderPaths.CHINESE_TEST_ROOT,
            path_to_annotations=FileFolderPaths.CHINESE_TEST_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=None
        )
        self.init_loaders(config)


class GermanTraffic(IBenchmark):
    """ German Traffic Benchmark. """

    def __init__(self, config: ConfigBenchmark) -> None:
        """ Initialization method. Sets datasets and initializes loaders. 
        
        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.train_dataset: GermanDataset = GermanDataset(
            path_to_img_dir=FileFolderPaths.GERMAN_TRAIN_ROOT,
            path_to_annotations=FileFolderPaths.GERMAN_TRAIN_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=self.random_transform
        )
        self.test_dataset: GermanDataset = GermanDataset(
            path_to_img_dir=FileFolderPaths.GERMAN_TEST_ROOT,
            path_to_annotations=FileFolderPaths.GERMAN_TEST_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=None
        )
        self.init_loaders(config)


class BelgiumTraffic(IBenchmark):
    """ Belgium Traffic Benchmark. """

    def __init__(self, config: ConfigBenchmark) -> None:
        """ Initialization method. Sets datasets and initializes loaders. 
        
        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.train_dataset: BelgiumDataset = BelgiumDataset(
            path_to_img_dir=FileFolderPaths.BELGIUM_TRAIN_ROOT,
            path_to_annotations=FileFolderPaths.BELGIUM_TRAIN_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=self.random_transform
        )
        self.test_dataset: BelgiumDataset = BelgiumDataset(
            path_to_img_dir=FileFolderPaths.BELGIUM_TEST_ROOT,
            path_to_annotations=FileFolderPaths.BELGIUM_TEST_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=None
        )
        self.init_loaders(config)


class RussianTraffic(IBenchmark):
    """ Russian Traffic Benchmark. """
    def __init__(self, config: ConfigBenchmark) -> None:
        """ Initialization method. Sets datasets and initializes loaders. 
        
        Args:
            config: Benchmark configuration.
        """
        super().__init__(config)
        self.train_dataset: RussianDataset = RussianDataset(
            path_to_img_dir=FileFolderPaths.RUSSIAN_TRAIN_ROOT,
            path_to_annotations=FileFolderPaths.RUSSIAN_TRAIN_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=self.random_transform
        )
        self.test_dataset: RussianDataset = RussianDataset(
            path_to_img_dir=FileFolderPaths.RUSSIAN_TEST_ROOT,
            path_to_annotations=FileFolderPaths.RUSSIAN_TEST_ANNOTATIONS,
            static_transform=self.static_transform,
            random_transform=None
        )
        self.init_loaders(config)

