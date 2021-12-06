from typing import List
from typing import Optional

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms as T

from dataset import ChineseDataset
from dataset import TransformType
from dataset import DynamicDataset
from config import SetupConfig
from keys import ColorSchema
from keys import BenchmarkName


class IBenchmark:
    def __init__(self, config: SetupConfig) -> None:
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

        entry_list: List[TransformType]
        for entry_list in [static_list, random_list]:
            if config.image_color == ColorSchema.GRAYSCALE:
                entry_list.append(T.Grayscale())
            entry_list.append(T.Normalize(config.mean_normalize, config.std_normalize))

        self.static_transform: TransformType = T.Compose(static_list)
        self.random_transform: TransformType = T.Compose(random_list)

        if not config.use_augmentation:
            self.random_transform = self.static_transform

        self.train_dataset: Optional[DynamicDataset] = None
        self.test_dataset: Optional[DynamicDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def set_random_mode(self) -> None:
        if self.train_dataset is not None:
            self.train_dataset.random_mode()

    def set_static_mode(self) -> None:
        if self.train_dataset is not None:
            self.train_dataset.static_mode()


def build_benchmark(config: SetupConfig) -> IBenchmark:
    if config.benchmark == BenchmarkName.CHINESE:
        return ChineseTraffic(config)
    else:
        raise ValueError("Unknown benchmark name.")


class ChineseTraffic(IBenchmark):
    def __init__(self, config: SetupConfig) -> None:
        super().__init__(config)

        self.train_dataset: ChineseDataset = ChineseDataset(
            path_to_img_dir='../China-TSRD/TSRD-Train-Images/',
            path_to_annotations='../China-TSRD/TSRD-Train-Annotation/TsignRecgTrain4170Annotation.txt',
            static_transform=self.static_transform,
            random_transform=self.random_transform)
        self.train_dataset.random_mode()

        self.test_dataset: ChineseDataset = ChineseDataset(
            path_to_img_dir='../China-TSRD/TSRD-Test-Images/',
            path_to_annotations='../China-TSRD/TSRD-Test-Annotation/TsignRecgTest1994Annotation.txt',
            static_transform=self.static_transform,
            random_transform=self.static_transform)
        self.test_dataset.static_mode()

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader: DataLoader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, shuffle=True)

        self.n_classes: int = self.train_dataset.n_classes
        assert (self.test_dataset.n_classes == self.train_dataset.n_classes)
