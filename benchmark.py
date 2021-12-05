from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChineseDataset


class IBenchmark:
    def __init__(self, batch_size: int) -> None:
        pass


class ChineseTraffic(IBenchmark):
    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size)
        dataset_transform: Union[nn.Module, transforms.Compose] = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(28, 28)),
            transforms.Grayscale(), transforms.Normalize(0.4255, 0.2235)])
        # Transforms Normalize
        train_dataset: ChineseDataset = ChineseDataset(path_to_img_dir='../China-TSRD/TSRD-Train-Images/',
                                                       path_to_annotations='../China-TSRD/TSRD-Train-Annotation/TsignRecgTrain4170Annotation.txt',
                                                       transform=dataset_transform)
        test_dataset: ChineseDataset = ChineseDataset(path_to_img_dir='../China-TSRD/TSRD-Test-Images/',
                                                      path_to_annotations='../China-TSRD/TSRD-Test-Annotation/TsignRecgTest1994Annotation.txt',
                                                      transform=dataset_transform)
        self.train_loader: DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader: DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.n_classes = train_dataset.n_classes
        assert (test_dataset.n_classes == train_dataset.n_classes)
