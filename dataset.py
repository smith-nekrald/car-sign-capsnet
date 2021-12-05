from typing import Optional
from typing import List
from typing import Tuple
from typing import Union

import os

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ChineseDataset(Dataset):
    def __init__(self, path_to_img_dir: str, path_to_annotations: str, transform=Optional[nn.Module]) -> None:
        column_names: List[str] = ['file_name', 'img_width', 'img_height', 'sign_x_top', 'sign_y_top',
                                   'sign_x_bottom', 'sign_y_bottom', 'label']
        self.transform: Optional[nn.Module] = transform
        self.annotation_df: pd.DataFrame = pd.read_csv(path_to_annotations, sep=';', header=None,
                                                       index_col=False, names=column_names)
        self.root_dir: str = path_to_img_dir

        self.idx2class: List[int] = list(self.annotation_df['label'])
        self.idx2name: List[str] = list(self.annotation_df['file_name'])

        self.n_classes: int = max(self.idx2class) + 1

    def __len__(self) -> int:
        return len(self.idx2class)

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