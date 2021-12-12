from typing import Union

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import shutil

import torch
import torchvision.transforms as T
import torch.nn as nn
from skimage.color import gray2rgb
from skimage.color import label2rgb
from lime import lime_image
from lime.explanation import Explanation
from lime.wrappers.scikit_image import SegmentationAlgorithm

from benchmark import IBenchmark
from keys import NameKeys

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


def check_and_make_folder(path_to_dir: str) -> None:
    if not os.path.isdir(path_to_dir):
        if os.path.exists(path_to_dir):
            shutil.rmtree(path_to_dir)
        os.makedirs(path_to_dir)


class CapsNetCallable:
    def __init__(self, model: nn.Module,
                 normalize_transform: nn.Module, use_cuda: bool):
        self.use_cuda: bool = use_cuda
        self.capsnet: nn.Module = model
        self.normalize_transform: nn.Module = normalize_transform

    def __call__(self, images: np.array) -> np.array:
        images: TypingFloatTensor = torch.Tensor(images).permute(0, 3, 1, 2)
        batch: TypingFloatTensor = self.normalize_transform(T.Grayscale()(images)).float()

        if self.use_cuda:
            batch = batch.cuda()
        class_probas: TypingFloatTensor
        _, _, _, class_probas = self.capsnet(batch)
        return class_probas.detach().cpu().numpy()


def explain_lime(benchmark: IBenchmark, model: nn.Module,
                 use_cuda: bool, explanation_dir: str) -> None:
    logging.info("Started LIME explanation.")
    check_and_make_folder(explanation_dir)

    targets: TypingIntTensor; images: TypingFloatTensor
    images, targets = next(iter(benchmark.test_loader))

    idx_image: int
    for idx_image in range(images.shape[0]):
        image_tensor: TypingFloatTensor = benchmark.recover_normalize(
            images[idx_image]).squeeze(0)
        image_rgb: np.array = gray2rgb(image_tensor.numpy())
        image_f64 = np.array(image_rgb, dtype=np.float64)
        plt.imsave(os.path.join(
            explanation_dir, NameKeys.EXPLAIN_SRC_PNG.format(idx_image)), image_f64)

        explainer: lime_image.LimeImageExplainer = lime_image.LimeImageExplainer()
        predictor_fn: CapsNetCallable = CapsNetCallable(
            model, benchmark.normalize_transform, use_cuda)
        segmenter: SegmentationAlgorithm = SegmentationAlgorithm(
            'quickshift', kernel_size=1, max_dist=200, ratio=0.2)

        explanation: Explanation = explainer.explain_instance(
            image_f64, predictor_fn,
            top_labels=40, hide_color=0, batch_size=16,
            num_samples=30000, segmentation_fn=segmenter)

        temp_positive: np.array; mask_positive: np.array
        temp_positive, mask_positive = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True,
            num_features=10, hide_rest=False, min_weight=0.01)
        plt.imsave(os.path.join(
            explanation_dir, NameKeys.EXPLAIN_POSITIVE_PNG.format(idx_image)),
            label2rgb(mask_positive, temp_positive, bg_label=0))

        temp_all: np.array; mask_all: np.array
        temp_all, mask_all = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=False,
            num_features=10, hide_rest=False, min_weight=0.01)
        plt.imsave(os.path.join(
            explanation_dir, NameKeys.EXPLAIN_ALL_PNG.format(idx_image)),
                   label2rgb(3 - mask_all, temp_all, bg_label=0))

    logging.info("Done with LIME explanation.")
