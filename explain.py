import os
import logging

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

import shutil
from lime import lime_image
import torch
import torchvision.transforms as T
import torch.nn as nn
from skimage.color import gray2rgb

from benchmark import IBenchmark


def check_and_make_folder(path_to_dir: str) -> None:
    if not os.path.isdir(path_to_dir):
        if os.path.exists(path_to_dir):
            shutil.rmtree(path_to_dir)
        os.makedirs(path_to_dir)


class CapsNetCallable:
    def __init__(self, model, normalize_transform, use_cuda):
        self.use_cuda = use_cuda
        self.capsnet = model
        self.normalize_transform = normalize_transform

    def __call__(self, images):
        images = torch.Tensor(images).permute(0, 3, 1, 2)
        batch = self.normalize_transform(T.Grayscale()(images)).float()

        if self.use_cuda:
            batch = batch.cuda()
        _, _, _, class_probas = self.capsnet(batch)
        return class_probas.detach().cpu().numpy()


def explain_lime(benchmark: IBenchmark, model: nn.Module,
                 use_cuda: bool, explanation_dir: str) -> None:
    logging.info("Started LIME explanation.")
    check_and_make_folder(explanation_dir)

    images, targets = next(iter(benchmark.test_loader))
    image = benchmark.recover_normalize(images[0]).squeeze(0)
    image = gray2rgb(image.numpy())
    image = np.array(image, dtype=np.float64)
    plt.imsave(os.path.join(explanation_dir, "image-0.png"), image)

    explainer = lime_image.LimeImageExplainer()
    predictor_fn = CapsNetCallable(model, benchmark.normalize_transform, use_cuda)
    explanation = explainer.explain_instance(
        image,
        predictor_fn,
        top_labels=5,
        hide_color=0,
        batch_size=16,
        num_samples=1000)  # number of images that will be sent to classification function

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True,
        num_features=5, hide_rest=False)
    img_boundary = mark_boundaries(temp / 255.0, mask)
    plt.imsave(os.path.join(explanation_dir, "image-1.png"), img_boundary)

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False,
        num_features=10, hide_rest=False)
    img_boundary_regions = mark_boundaries(temp / 255.0, mask)
    plt.imsave(os.path.join(explanation_dir, "image-2.png"), img_boundary_regions)
    logging.info("Done with LIME explanation.")
