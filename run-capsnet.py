#!/usr/bin/env python3
import logging
import os
import uuid

import random
import numpy as np
import torch

from keys import NameKeys
from launch import perform_launches


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(NameKeys.TRAINDIR):
        os.makedirs(NameKeys.TRAINDIR)

    FORMAT = '%(asctime)s %(levelname)s %(funcName)s %(lineno)d : %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(NameKeys.TRAINDIR, f'main-{uuid.uuid4()}.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    perform_launches()

