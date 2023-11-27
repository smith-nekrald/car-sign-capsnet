#!/usr/bin/env python3
""" Entry point. Parses CLI arguments, setups logging and calls API to perform experiments. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

import logging
import os
import uuid
import argparse

import random
import numpy as np
import torch

from keys import NameKeys
from launch import perform_launches
from keys import BenchmarkName


def parse_arguments() -> argparse.Namespace:
    """ Defines and parses CLI arguments. 

    Returns:
        Namespace with parsed CLI arguments. 
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=BenchmarkName.CHOICE_OPTIONS, help="Benchmark for experiment.")
    return parser.parse_args()


def setup_logging() -> None:
    """ Setups logging format. """
    FORMAT: str = '%(asctime)s %(levelname)s %(funcName)s %(lineno)d : %(message)s'
    logging.basicConfig(format=FORMAT)
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler: logging.FileHandler = logging.FileHandler(
        os.path.join(NameKeys.TRAINDIR, f'main-{uuid.uuid4()}.log'))
    file_handler.setLevel(logging.INFO)
    formatter: logging.Formatter = logging.Formatter(FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


if __name__ == '__main__':
    args: argparse.Namespace = parse_arguments()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(NameKeys.TRAINDIR):
        os.makedirs(NameKeys.TRAINDIR)

    setup_logging()
    perform_launches(args)

