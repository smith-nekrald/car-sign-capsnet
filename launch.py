import logging

from config import SetupConfig
from keys import BenchmarkName
from train import do_training


def perform_chinese_launches():
    logging.info("Performing Chinese Launches.")
    config: SetupConfig = SetupConfig()
    do_training(config)
    logging.info("Done with Chinese Launches.")


def perform_german_launches():
    logging.info("Performing German Launches.")
    config: SetupConfig = SetupConfig()
    config.benchmark = BenchmarkName.GERMANY
    config.estimate_normalization = True
    config.mean_normalize = None
    config.std_normalize = None
    do_training(config)
    logging.info("Done with German Launches.")


def perform_launches():
    perform_german_launches()
    perform_chinese_launches()
