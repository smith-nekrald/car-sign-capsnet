from typing import Dict
from typing import Any
from collections import OrderedDict
import logging
import json

from config import SetupConfig
from keys import BenchmarkName
from train import do_training


def perform_chinese_launches(stats: Dict[str, Any]):
    logging.info("Performing Chinese Launches.")
    config: SetupConfig = SetupConfig()
    config.training_config.n_epochs = 60
    accuracy, epoch = do_training(config)
    stats['chinese_accuracy'] = accuracy
    stats['chinese_epoch'] = epoch
    logging.info("Done with Chinese Launches.")


def perform_german_launches(stats: Dict[str, Any]):
    logging.info("Performing German Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config = config.network_config.reconstruction_config
    recognition_config = config.network_config.recognition_config
    primary_config = config.network_config.primary_config
    agreement_config = config.network_config.agreement_config
    training_config = config.training_config
    benchmark_config = config.benchmark_config
    conv_config = config.network_config.conv_config

    benchmark_config.benchmark = BenchmarkName.GERMANY

    training_config.n_classes = 43
    training_config.n_epochs = 5

    benchmark_config.estimate_normalization = True
    benchmark_config.mean_normalize = None
    benchmark_config.std_normalize = None

    conv_config.use_batch_norm = False
    primary_config.use_dropout = False

    agreement_config.num_output_caps = 43

    recognition_config.num_output_caps = 43
    recognition_config.use_dropout = False

    reconstruction_config.linear_input_dim = 43 * 16
    reconstruction_config.num_classes = 43

    accuracy, epoch = do_training(config)
    stats['german_accuracy'] = accuracy
    stats['german_epoch'] = epoch
    logging.info("Done with German Launches.")


def perform_belgium_launches(stats: Dict[str, Any]) -> None:
    logging.info("Performing Belgium Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config = config.network_config.reconstruction_config
    recognition_config = config.network_config.recognition_config
    primary_config = config.network_config.primary_config
    agreement_config = config.network_config.agreement_config
    training_config = config.training_config
    benchmark_config = config.benchmark_config
    conv_config = config.network_config.conv_config

    benchmark_config.benchmark = BenchmarkName.BELGIUM

    training_config.n_classes = 62
    training_config.n_epochs = 5

    benchmark_config.estimate_normalization = True
    benchmark_config.mean_normalize = None
    benchmark_config.std_normalize = None

    conv_config.use_batch_norm = False
    primary_config.use_dropout = False

    agreement_config.num_output_caps = 62

    recognition_config.num_output_caps = 62
    recognition_config.use_dropout = False

    reconstruction_config.linear_input_dim = 62 * 16
    reconstruction_config.num_classes = 62

    accuracy, epoch = do_training(config)
    stats['belgium_accuracy'] = accuracy
    stats['belgium_epoch'] = epoch
    logging.info("Done with Belgium Launches.")


def perform_russian_launches(stats: Dict[str, Any]) -> None:
    logging.info("Performing Russian Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config = config.network_config.reconstruction_config
    recognition_config = config.network_config.recognition_config
    primary_config = config.network_config.primary_config
    agreement_config = config.network_config.agreement_config
    training_config = config.training_config
    benchmark_config = config.benchmark_config
    conv_config = config.network_config.conv_config

    benchmark_config.benchmark = BenchmarkName.RUSSIAN

    training_config.n_classes = 67
    training_config.n_epochs = 5

    benchmark_config.estimate_normalization = True
    benchmark_config.mean_normalize = None
    benchmark_config.std_normalize = None

    conv_config.use_batch_norm = False
    primary_config.use_dropout = False

    agreement_config.num_output_caps = 67

    recognition_config.num_output_caps = 67
    recognition_config.use_dropout = False

    reconstruction_config.linear_input_dim = 67 * 16
    reconstruction_config.num_classes = 67

    accuracy, epoch = do_training(config)
    stats['russian_accuracy'] = accuracy
    stats['russian_epoch'] = epoch
    logging.info("Done with Russian Launches.")


def perform_launches():
    stats = OrderedDict()
    perform_russian_launches(stats)
    # perform_belgium_launches(stats)
    # perform_german_launches(stats)
    # perform_chinese_launches(stats)
    with open("traindir/stats.json", "w") as file_stats:
        json.dump(stats, file_stats)
