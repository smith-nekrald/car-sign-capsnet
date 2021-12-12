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


def perform_launches():
    stats = OrderedDict()
    # perform_german_launches(stats)
    perform_chinese_launches(stats)
    with open("traindir/stats.json") as file_stats:
        json.dump(stats, file_stats)
