from typing import Dict
from typing import List
from typing import Any
from typing import TextIO
from collections import OrderedDict
from collections import defaultdict

import logging
import argparse
import json
import os

import pandas as pd

from config import SetupConfig
from config import ConfigSquash
from config import ConfigRecognition
from config import ConfigReconstruction
from config import ConfigAgreement
from config import ConfigTraining
from config import ConfigBenchmark
from config import ConfigPrimary
from config import ConfigConv
from train import do_training
from keys import BenchmarkName
from keys import StatsTableKeys
from keys import NameKeys


def fill_stats(json_stats: Dict[str, Any], table_stats: Dict[str, Any], epoch_id: int,
               test_accuracy: float, test_loss: float,
               train_accuracy: float, train_loss: float,
               benchmark_name: str) -> None:
    benchmark_stats: Dict[str, Any] = dict()
    json_stats[benchmark_name] = benchmark_stats
    table_stats[StatsTableKeys.DATASET].append(benchmark_name)
    benchmark_stats[StatsTableKeys.TEST_ACCURACY] = test_accuracy * 100.
    table_stats[StatsTableKeys.TEST_ACCURACY].append(test_accuracy * 100.)
    benchmark_stats[StatsTableKeys.TRAIN_ACCURACY] = train_accuracy * 100.
    table_stats[StatsTableKeys.TRAIN_ACCURACY].append(train_accuracy * 100.)
    benchmark_stats[StatsTableKeys.TEST_LOSS] = test_loss
    table_stats[StatsTableKeys.TEST_LOSS].append(test_loss)
    benchmark_stats[StatsTableKeys.TRAIN_LOSS] = train_loss
    table_stats[StatsTableKeys.TRAIN_LOSS].append(train_loss)
    benchmark_stats[StatsTableKeys.EPOCH_ID] = epoch_id
    table_stats[StatsTableKeys.EPOCH_ID].append(epoch_id)


def perform_test_launches(json_stats: Dict[str, Any],
                          table_stats: Dict[str, Any]) -> None:
    logging.info("Test launch.")
    config: SetupConfig = SetupConfig()
    config.training_config.n_epochs = 1
    config.training_config.batch_size = 16
    config.training_config.n_visualize = 6
    config.training_config.use_lime = False
    config.benchmark_config.batch_size = 16

    epoch_id: int
    test_accuracy: float; test_loss: float
    train_accuracy: float; train_loss: float
    (epoch_id, test_accuracy, test_loss,
     train_accuracy, train_loss) = do_training(config)
    fill_stats(json_stats, table_stats, epoch_id,
               test_accuracy, test_loss,
               train_accuracy, train_loss,
               'Test Benchmark')

    logging.info("Finished test launch.")


def perform_chinese_launches(json_stats: Dict[str, Any],
                             table_stats: Dict[str, Any]) -> None:
    logging.info("Performing Chinese Launches.")
    config: SetupConfig = SetupConfig()
    config.training_config.n_epochs = 100
    config.benchmark_config.augment_proba = 0.6
    config.network_config.primary_config.dropout_proba = 0.5
    config.network_config.recognition_config.dropout_proba = 0.5

    epoch_id: int
    test_accuracy: float; test_loss: float
    train_accuracy: float; train_loss: float
    (epoch_id, test_accuracy, test_loss,
     train_accuracy, train_loss) = do_training(config)
    fill_stats(json_stats, table_stats, epoch_id,
               test_accuracy, test_loss,
               train_accuracy, train_loss,
               BenchmarkName.CHINESE)

    logging.info("Done with Chinese Launches.")


def perform_german_launches(json_stats: Dict[str, Any],
                            table_stats: Dict[str, Any]) -> None:
    logging.info("Performing German Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config: ConfigReconstruction = config.network_config.reconstruction_config
    recognition_config: ConfigRecognition = config.network_config.recognition_config
    primary_config: ConfigPrimary = config.network_config.primary_config
    agreement_config: ConfigAgreement = config.network_config.agreement_config
    training_config: ConfigTraining = config.training_config
    benchmark_config: ConfigBenchmark = config.benchmark_config
    conv_config: ConfigConv = config.network_config.conv_config
    squash_config: ConfigSquash = config.network_config.squash_config

    benchmark_config.benchmark = BenchmarkName.GERMANY
    training_config.n_classes = 43
    training_config.n_epochs = 30
    benchmark_config.estimate_normalization = True
    benchmark_config.mean_normalize = None
    benchmark_config.std_normalize = None
    conv_config.use_batch_norm = True
    primary_config.use_dropout = False
    agreement_config.num_output_caps = 43
    recognition_config.num_output_caps = 43
    recognition_config.use_dropout = False
    reconstruction_config.linear_input_dim = 43 * 16
    reconstruction_config.num_classes = 43
    squash_config.eps_norm = 1e-4
    squash_config.eps_sqrt = 1e-4
    squash_config.eps_denom = 1e-4
    squash_config.eps_input = 1e-4

    epoch_id: int
    test_accuracy: float; test_loss: float
    train_accuracy: float; train_loss: float
    (epoch_id, test_accuracy, test_loss,
     train_accuracy, train_loss) = do_training(config)
    fill_stats(json_stats, table_stats, epoch_id,
               test_accuracy, test_loss,
               train_accuracy, train_loss,
               BenchmarkName.GERMANY)
    logging.info("Done with German Launches.")


def perform_belgium_launches(json_stats: Dict[str, Any],
                             table_stats: Dict[str, Any]) -> None:
    logging.info("Performing Belgium Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config: ConfigReconstruction = config.network_config.reconstruction_config
    recognition_config: ConfigRecognition = config.network_config.recognition_config
    primary_config: ConfigPrimary = config.network_config.primary_config
    agreement_config: ConfigAgreement = config.network_config.agreement_config
    training_config: ConfigTraining = config.training_config
    benchmark_config: ConfigBenchmark = config.benchmark_config
    conv_config: ConfigConv = config.network_config.conv_config

    benchmark_config.benchmark = BenchmarkName.BELGIUM
    training_config.n_classes = 62
    training_config.n_epochs = 30
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

    epoch_id: int
    test_accuracy: float; test_loss: float
    train_accuracy: float; train_loss: float
    (epoch_id, test_accuracy, test_loss,
     train_accuracy, train_loss) = do_training(config)
    fill_stats(json_stats, table_stats, epoch_id,
               test_accuracy, test_loss,
               train_accuracy, train_loss,
               BenchmarkName.BELGIUM)

    logging.info("Done with Belgium Launches.")


def perform_russian_launches(json_stats: Dict[str, Any],
                             table_stats: Dict[str, Any]) -> None:
    logging.info("Performing Russian Launches.")
    config: SetupConfig = SetupConfig()

    reconstruction_config: ConfigReconstruction = config.network_config.reconstruction_config
    recognition_config: ConfigRecognition = config.network_config.recognition_config
    primary_config: ConfigPrimary = config.network_config.primary_config
    agreement_config: ConfigAgreement = config.network_config.agreement_config
    training_config: ConfigTraining = config.training_config
    benchmark_config: ConfigBenchmark = config.benchmark_config
    conv_config: ConfigConv = config.network_config.conv_config
    squash_config: ConfigSquash = config.network_config.squash_config

    benchmark_config.benchmark = BenchmarkName.RUSSIAN
    training_config.n_classes = 67
    training_config.n_epochs = 30
    training_config.use_clipping = True
    training_config.clipping_threshold = 100.
    squash_config.eps_norm = 1e-4
    squash_config.eps_sqrt = 1e-4
    squash_config.eps_denom = 1e-4
    squash_config.eps_input = 1e-4
    benchmark_config.estimate_normalization = True
    benchmark_config.mean_normalize = None
    benchmark_config.std_normalize = None
    conv_config.use_batch_norm = True
    primary_config.use_dropout = False
    primary_config.use_nan_gradient_hook = True
    agreement_config.num_output_caps = 67
    recognition_config.num_output_caps = 67
    recognition_config.use_dropout = False
    recognition_config.use_nan_gradient_hook = True
    reconstruction_config.linear_input_dim = 67 * 16
    reconstruction_config.num_classes = 67

    epoch_id: int
    test_accuracy: float; test_loss: float
    train_accuracy: float; train_loss: float
    (epoch_id, test_accuracy, test_loss,
     train_accuracy, train_loss) = do_training(config)
    fill_stats(json_stats, table_stats, epoch_id,
               test_accuracy, test_loss,
               train_accuracy, train_loss,
               BenchmarkName.RUSSIAN)

    logging.info("Done with Russian Launches.")


def output_stats(json_stats, table_stats) -> None:
    file_stats: TextIO
    path_to_stats_json: str = os.path.join(
        NameKeys.TRAINDIR, NameKeys.STATS_JSON)
    with open(path_to_stats_json, "w") as file_stats:
        json.dump(json_stats, file_stats)

    stats_df: pd.DataFrame = pd.DataFrame.from_dict(table_stats)
    path_to_stats_xlsx: str = os.path.join(
        NameKeys.TRAINDIR, NameKeys.STATS_XLSX)
    path_to_stats_tex: str = os.path.join(
        NameKeys.TRAINDIR, NameKeys.STATS_TEX)
    stats_df.to_excel(path_to_stats_xlsx, float_format="%.2f")
    stats_df.to_latex(path_to_stats_tex, float_format="%.2f")


def perform_launches(args: argparse.Namespace) -> None:
    """ Performs experimetns and collects summary statistics. 

    Args:
        args: Namespace with CLI arguments.
    """
    json_stats: Dict[str, Any] = OrderedDict()
    table_stats: Dict[str, List[Any]] = defaultdict(list)

    if args.benchmark in [BenchmarkName.ALL, BenchmarkName.GERMANY]:
        perform_german_launches(json_stats, table_stats)

    if args.benchmark in [BenchmarkName.ALL, BenchmarkName.CHINESE]:
        perform_chinese_launches(json_stats, table_stats)

    if args.benchmark in [BenchmarkName.ALL, BenchmarkName.RUSSIAN]:
        perform_russian_launches(json_stats, table_stats)

    if args.benchmark in [BenchmarkName.ALL, BenchmarkName.BELGIUM]:
        perform_belgium_launches(json_stats, table_stats)

    output_stats(json_stats, table_stats)
