from typing import Dict
from typing import List
from typing import Any
from typing import TextIO
from collections import OrderedDict
from collections import defaultdict
import logging
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


def fill_stats(json_stats: Dict[str, Any], table_stats: Dict[str, Any],
               benchmark_name: str, accuracy: float, epoch: int) -> None:
    benchmark_stats: Dict[str, Any] = dict()
    json_stats[benchmark_name] = benchmark_stats
    table_stats[StatsTableKeys.DATASET].append(benchmark_name)
    benchmark_stats[StatsTableKeys.ACCURACY] = accuracy
    table_stats[StatsTableKeys.ACCURACY].append(accuracy)
    benchmark_stats[StatsTableKeys.EPOCH] = epoch
    table_stats[StatsTableKeys.EPOCH].append(epoch)


def perform_test_launches(json_stats: Dict[str, Any],
                          table_stats: Dict[str, Any]) -> None:
    logging.info("Test launch.")
    config: SetupConfig = SetupConfig()
    config.training_config.n_epochs = 2
    config.training_config.batch_size = 4
    config.training_config.n_visualize = 4
    config.benchmark_config.batch_size = 4
    accuracy: float; epoch: int
    accuracy, epoch = do_training(config)
    fill_stats(json_stats, table_stats,
               'test-benchmark', accuracy, epoch)
    logging.info("Finished test launch.")


def perform_chinese_launches(json_stats: Dict[str, Any],
                             table_stats: Dict[str, Any]) -> None:
    logging.info("Performing Chinese Launches.")
    config: SetupConfig = SetupConfig()
    config.training_config.n_epochs = 60
    accuracy: float; epoch: int
    accuracy, epoch = do_training(config)
    fill_stats(json_stats, table_stats,
               BenchmarkName.CHINESE, accuracy, epoch)
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

    benchmark_config.benchmark = BenchmarkName.GERMANY
    training_config.n_classes = 43
    training_config.n_epochs = 30
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

    accuracy: float; epoch: int
    accuracy, epoch = do_training(config)
    fill_stats(json_stats, table_stats,
               BenchmarkName.GERMANY, accuracy, epoch)
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

    accuracy: float; epoch: int
    accuracy, epoch = do_training(config)
    fill_stats(json_stats, table_stats,
               BenchmarkName.BELGIUM, accuracy, epoch)
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

    accuracy: float; epoch: int
    accuracy, epoch = do_training(config)
    fill_stats(json_stats, table_stats,
               BenchmarkName.RUSSIAN, accuracy, epoch)
    logging.info("Done with Russian Launches.")


def perform_launches() -> None:
    json_stats: Dict[str, Any] = OrderedDict()
    table_stats: Dict[str, List[Any]] = defaultdict(list)

    perform_russian_launches(json_stats, table_stats)
    perform_chinese_launches(json_stats, table_stats)
    perform_belgium_launches(json_stats, table_stats)
    perform_german_launches(json_stats, table_stats)

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
