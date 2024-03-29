""" Implements CapsNet training logic. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import Tuple
from typing import Optional
from typing import Union

import logging
import os
import gc

import numpy as np

import torch
import torch.nn as nn
import torch.backends
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from capsnet.network import CapsNet
from capsnet.config import SetupConfig
from capsnet.config import ConfigBenchmark
from capsnet.config import ConfigTraining
from capsnet.benchmark import build_benchmark
from capsnet.benchmark import IBenchmark
from capsnet.visualize import plot_images_separately
from capsnet.explain import explain_lime
from capsnet.explain import check_and_make_folder
from capsnet.keys import NameKeys
from capsnet.checkpoint import load_checkpoint
from capsnet.checkpoint import save_checkpoint

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]
ProcessEpochReturnTyping = Union[Tuple[float, float], Tuple[
    TypingFloatTensor, TypingFloatTensor, float, float]]


def process_epoch(epoch_idx: int, benchmark: IBenchmark,
                  capsule_net: nn.Module, optimizer: Optional[Optimizer],
                  use_cuda: bool, n_classes: int,
                  writer: SummaryWriter, train_mode: bool, log_frequency: int,
                  use_clipping: bool, clip_value: Optional[float]
                  ) -> ProcessEpochReturnTyping:
    """ Broad method responsible for epoch training or epoch testing, depending on the
    input parameters.

    Args:
        epoch_idx: Epoch index.
        benchmark: Benchmark of the current experiment.
        capsule_net: The capsule network for training or evaluation.
        optimizer: The optimizer for updating network weights, if relevant.
        use_cuda: Whether to use CUDA.
        n_classes: Number of classes.
        writer: Writer to TensorBoard.
        train_mode: If true, train logic is applied. If false, testing logic is applied.
        log_frequency: Frequency of logging.
        use_clipping: Whether to use gradient clipping.
        clip_value: The threshold for gradient clipping.


    Returns:
        A tuple with two or four elements, depending on train_mode: 
        If training, returns a tuple with two elements. The first is the 
        average epoch training loss.  The second is training epoch acuracy.
        If testing, returns a tuple with four elements. The first is some 
        testing batch with data. The second is the reconstructions on that batch. 
        The third is the average test loss value on that epoch. The fourth is the 
        test accuracy on that epoch. 
    """
    data_loader: DataLoader; mode_string: str
    if train_mode:
        capsule_net.train()
        data_loader = benchmark.train_loader
        mode_string = NameKeys.TRAIN_MODE_STRING
    else:
        capsule_net.eval()
        data_loader = benchmark.test_loader
        mode_string = NameKeys.TEST_MODE_STRING

    logging.info(f"{mode_string.capitalize()}ing.")
    epoch_loss: float = 0

    accuracy_match_count: int = 0
    sample_count: int = 0

    running_loss: float = 0.
    running_match_count: int = 0
    running_sample_count: int = 0

    data: Optional[TypingFloatTensor]; reconstructions: Optional[TypingFloatTensor]
    data, reconstructions = None, None
    batch_id: int; labels: TypingIntTensor
    for batch_id, (data, labels) in enumerate(data_loader):
        if batch_id % log_frequency == 0:
            logging.info(
                f"{mode_string.capitalize()} batch {batch_id} out of {len(data_loader)}")
        target: TypingFloatTensor
        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=labels)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        if train_mode:
            optimizer.zero_grad()

        output: TypingFloatTensor; reconstructions: TypingFloatTensor
        select_mask: TypingFloatTensor; class_probas: TypingFloatTensor
        output, reconstructions, select_mask, class_probas = capsule_net(data)
        batch_loss: TypingFloatTensor
        batch_loss = capsule_net.loss(data, output, target, reconstructions)

        if train_mode:
            batch_loss.backward()
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(
                    capsule_net.parameters(), clip_value, norm_type=2.0)
            optimizer.step()

        epoch_loss += batch_loss.data.item() * data.shape[0]

        batch_match_count: int = sum(np.argmax(select_mask.data.cpu().numpy(), 1)
                                == np.argmax(target.data.cpu().numpy(), 1))
        accuracy_match_count += batch_match_count
        sample_count += data.shape[0]

        running_loss += batch_loss.data.item() * data.shape[0]
        running_match_count += batch_match_count
        running_sample_count += data.shape[0]

        if batch_id > 0 and train_mode:
            writer.add_scalar(f'minibatch_running_{mode_string}_loss',
                              running_loss / running_sample_count,
                              epoch_idx * len(benchmark.train_loader) + batch_id)
            writer.add_scalar(f'minibatch_running_{mode_string}_accuracy',
                              running_match_count / running_sample_count,
                              epoch_idx * len(benchmark.train_loader) + batch_id)
            running_loss, running_sample_count, running_match_count = 0., 0, 0

        if batch_id % log_frequency == 0:
            logging.info(
                f"{mode_string.capitalize()} batch accuracy: " 
                + f"{batch_match_count/ float(data.shape[0])}")

    avg_epoch_loss: float = epoch_loss / sample_count
    epoch_accuracy: float = accuracy_match_count / sample_count
    logging.info(f"Average {mode_string} Loss: {avg_epoch_loss}.")
    logging.info(f"{mode_string.capitalize()} Accuracy: {epoch_accuracy}.")

    writer.add_scalar(f'epoch_{mode_string}ing_loss', avg_epoch_loss, epoch_idx)
    writer.add_scalar(f'epoch_{mode_string}ing_accuracy', epoch_accuracy, epoch_idx)

    writer.flush()
    if train_mode:
        return avg_epoch_loss, epoch_accuracy

    return data, reconstructions, avg_epoch_loss, epoch_accuracy


def train_epoch(epoch_idx: int, benchmark: IBenchmark, capsule_net: nn.Module,
                optimizer: Optimizer, use_cuda: bool, n_classes: int,
                writer: SummaryWriter, log_frequency: int,
                use_clipping: bool, clip_threshold: Optional[float]
                ) -> ProcessEpochReturnTyping:
    """ Performs epoch of training.

    Args:
        epoch_idx: Epoch index.
        benchmark: Benchmark of the current experiment.
        capsule_net: The capsule network for training.
        optimizer: The optimizer for updating network weights.
        use_cuda: Whether to use CUDA.
        n_classes: Number of classes.
        writer: Writer to TensorBoard.
        log_frequency: Frequency of logging.
        use_clipping: Whether to use gradient clipping.
        clip_threshold: The threshold for gradient clipping.

    Returns:
        Tuple with two elements. The first is the 
        average epoch training loss.  The second is 
        training epoch acuracy.
    """
    return process_epoch(epoch_idx, benchmark, capsule_net, optimizer,
                  use_cuda, n_classes, writer, True, log_frequency,
                  use_clipping, clip_threshold)


def eval_epoch(epoch_idx: int, benchmark: IBenchmark, capsule_net: nn.Module,
               use_cuda: bool, n_classes: int, writer: SummaryWriter, log_frequency: int
               ) -> ProcessEpochReturnTyping:
    """ Evaluates epoch. 

    Args:
        epoch_idx: Epoch index.
        benchmark: Benchmark of the current experiment.
        capsule_net: The capsule network for evaluation.
        use_cuda: Whether to use CUDA.
        n_classes: Number of classes.
        writer: Writer to TensorBoard.
        log_frequency: Frequency of logging.

    Returns:
        Tuple with four elements. 
        The first is some testing batch with data. 
        The second is the reconstructions on that batch. 
        The third is the average test loss value on that epoch. 
        The fourth is the test accuracy on that epoch. 

    """
    return process_epoch(epoch_idx, benchmark, capsule_net, None,
                         use_cuda, n_classes, writer, False,
                         log_frequency, False, None)


def do_training(setup_config: SetupConfig) -> Tuple[int, float, float, float, float]:
    """ Entry point to the training procedure. Trains capsule network,
    creates visualizations and provides LIME explanations if requested.

    Args:
        setup_config: Experiment configuration.

    Returns:
        Tuple with five elements. The first is the epoch number that achieved the best test 
        performance. The second is the test accuracy on that epoch. The third is the test 
        loss value on that epoch. The fourth is the train accuracy on that epoch. The fifth
        is the train loss value on that epoch.
    """
    logging.info("Started training.")

    benchmark_config: ConfigBenchmark = setup_config.benchmark_config
    benchmark_name: str = benchmark_config.benchmark
    training_config: ConfigTraining = setup_config.training_config

    if training_config.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    else:
        if training_config.use_cuda:
            torch.backends.cudnn.benchmark = True

    capsule_net: nn.Module = CapsNet(setup_config.network_config)
    logging.info("Network is built.")

    use_cuda: bool = training_config.use_cuda
    if use_cuda:
        capsule_net: nn.Module = capsule_net.cuda()
        logging.info("Transferred CapsNet to CUDA.")

    optimizer: Optimizer = Adam(capsule_net.parameters())
    logging.info("Optimizer is built.")

    start_epoch_idx: int = 0
    if training_config.load_checkpoint:
        start_epoch_idx = load_checkpoint(training_config.path_to_checkpoint,
                                          capsule_net, optimizer, use_cuda)
        logging.info("Loaded checkpoint.")

    benchmark: IBenchmark = build_benchmark(benchmark_config)
    logging.info("Benchmark is built.")

    writer: SummaryWriter = SummaryWriter(
        os.path.join(NameKeys.TRAINDIR, benchmark_name),
        filename_suffix=benchmark_name, flush_secs=60)

    data: TypingFloatTensor; reconstructions: TypingFloatTensor
    epoch_on_best_test: int
    best_test_accuracy: float; test_loss_on_best: float
    train_accuracy_on_best: float; train_loss_on_best: float

    (data, reconstructions, epoch_on_best_test,
     best_test_accuracy, test_loss_on_best,
     train_accuracy_on_best, train_loss_on_best) = iterate_training(
        start_epoch_idx, training_config.n_epochs, benchmark, capsule_net,
        optimizer, use_cuda, training_config.n_classes, writer, training_config,
        benchmark_name)

    image: TypingFloatTensor; labels: TypingIntTensor
    images, labels = next(iter(benchmark.train_loader))
    if training_config.graph_to_tensorboard:
        capsule_net.remove_hooks()
        writer.add_graph(capsule_net, images.cuda())

    logging.info("Visualizations.")
    visualizations_path: str = os.path.join(NameKeys.TRAINDIR,
                                          NameKeys.VISUALIZATIONS.format(benchmark_name))
    check_and_make_folder(visualizations_path)
    n_visualize: int = min(training_config.n_visualize, data.cpu().numpy().shape[0])
    plot_images_separately(
        data[:n_visualize, 0].data.cpu().numpy(),
        os.path.join(visualizations_path, NameKeys.SOURCE_PNG),
        n_visualize)
    plot_images_separately(
        reconstructions[:n_visualize, 0].data.cpu().numpy(),
        os.path.join(visualizations_path, NameKeys.RESTORED_PNG),
        n_visualize)
    del data, reconstructions

    if training_config.use_lime:
        logging.info("Started explanations.")
        load_checkpoint(os.path.join(training_config.checkpoint_root,
                        training_config.checkpoint_template.format(
                        benchmark_name, NameKeys.BEST_CHECKPOINT)),
                        capsule_net, optimizer, training_config.use_cuda)
        explanations_path: str = os.path.join(NameKeys.TRAINDIR,
                                              NameKeys.EXPLANATIONS.format(benchmark_name))
        check_and_make_folder(explanations_path)
        explain_lime(benchmark, capsule_net,
                     training_config.use_cuda, explanations_path)
        logging.info("Finished explanations.")

    logging.info("Finished training.")
    return (epoch_on_best_test,
            best_test_accuracy, test_loss_on_best,
            train_accuracy_on_best, train_loss_on_best)


def cuda_cache_reset(use_cuda: bool) -> None:
    """ Explicitly called memory management. 

    Args:
        use_cuda: Whether CUDA is used, since then CUDA-specific memory management is requested.
    """
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()


def dump_checkpoint(training_config: ConfigTraining, benchmark_name: str,
                    checkpoint_id: str, epoch_idx: int,
                    capsule_net: nn.Module, optimizer: Optimizer,
                    test_loss: float, test_accuracy: float,
                    train_loss: float, train_accuracy: float) -> None:
    """ Dumps checkpoint.  

    Args:
        training_config: The training configuration.
        benchmark_name: The name of the benchmark.
        checkpoint_id: The checkpoint identifier.
        epoch_idx: The epoch index.
        capsule_net: The network at the current state.
        optimizer: The optimizer at the current state.
        test_loss: The loss value on test set.
        test_accuracy: The accuracy value on test set.
        train_loss: The loss value on train set.
        train_accuracy: The accuracy value on train set.
    """
    if not training_config.dump_checkpoints:
        return
    if not os.path.isdir(training_config.checkpoint_root):
        os.makedirs(training_config.checkpoint_root)
    save_path: str = os.path.join(
        training_config.checkpoint_root,
        training_config.checkpoint_template.format(
            benchmark_name, checkpoint_id))
    save_checkpoint(save_path, epoch_idx, capsule_net, optimizer,
                    test_loss, test_accuracy, train_loss, train_accuracy)


def iterate_training(start_epoch_idx: int, n_epochs: int, benchmark: IBenchmark,
                     capsule_net: nn.Module, optimizer: Optimizer, use_cuda: bool,
                     n_classes: int, writer: SummaryWriter,
                     training_config: ConfigTraining, benchmark_name: str
                     ) -> Tuple[TypingFloatTensor, TypingFloatTensor,
                                int, float, float, float, float]:
    """ Method responsible for iterative training. 
    Iterates until the desired number of epochs is achieved. 

    Args:
        start_epoch_idx: The epoch intex to start.
        n_epochs: The total number of epochs.
        benchmark: The current benchmark for training and evaluation.
        capsule_net: The network to train and evaluate.
        optimizer: The optimizer for loss optimization.
        use_cuda: Whether to use CUDA.
        n_classes: The number of classes.
        writer: Writer for providing information to TensorBoard.
        training_config: The training configuration.
        benchmark_name: The name of the current benchmark for training and evaluation.

    Returns:
        Tuple with seven elements. The first is some testing batch with data. The second
        is the reconstructions on that batch. The third is the epoch number that achieved 
        the best test performance. The fourth is the test accuracy on that epoch. The fifth 
        is the test loss value on that epoch. The sixth is the train accuracy on that epoch. 
        The seventh is the train loss value on that epoch.
    """
    logging.info("Started training iterations.")
    best_test_accuracy: float = 0.
    epoch_on_best_test: int = -1
    test_loss_on_best_test: float = 0.
    train_loss_on_best_test: float = 0.
    train_accuracy_on_best_test: float = 0.

    data: TypingFloatTensor; reconstructions: TypingFloatTensor
    data, reconstructions = None, None
    epoch_idx: int
    train_loss: float; train_accuracy: float
    test_loss: float; test_accuracy: float
    for epoch_idx in range(start_epoch_idx, n_epochs):
        logging.info(f"Epoch {epoch_idx + 1} out of {n_epochs}.")
        del data, reconstructions
        cuda_cache_reset(use_cuda)
        train_loss, train_accuracy = train_epoch(epoch_idx, benchmark, capsule_net,
            optimizer, use_cuda, n_classes, writer,
            training_config.log_frequency, training_config.use_clipping,
            training_config.clipping_threshold)

        cuda_cache_reset(use_cuda)
        with torch.no_grad():
            if epoch_idx + 1 == n_epochs:
                benchmark.reset_test_loader(training_config.batch_size, True)
            data, reconstructions, test_loss, test_accuracy = eval_epoch(
                epoch_idx, benchmark, capsule_net,
                use_cuda, n_classes, writer, training_config.log_frequency)
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                epoch_on_best_test = epoch_idx
                test_loss_on_best_test = test_loss
                train_loss_on_best_test = train_loss
                train_accuracy_on_best_test = train_accuracy
                dump_checkpoint(training_config, benchmark_name, NameKeys.BEST_CHECKPOINT,
                                epoch_idx, capsule_net, optimizer, test_loss, test_accuracy,
                                train_loss, train_accuracy)

        if epoch_idx % training_config.checkpoint_frequency == 1 or epoch_idx + 1 == n_epochs:
            dump_checkpoint(training_config, benchmark_name, f"{epoch_idx + 1}",
                epoch_idx, capsule_net, optimizer, test_loss,
                test_accuracy, train_loss, train_accuracy)
    writer.close()
    logging.info("Finished training iterations.")
    return (data, reconstructions, epoch_on_best_test,
            best_test_accuracy, test_loss_on_best_test,
            train_accuracy_on_best_test, train_loss_on_best_test)

