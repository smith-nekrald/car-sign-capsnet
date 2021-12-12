from typing import Tuple
from typing import Optional

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

from network import CapsNet
from config import SetupConfig
from benchmark import build_benchmark
from visualize import plot_images_separately
from explain import explain_lime
from explain import check_and_make_folder
from keys import NameKeys
from checkpoint import load_checkpoint
from checkpoint import save_checkpoint


def process_epoch(epoch_idx, benchmark, capsule_net, optimizer: Optional[Optimizer],
                  use_cuda, n_classes, writer: SummaryWriter, train_mode: bool, log_frequency: int):
    if train_mode:
        capsule_net.train()
        data_loader = benchmark.train_loader
        mode_string: str = NameKeys.TRAIN_MODE_STRING
    else:
        capsule_net.eval()
        data_loader = benchmark.test_loader
        mode_string: str = NameKeys.TEST_MODE_STRING

    logging.info(f"{mode_string.capitalize()}ing.")
    epoch_loss: float = 0

    accuracy_match_count: int = 0
    sample_count: int = 0

    running_loss: float = 0.
    running_match_count: int = 0
    running_sample_count: int = 0

    data, reconstructions = None, None
    for batch_id, (data, target) in enumerate(data_loader):
        if batch_id % log_frequency == 0:
            logging.info(f"{mode_string.capitalize()} batch {batch_id} out of {len(data_loader)}")
        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        if train_mode:
            optimizer.zero_grad()

        output, reconstructions, masked, class_probas = capsule_net(data)
        batch_loss = capsule_net.loss(data, output, target, reconstructions)

        if train_mode:
            batch_loss.backward()
            optimizer.step()

        epoch_loss += batch_loss.data.item() * data.shape[0]

        batch_match_count = sum(np.argmax(masked.data.cpu().numpy(), 1)
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
            logging.info(f"{mode_string.capitalize()} batch accuracy: {batch_match_count/ float(data.shape[0])}")

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


def train_epoch(epoch_idx, benchmark, capsule_net, optimizer,
                use_cuda, n_classes, writer: SummaryWriter, log_frequency: int):
    return process_epoch(epoch_idx, benchmark, capsule_net, optimizer,
                  use_cuda, n_classes, writer, True, log_frequency)


def eval_epoch(epoch_idx, benchmark, capsule_net,
               use_cuda, n_classes, writer: SummaryWriter, log_frequency: int):
    return process_epoch(epoch_idx, benchmark, capsule_net, None,
                         use_cuda, n_classes, writer, False, log_frequency)


def do_training(setup_config: SetupConfig) -> Tuple[float, int]:
    logging.info("Started training.")

    benchmark_config = setup_config.benchmark_config
    benchmark_name = benchmark_config.benchmark
    training_config = setup_config.training_config

    if training_config.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    else:
        if training_config.use_cuda:
            torch.backends.cudnn.benchmark = True

    capsule_net: nn.Module = CapsNet(setup_config.network_config)
    logging.info("Network is built.")

    use_cuda: bool = training_config.use_cuda
    if use_cuda:
        capsule_net = capsule_net.cuda()
        logging.info("Transferred CapsNet to CUDA.")

    optimizer = Adam(capsule_net.parameters())
    logging.info("Optimizer is built.")

    start_epoch_idx: int = 0
    if training_config.load_checkpoint:
        start_epoch_idx = load_checkpoint(training_config.path_to_checkpoint,
                                          capsule_net, optimizer, use_cuda)
        logging.info("Loaded checkpoint.")

    benchmark = build_benchmark(benchmark_config)
    logging.info("Benchmark is built.")

    writer: SummaryWriter = SummaryWriter(
        os.path.join(NameKeys.TRAINDIR, benchmark_name),
        filename_suffix=benchmark_name, flush_secs=60)
    images, labels = next(iter(benchmark.train_loader))
    writer.add_graph(capsule_net, images.cuda())

    data, reconstructions, best_test_accuracy, epoch_on_best_test = iterate_training(
        start_epoch_idx, training_config.n_epochs, benchmark, capsule_net,
        optimizer, use_cuda, training_config.n_classes, writer, training_config,
        benchmark_name)

    logging.info("Visualizations.")
    visualizations_path: str = os.path.join(NameKeys.TRAINDIR,
                                          NameKeys.VISUALIZATIONS.format(benchmark_name))
    check_and_make_folder(visualizations_path)
    plot_images_separately(
        data[:training_config.n_visualize, 0].data.cpu().numpy(),
        os.path.join(visualizations_path, NameKeys.SOURCE_PNG))
    plot_images_separately(
        reconstructions[:training_config.n_visualize, 0].data.cpu().numpy(),
        os.path.join(visualizations_path, NameKeys.RESTORED_PNG))
    del data, reconstructions

    logging.info("Explanations.")
    explanations_path: str = os.path.join(NameKeys.TRAINDIR,
                                          NameKeys.EXPLANATIONS.format(benchmark_name))
    check_and_make_folder(explanations_path)
    explain_lime(benchmark, capsule_net,
                 training_config.use_cuda, explanations_path)

    logging.info("Finished training.")
    return best_test_accuracy, epoch_on_best_test


def cuda_cache_reset(use_cuda: bool) -> None:
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()


def dump_checkpoint(training_config, benchmark_name, checkpoint_id,
                    epoch_idx, capsule_net, optimizer,
                    test_loss, test_accuracy) -> None:
    if not training_config.dump_checkpoints:
        return
    if not os.path.isdir(training_config.checkpoint_root):
        os.makedirs(training_config.checkpoint_root)
    save_path: str = os.path.join(training_config.checkpoint_root,
                                  training_config.checkpoint_template.format(
                                      benchmark_name, checkpoint_id))
    save_checkpoint(save_path, epoch_idx, capsule_net, optimizer,
                    test_loss, test_accuracy)


def iterate_training(start_epoch_idx, n_epochs, benchmark, capsule_net,
            optimizer, use_cuda, n_classes, writer, training_config, benchmark_name):
    logging.info("Started training iterations.")
    best_test_accuracy = 0.
    epoch_on_best_test = -1
    data, reconstructions = None, None
    for epoch_idx in range(start_epoch_idx, n_epochs):
        logging.info(f"Epoch {epoch_idx} out of {n_epochs}.")
        del data, reconstructions
        cuda_cache_reset(use_cuda)
        train_epoch(epoch_idx, benchmark, capsule_net,
            optimizer, use_cuda, n_classes, writer,
            training_config.log_frequency)

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
                dump_checkpoint(training_config, benchmark_name, 'best',
                                epoch_idx, capsule_net, optimizer, test_loss, test_accuracy)

        if (epoch_idx % training_config.checkpoint_frequency == 1
                or epoch_idx + 1 == n_epochs):
            dump_checkpoint(training_config, benchmark_name, epoch_idx + 1,
                                epoch_idx, capsule_net, optimizer, test_loss, test_accuracy)
    writer.close()
    logging.info("Finished training iterations.")
    return data, reconstructions, best_test_accuracy, epoch_on_best_test
