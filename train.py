from typing import Dict
from typing import Tuple
from typing import Any

import logging
import os
import gc

import numpy as np

import torch
import torchvision
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


def train_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer,
                use_cuda, n_classes, writer: SummaryWriter):
    logging.info(f"Training epoch {epoch_idx + 1} of {n_epochs}")
    capsule_net.train()
    train_loss: float = 0

    accuracy_match_count: int = 0
    sample_count: int = 0

    running_loss: float = 0.
    running_match_count: int = 0
    running_sample_count: int = 0

    for batch_id, (data, target) in enumerate(benchmark.train_loader):
        if batch_id % 100 == 0:
            logging.info(f"Training: batch {batch_id} out of {len(benchmark.train_loader)}")
        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked, class_probas = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item() * data.shape[0]

        batch_match_count = sum(np.argmax(masked.data.cpu().numpy(), 1)
                                == np.argmax(target.data.cpu().numpy(), 1))
        accuracy_match_count += batch_match_count
        sample_count += data.shape[0]

        running_loss += loss.data.item()
        running_match_count += batch_match_count
        running_sample_count += data.shape[0]

        if batch_id % 10 == 0 and batch_id > 0:
            writer.add_scalar('minibatch_running_train_loss', running_loss / 10,
                              epoch_idx * len(benchmark.train_loader) + batch_id)
            running_loss = 0.
            writer.add_scalar('minibatch_running_train_accuracy',
                              running_match_count / running_sample_count,
                              epoch_idx * len(benchmark.train_loader) + batch_id)
            running_sample_count = 0
            running_match_count = 0

        if batch_id % 100 == 0:
            logging.info(f"Train batch accuracy: {batch_match_count/ float(data.shape[0])}")

    avg_train_loss: float = train_loss / sample_count
    train_accuracy: float = accuracy_match_count / sample_count
    logging.info(f"Average Train Loss: {avg_train_loss}.")
    logging.info(f"Train Accuracy: {train_accuracy}.")

    writer.add_scalar('epoch_training_loss', avg_train_loss, epoch_idx)
    writer.add_scalar('epoch_training_accuracy', train_accuracy, epoch_idx)

    writer.flush()
    return avg_train_loss, train_accuracy


def eval_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer,
               use_cuda, n_classes, writer: SummaryWriter):
    capsule_net.eval()
    test_loss = 0

    accuracy_match_count = 0
    sample_count = 0

    data = None
    reconstructions = None

    for batch_id, (data, target) in enumerate(benchmark.test_loader):

        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked, class_probas = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data.item() * data.shape[0]

        batch_match_count: int = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        accuracy_match_count += batch_match_count
        sample_count += data.shape[0]

        if batch_id % 100 == 0:
            logging.info(f"Test batch accuracy: {batch_match_count/ float(data.shape[0])}")

    avg_test_loss: float = test_loss / sample_count
    test_accuracy: float = accuracy_match_count / sample_count

    logging.info(f"Average Test Loss: {avg_test_loss}")
    logging.info(f"Test Accuracy: {test_accuracy}")

    writer.add_scalar('epoch_test_loss', avg_test_loss, epoch_idx)
    writer.add_scalar('epoch_test_accuracy', test_accuracy, epoch_idx)

    writer.flush()
    return data, reconstructions, avg_test_loss, test_accuracy


def save_checkpoint(checkpoint_path: str, epoch: int, model: nn.Module,
                    optimizer: Optimizer,
                    train_loss: float, train_accuracy: float,
                    test_loss: float, test_accuracy: float) -> None:
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }, checkpoint_path)
    logging.info(f"Saved checkpoint for epoch {epoch + 1}.")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optimizer, use_cuda: bool) -> int:
    device: str = 'cpu'
    if use_cuda:
        device: str = 'cuda'

    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch_idx: int = checkpoint['epoch']
    train_accuracy: float = checkpoint['train_accuracy']
    train_loss: float = checkpoint['train_loss']
    test_accuracy: float = checkpoint['test_accuracy']
    test_loss: float = checkpoint['test_loss']

    logging.info(f"Loaded checkpoint after epoch: {epoch_idx}. ")
    logging.info(f"Checkpoint train accuracy: {train_accuracy}.")
    logging.info(f"Checkpoint train loss: {train_loss}.")
    logging.info(f"Checkpoint test accuracy: {test_accuracy}.")
    logging.info(f"Checkpoint test loss: {test_loss}.")

    return epoch_idx


def do_training(setup_config: SetupConfig) -> Tuple[float, int]:
    benchmark_config = setup_config.benchmark_config
    config = setup_config.training_config
    logging.info("Started training.")
    if config.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    else:
        if config.use_cuda:
            torch.backends.cudnn.benchmark = True

    capsule_net: nn.Module = CapsNet(setup_config.network_config)
    logging.info("Network is built.")

    use_cuda: bool = config.use_cuda
    if use_cuda:
        capsule_net = capsule_net.cuda()
        logging.info("Transferred CapsNet to CUDA.")

    optimizer = Adam(capsule_net.parameters())
    logging.info("Optimizer is built.")

    start_epoch_idx: int = 0
    if config.load_checkpoint:
        start_epoch_idx = load_checkpoint(config.path_to_checkpoint,
                                          capsule_net, optimizer, use_cuda)
        logging.info("Loaded checkpoint.")

    benchmark = build_benchmark(benchmark_config)
    logging.info("Benchmark is built.")

    n_epochs = config.n_epochs
    n_classes = config.n_classes

    logging.info("Iterating between epochs.")
    data = None
    reconstructions = None

    writer: SummaryWriter = SummaryWriter(
        "traindir/" + benchmark_config.benchmark,
        filename_suffix="_" + benchmark_config.benchmark, flush_secs=60)
    images, labels = next(iter(benchmark.train_loader))
    writer.add_graph(capsule_net, images.cuda())

    best_test_accuracy = 0.
    epoch_on_best_test = -1
    for epoch_idx in range(start_epoch_idx, n_epochs):
        del data, reconstructions
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

        train_loss, train_accuracy = train_epoch(
            epoch_idx, n_epochs, benchmark, capsule_net,
            optimizer, use_cuda, n_classes, writer)

        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

        with torch.no_grad():
            if epoch_idx + 1 == n_epochs:
                benchmark.reset_test_loader(config.batch_size, True)

            data, reconstructions, test_loss, test_accuracy = eval_epoch(
                epoch_idx, n_epochs, benchmark, capsule_net,
                optimizer, use_cuda, n_classes, writer)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epoch_on_best_test = epoch_idx

        if config.dump_checkpoints and (
                epoch_idx % 5 == 1 or epoch_idx + 1 == n_epochs):
            if not os.path.isdir(config.checkpoint_root):
                os.makedirs(config.checkpoint_root)
            save_path: str = os.path.join(config.checkpoint_root,
                config.checkpoint_template.format(benchmark_config.benchmark, epoch_idx + 1))
            save_checkpoint(save_path, epoch_idx, capsule_net, optimizer,
                    train_loss, train_accuracy, test_loss, test_accuracy)

    writer.close()
    logging.info("Visualizations.")
    explanations_path = 'traindir/explanations_' + benchmark_config.benchmark
    check_and_make_folder(explanations_path)
    plot_images_separately(data[:6, 0].data.cpu().numpy(),
                           os.path.join(explanations_path, 'source.png'))
    plot_images_separately(reconstructions[:6, 0].data.cpu().numpy(),
                           os.path.join(explanations_path, 'restored.png'))
    logging.info("Finished training.")
    del data, reconstructions
    explain_lime(benchmark, capsule_net, config.use_cuda,
                 'traindir/explanations_' + benchmark_config.benchmark)

    return best_test_accuracy, epoch_on_best_test

