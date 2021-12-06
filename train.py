import logging

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam

from network import CapsNet
from config import SetupConfig
from benchmark import build_benchmark
from visualize import plot_images_separately


def train_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer, use_cuda, n_classes):
    print(f"epoch {epoch_idx + 1} of {n_epochs}")
    capsule_net.train()
    train_loss = 0

    accuracy_match_count = 0
    sample_count = 0

    for batch_id, (data, target) in enumerate(benchmark.train_loader):
        if batch_id % 100 == 0:
            print(f"batch {batch_id} out of {len(benchmark.train_loader)}")
        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        batch_match_count = sum(np.argmax(masked.data.cpu().numpy(), 1)
                                == np.argmax(target.data.cpu().numpy(), 1))
        accuracy_match_count += batch_match_count
        sample_count += data.shape[0]

        if batch_id % 100 == 0:
            print("train batch accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                np.argmax(target.data.cpu().numpy(), 1)) / float(data.shape[0]))

    print("Average Train Loss:", train_loss / len(benchmark.train_loader))
    print("Train Accuracy:", accuracy_match_count / sample_count)


def eval_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer, use_cuda, n_classes):
    capsule_net.eval()
    test_loss = 0

    accuracy_match_count = 0
    sample_count = 0
    for batch_id, (data, target) in enumerate(benchmark.test_loader):

        target = torch.sparse.torch.eye(n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data.item()

        accuracy_match_count += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        sample_count += data.shape[0]

        if batch_id % 100 == 0:
            print("test batch accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                              np.argmax(target.data.cpu().numpy(), 1)) / float(data.shape[0]))

    print("Average Test Loss", test_loss / len(benchmark.test_loader))
    print("Test Accuracy:", accuracy_match_count / sample_count)

    return data, reconstructions


def do_training(config: SetupConfig) -> None:
    logging.info("Started training.")
    capsule_net = CapsNet(config)
    logging.info("Network is built.")
    use_cuda = config.use_cuda
    if use_cuda:
        capsule_net = capsule_net.cuda()
    optimizer = Adam(capsule_net.parameters())
    logging.info("Optimizer is built.")

    benchmark = build_benchmark(config)
    logging.info("Benchmark is built.")

    n_epochs = config.n_epochs
    n_classes = config.n_classes

    logging.info("Iterating between epochs.")
    for epoch_idx in range(n_epochs):
        train_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer, use_cuda, n_classes)
        data, reconstructions = eval_epoch(
            epoch_idx, n_epochs, benchmark, capsule_net, optimizer, use_cuda, n_classes)

    logging.info("Visualizations.")
    plot_images_separately(data[:6,0].data.cpu().numpy(), 'source.png')
    plot_images_separately(reconstructions[:6,0].data.cpu().numpy(), 'restored.png')
    logging.info("Finished training.")
