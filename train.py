import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

from network import CapsNet


def train_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer):
    print(f"epoch {epoch_idx + 1} of {n_epochs}")
    capsule_net.train()
    train_loss = 0

    accuracy_match_count = 0
    sample_count = 0

    for batch_id, (data, target) in enumerate(benchmark.train_loader):
        if batch_id % 100 == 0:
            print(f"batch {batch_id} out of {len(benchmark.train_loader)}")
        target = torch.sparse.torch.eye(58).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()

        batch_match_count = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        accuracy_match_count += batch_match_count
        sample_count += data.shape[0]

        if batch_id % 100 == 0:
            print("train batch accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                               np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))

    print("Average Train Loss:", train_loss / len(benchmark.train_loader))
    print("Train Accuracy:", accuracy_match_count / sample_count)


def eval_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer):
    capsule_net.eval()
    test_loss = 0

    accuracy_match_count = 0
    sample_count = 0
    for batch_id, (data, target) in enumerate(benchmark.test_loader):

        target = torch.sparse.torch.eye(58).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data.item()

        accuracy_match_count += sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        sample_count += data.shape[0]

        if batch_id % 100 == 0:
            print("test batch accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                              np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))

    print("Average Test Loss", test_loss / len(benchmark.test_loader))
    print("Test Accuracy:", accuracy_match_count / sample_count)


def do_training(config):
    capsule_net = CapsNet()
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    optimizer = Adam(capsule_net.parameters())

    batch_size = 16
    benchmark = ChineseTraffic(batch_size)

    n_epochs = 30

    for epoch_idx in range(n_epochs):
        train_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer)
        eval_epoch(epoch_idx, n_epochs, benchmark, capsule_net, optimizer)

