import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import SetupConfig


class ConvLayer(nn.Module):
    def __init__(self, config: SetupConfig):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=config.conv_in_channels,
                               out_channels=config.conv_out_channels,
                               kernel_size=config.conv_kernel_size,
                               stride=config.conv_stride)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, config: SetupConfig) -> None:
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=config.primary_in_channels,
                      out_channels=config.primary_out_channels,
                      kernel_size=config.primary_kernel_size,
                      stride=config.primary_stride,
                      padding=config.primary_padding)
            for _ in range(config.primary_num_capsules)])

        self.num_routes: int = config.primary_num_routes
        self.eps_denom: float = config.primary_eps_denom
        self.eps_sqrt: float = config.primary_eps_sqrt
        self.eps_input_shift: float = config.primary_eps_input_shift
        self.eps_squared_shift: float = config.primary_eps_squared_shift

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        input_tensor = input_tensor + self.eps_input_shift
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) + self.eps_squared_shift
        output_tensor = squared_norm * input_tensor / (
                    self.eps_denom + (1. + squared_norm) * torch.sqrt(squared_norm + self.eps_sqrt))
        return output_tensor


class RecognitionCaps(nn.Module):
    def __init__(self, config: SetupConfig):
        super(RecognitionCaps, self).__init__()

        self.in_channels = config.recognition_in_channels
        self.num_routes = config.recognition_num_routes
        self.num_capsules = config.recognition_num_classes

        self.W = nn.Parameter(torch.randn(1, self.num_routes, self.num_capsules,
            config.recognition_out_channels, config.recognition_in_channels))
        self.use_cuda: bool = config.use_cuda
        self.num_routing_iterations: int = config.recognition_routing_iterations
        self.eps_denom: float = config.recognition_eps_denom
        self.eps_sqrt: float = config.recognition_eps_sqrt
        self.eps_input_shift: float = config.recognition_eps_input_shift
        self.eps_squared_shift: float = config.recognition_eps_squared_shift

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if self.use_cuda:
            b_ij = b_ij.cuda()

        num_iterations: int = self.num_routing_iterations
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        input_tensor = input_tensor + self.eps_input_shift
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) + self.eps_squared_shift
        output_tensor = squared_norm * input_tensor / (
                    self.eps_denom + (1. + squared_norm) * torch.sqrt(squared_norm + self.eps_sqrt))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, config: SetupConfig) -> None:
        super(Decoder, self).__init__()
        modules = [nn.Linear(config.decoder_input_dimension, config.decoder_hidden_layers[0])]
        idx: int
        for idx in range(len(config.decoder_hidden_layers)):
            modules.append(nn.ReLU(inplace=True))
            if idx + 1 != len(config.decoder_hidden_layers):
                modules.append(nn.Linear(config.decoder_hidden_layers[idx],
                          config.decoder_hidden_layers[idx + 1]))
            else:
                modules.append(nn.Linear(config.decoder_hidden_layers[idx],
                                         config.decoder_output_size))
        modules.append(nn.Sigmoid())
        self.reconstruction_layers = nn.Sequential(*modules)
        self.use_cuda = config.use_cuda
        self.num_output_channels = config.decoder_num_channels
        self.output_image_size = config.decoder_image_size
        self.num_classes = config.decoder_n_classes

    def forward(self, x):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(self.num_classes))
        if self.use_cuda:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

        reconstructions = self.reconstruction_layers(
            (x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, self.num_output_channels,
                                               self.output_image_size[0], self.output_image_size[1])
        return reconstructions, masked

