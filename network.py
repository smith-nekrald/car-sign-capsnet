import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvLayer
from layers import PrimaryCaps
from layers import RecognitionCaps
from layers import Decoder
from config import SetupConfig


class CapsNet(nn.Module):
    def __init__(self, config: SetupConfig) -> None:
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(config)
        self.primary_capsules = PrimaryCaps(config)
        self.recognition_capsules = RecognitionCaps(config)
        self.decoder = Decoder(config)

        self.net_reconstruction_loss_reg: float = config.net_reconstruction_loss_reg
        self.net_margin_loss_blend: float = config.net_margin_loss_right_blend
        self.net_margin_lhs: float = config.net_margin_left
        self.net_margin_rhs: float = config.net_margin_right
        self.use_square_in_margin_loss: bool = config.net_margin_loss_square

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.recognition_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(self.net_margin_lhs - v_c).view(batch_size, -1)
        right = F.relu(v_c - self.net_margin_rhs).view(batch_size, -1)

        if self.use_square_in_margin_loss:
            loss = labels * left ** 2 + self.net_margin_loss_blend * (1.0 - labels) * right ** 2
        else:
            loss = labels * left + self.net_margin_loss_blend * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                             data.view(reconstructions.size(0), -1))
        return loss * self.net_reconstruction_loss_reg