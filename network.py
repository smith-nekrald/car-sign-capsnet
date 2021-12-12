from typing import Union
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvLayer
from layers import PrimaryCaps
from layers import RecognitionCaps
from layers import ReconstructionNet
from config import ConfigNetwork

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


class CapsNet(nn.Module):
    def __init__(self, config: ConfigNetwork) -> None:
        super(CapsNet, self).__init__()
        self.conv_layer: nn.Module = ConvLayer(config.conv_config)
        self.primary_capsules: nn.Module = PrimaryCaps(
            config.primary_config, config.squash_config)
        self.recognition_capsules: nn.Module = RecognitionCaps(
            config.recognition_config, config.agreement_config, config.squash_config)
        self.reconstruction_net: nn.Module = ReconstructionNet(config.reconstruction_config)

        self.net_reconstruction_loss_reg: float = config.net_reconstruction_loss_reg
        self.net_margin_loss_blend: float = config.net_margin_loss_blend
        self.net_margin_upper: float = config.net_margin_upper
        self.net_margin_lower: float = config.net_margin_lower
        self.use_square_in_margin_loss: bool = config.use_square_in_margin_loss

        self.mse_loss: nn.Module = nn.MSELoss()

    def forward(self, input_tensor: TypingFloatTensor
                ) -> Tuple[TypingFloatTensor, TypingFloatTensor,
                           TypingFloatTensor, TypingFloatTensor]:
        capsule_tensor: TypingFloatTensor = self.recognition_capsules(
            self.primary_capsules(self.conv_layer(input_tensor)))
        class_logits: TypingFloatTensor = torch.sqrt((capsule_tensor ** 2).sum(2)).squeeze(2)
        class_probas: TypingFloatTensor = F.softmax(class_logits, dim=1)

        reconstructions: TypingFloatTensor; selection_mask_2d: TypingFloatTensor
        reconstructions, selection_mask_2d = self.reconstruction_net(capsule_tensor)
        return capsule_tensor, reconstructions, selection_mask_2d, class_probas

    def loss(self, data_tensor: TypingFloatTensor, output_tensor: TypingFloatTensor,
             target_tensor: TypingFloatTensor, reconstructions_tensor: TypingFloatTensor
             ) -> TypingFloatTensor:
        return self.margin_loss(output_tensor, target_tensor) + self.reconstruction_loss(
            data_tensor, reconstructions_tensor)

    def margin_loss(self, capsule_tensor: TypingFloatTensor,
                    target_mask: TypingFloatTensor) -> TypingFloatTensor:
        batch_size: Union[int, torch.int32] = capsule_tensor.size(0)

        caps_lengths: TypingFloatTensor = torch.sqrt(
            (capsule_tensor ** 2).sum(dim=2, keepdim=True))

        upper_margin: TypingFloatTensor = F.relu(
            self.net_margin_upper - caps_lengths).view(batch_size, -1)
        lower_margin: TypingFloatTensor = F.relu(
            caps_lengths - self.net_margin_lower).view(batch_size, -1)

        loss: TypingFloatTensor
        if self.use_square_in_margin_loss:
            loss = target_mask * upper_margin ** 2 + self.net_margin_loss_blend * (
                        1.0 - target_mask) * lower_margin ** 2
        else:
            loss = target_mask * upper_margin + self.net_margin_loss_blend * (
                    1.0 - target_mask) * lower_margin
        average_loss: TypingFloatTensor = loss.sum(dim=1).mean()

        return average_loss

    def reconstruction_loss(self, data_tensor: TypingFloatTensor,
                            reconstructions_tensor: TypingFloatTensor):
        assert reconstructions_tensor.size(0) == data_tensor.size(0)
        batch_size = reconstructions_tensor.size(0)

        loss: TypingFloatTensor = self.mse_loss(
            reconstructions_tensor.view(batch_size, -1), data_tensor.view(batch_size, -1))
        return loss * self.net_reconstruction_loss_reg
