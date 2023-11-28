""" Implements Capsule Network for images classification. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import Union
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from capsnet.layers import ConvLayer
from capsnet.layers import PrimaryCaps
from capsnet.layers import RecognitionCaps
from capsnet.layers import ReconstructionNet
from capsnet.config import ConfigNetwork

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


class CapsNet(nn.Module):
    """ Implements Capsule Network for image classification. This architectures 
    starts by applying convolution layers, then uses primary capsules and applies
    recognition capsules afterwards. The loss function consists of two components: 
    margin_loss and reconstruction_loss. Margin loss estimates the quality of the 
    actual classification, while reconstruction loss regularizes on the difference 
    between reconstructed class and original image.

    Attributes:
        conv_layer: The initial convolution layer, pre-processing for input images.
        primary_capsules: The primary capsule layer.
        recognition_capsules: The layer with recognition capsules.
        reconstruction_net: Module for reconstructing the most probable class.
        net_reconstruction_loss_reg: Regularization coefficient for reconstruction loss.
        net_margin_loss_blend: Coefficient to blend impact of lower and upper margin parts.
        net_margin_upper: Upper margin in the margin loss.
        net_margin_lower: Lower margin in the margin loss.
        use_square_in_margin_loss: Whether to use squared margins in margin loss computation.
        mse_loss: Module to compute MSE loss.
    """
    def __init__(self, config: ConfigNetwork) -> None:
        """ Initializer method. Prepares conv_layer, primary_capsules, recognition_capsules,
        reconstruction_net and margin_loss.

        Args:
            config: Configuration for Capsule Network.
        """
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

    def remove_hooks(self) -> None:
        """ Removes hooks. Hooks are tricks to simplify/fix learning, 
        but can slow down the inference process.
        """
        self.primary_capsules.remove_hooks()
        self.recognition_capsules.remove_hooks()

    def forward(self, input_tensor: TypingFloatTensor
                ) -> Tuple[TypingFloatTensor, TypingFloatTensor,
                           TypingFloatTensor, TypingFloatTensor]:
        """ Applies capsule network to input tensor (batch with images).  

        Args:
            input_tensor: The batch with images.

        Returns:
            Tuple with four tensors. The first  is the output from final capsules,
            the second is the reconstructed most probable class, the third is the
            mask with predicted class, the fourth contains class probabilities.
        
        """
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
        """ Method to compute loss function. 

        Args:
            data_tensor: The tensor with input batch.
            output_tensor: The tensor with capsule outputs.
            target_tensor: Target classes.
            reconstructions_tensor: The tensor with reconstructed batch.

        Returns:
            Tensor with loss function. 
        """
        return self.margin_loss(output_tensor, target_tensor) + self.reconstruction_loss(
            data_tensor, reconstructions_tensor)

    def margin_loss(self, capsule_tensor: TypingFloatTensor,
                    target_mask: TypingFloatTensor) -> TypingFloatTensor:
        """ Method to compute margin loss. 
        
        Args:
            capsule_tensor: Tensor with capsule outputs.
            target_mask: Mask with true target labels.

        Returns:
            Tensor with average margin loss.
        """
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
                            reconstructions_tensor: TypingFloatTensor) -> TypingFloatTensor:
        """ Method to compute reconstruction loss.

        Args:
            data_tensor: Tensor containing batch with original images.
            reconstructions_tensor: Tensor containing batch with reconstructed images.

        Returns:
            Tensor with average reconstruction loss, multiplied by regularization coefficient. 
        """
        assert reconstructions_tensor.size(0) == data_tensor.size(0)
        batch_size = reconstructions_tensor.size(0)

        loss: TypingFloatTensor = self.mse_loss(
            reconstructions_tensor.view(batch_size, -1), data_tensor.view(batch_size, -1))
        return loss * self.net_reconstruction_loss_reg

