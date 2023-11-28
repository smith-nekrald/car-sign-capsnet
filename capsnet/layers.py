""" Implements modules and layers used for building Capsule Network. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import Union
from typing import Optional
from typing import Tuple
from typing import List
from typing import Any

import logging

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from capsnet.config import ConfigSquash
from capsnet.config import ConfigConv
from capsnet.config import ConfigPrimary
from capsnet.config import ConfigAgreement
from capsnet.config import ConfigRecognition
from capsnet.config import ConfigReconstruction

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


def fix_nan_gradient_hook(gradient: TypingFloatTensor) -> Optional[TypingFloatTensor]:
    """ Replaces nans with zeroes in PyTorch gradient.

    Args:
        gradient: The gradient tensor to process.

    Returns:
        None if no NaN values is found in the gradient. Otherwise,
        returns gradient with NaNs replaced by zeroes.
    """
    if torch.any(torch.isnan(gradient)):
        logging.info("Fixing NaN gradient.")
        fixed_gradient: TypingFloatTensor = torch.where(
            torch.logical_not(torch.isnan(gradient)), gradient,
            torch.zeros_like(gradient).to(gradient.device))
        return fixed_gradient


def nan_gradient_hook_module(module: nn.Module, in_gradient: TypingFloatTensor,
                      out_gradient: TypingFloatTensor) -> Optional[List[TypingFloatTensor]]:
    """ Trick to remove nan values from gradients. The format of the argument is 
    to support PyTorch API; in reality, only out_gradient is updated. Iterates through
    all members in out_gradient and applies fix_nan_gradient_hook method.

    Args:
        module: The relevant module to process.
        in_gradient: The input gradient of the module to process.
        out_gradient: The output gradient of the module to process.

    Returns:
        None if nothing is updated/rewrited. Outherwise, returns list with updated gradients.
    """
    rewrite_grads: bool = False
    fixed_list: List[TypingFloatTensor] = list()
    grad_entry: TypingFloatTensor
    for grad_entry in out_gradient:
        fixed_grad: Optional[TypingFloatTensor] = fix_nan_gradient_hook(grad_entry)
        if fixed_grad is not None:
            fixed_list.append(fixed_grad)
            rewrite_grads = True
        else:
            fixed_list.append(grad_entry)
    if rewrite_grads:
        return fixed_list


class SquashLayer(nn.Module):
    """ Implements squashing. The formula from original paper is:
        $v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||$
        However, due to numerical stability issues, the applied formula is:
        $v_j = (||s_j + eps_input||^2 + eps_norm) 
        / (eps_denom + 1 + eps_norm  + ||s_j + eps_input||^2) 
        * (s_j + eps_input) / sqrt(eps_norm + ||s_j + eps_input||^2 + eps_sqrt)$                
    
    Attributes:
        eps_denom: Shift added in denominator for numerical stability.
        eps_sqrt: Shift added under square root for numerical stability.
        eps_input: Shift for input tensor added for numerical stability.
        eps_norm: Shift for squared norm added for numerical stability.
    """

    def __init__(self, config: ConfigSquash) -> None:
        """ Initializer method. 

        Args:
            config: Configuration for the squash layers. Specifies attributes.
        """
        super(SquashLayer, self).__init__()
        self.eps_denom: float = config.eps_denom
        self.eps_sqrt: float = config.eps_sqrt
        self.eps_input: float = config.eps_input
        self.eps_norm: float = config.eps_norm

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        """ Applies squashing.

        Args:
            input_tensor: The tensor to squash.

        Returns:
            The squashed tensor.
        """
        shifted_tensor: TypingFloatTensor = input_tensor + self.eps_input
        squared_norm: TypingFloatTensor = (shifted_tensor ** 2).sum(
            -1, keepdim=True) + self.eps_norm
        scaling_factor: TypingFloatTensor = squared_norm / (self.eps_denom
            + (1. + squared_norm) * torch.sqrt(squared_norm + self.eps_sqrt))
        output_tensor: TypingFloatTensor = scaling_factor * shifted_tensor
        return output_tensor


class ConvLayer(nn.Module):
    """ Applies convolution and batch normalization (if configured), with ReLU activation
    afterwards.

    Attributes:
        conv: The convolution module from PyTorch, configured.
        batch_norm: The batch normalization module from PyTorch, configured.
        use_batch_norm: Boolean flag specifying whether batch norm is applied.

    """
    def __init__(self, config: ConfigConv) -> None:
        """ Initializer method. Configures batch normalization and convolution.

        Args:
            config: The configuration object for convolution layer.
        """
        super(ConvLayer, self).__init__()
        self.conv: nn.Module = nn.Conv2d(in_channels=config.in_channels,
                               out_channels=config.out_channels,
                               kernel_size=config.kernel_size,
                               stride=config.stride)
        self.batch_norm: nn.Module = nn.BatchNorm2d(config.out_channels)
        self.use_batch_norm: bool = config.use_batch_norm

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        """ Applies convolution layer to input_tensor. Then applies batch normalization,
        if configured, and ReLU activation on top.

        Args:
            input_tensor: Input tensor to convolution.

        Returns:
            The tensor after applying the announced transformations.
        """
        convolved_input: TypingFloatTensor = self.conv(input_tensor)
        to_activate: TypingFloatTensor = convolved_input
        if self.use_batch_norm:
            to_activate = self.batch_norm(convolved_input)
        output_tensor: TypingFloatTensor = F.relu(to_activate)
        return output_tensor


class PrimaryCaps(nn.Module):
    """ Implements primary capsules. Primary capsules are the first layer of capsules applied. 
    Capsules are interpreted as something for finding important objects and measuring properties 
    of those objects. The length of the capsule output corresponds to the probability to have the 
    object in the input. Deeper capsules account for deeper objects.

    Attributes:
        capsules: A module list with configured convolution layers applied at each capsule.
        hook_handles: List with hooks/tricks to improve learning, e.g. replacing NaNs with
            zeroes in gradients for stabilization.
        dropouts: A module list with configured dropouts to be applied 
            at capsule convolution outputs if configured.
        capsule_output_dim: The flattened output dimension the capsule.
        squash: Module for squashing pre-squashed stacked capsule outputs.
        use_dropout: Boolean flag specifying whether dropout should be applied.
    
    """
    def __init__(self, primary_config: ConfigPrimary,
                 squash_config: ConfigSquash) -> None:
        """ Initializer method. Configures convolutions and dropouts for each capsule,
        creates squashing module and adds gradient hooks (if requested in config).

        Args:
            primary_config: Configuration for primary capsules.
            squash_config: Configuration for squashing module.
        """
        super(PrimaryCaps, self).__init__()
        self.capsules: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(in_channels=primary_config.in_conv_channels,
                      out_channels=primary_config.out_conv_channels,
                      kernel_size=primary_config.conv_kernel_size,
                      stride=primary_config.conv_stride,
                      padding=primary_config.conv_padding)
            for _ in range(primary_config.num_capsules)])
        self.hook_handles: List[Any] = list()
        if primary_config.use_nan_gradient_hook:
            entry: nn.Module
            for entry in self.capsules:
                handle: Any = entry.register_full_backward_hook(nan_gradient_hook_module)
                self.hook_handles.append(handle)
        self.dropouts: nn.ModuleList = nn.ModuleList([
            nn.Dropout(p=primary_config.dropout_proba)
            for _ in range(primary_config.num_capsules)
        ])
        self.capsule_output_dim: int = primary_config.capsule_output_dim
        self.squash: nn.Module = SquashLayer(squash_config)
        self.use_dropout: bool = primary_config.use_dropout

    def remove_hooks(self) -> None:
        """ Deactivates all applied hooks."""
        handle: Any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = list()

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        """ Applies primary capsules. Convolves, applies dropout (if specified 
        by self.use_dropout), stacks, and squashes. 

        Args:
            input_tensor: The input tensor to the primary capsule layer.

        Returns:
            Squashed capsule outputs.
        """
        capsule_list: List[TypingFloatTensor]
        if self.use_dropout:
            capsule_list = [drop(capsule(input_tensor))
                            for capsule, drop in zip(self.capsules, self.dropouts)]
        else:
            capsule_list = [capsule(input_tensor) for capsule in self.capsules]

        stacked_capsules: TypingFloatTensor = torch.stack(capsule_list, dim=1)
        batch_size: Union[int, torch.int] = input_tensor.size(0)
        presquashed_capsules: TypingFloatTensor = stacked_capsules.view(
            batch_size, -1, self.capsule_output_dim)
        squashed_capsules: TypingFloatTensor = self.squash(presquashed_capsules)
        return squashed_capsules


class AgreementRouting(nn.Module):
    """ Implements routing agreement. This process updates coupling coefficients
    between primary capsule layers through several iterations, and returns capsule 
    outputs after routing with iterative coupling.

    Attributes:
        n_iterations: Number of routing iterations.
        num_input_caps: Number of capsules in the input layer.
        num_output_caps: Number of capsules in the output layer.
        use_cuda: Whether to use CUDA (true/false).
        squash: The squashing module to apply while routing.
    """
    def __init__(self, agreement_config: ConfigAgreement,
                 squash_config: ConfigSquash) -> None:
        """ Initializer method. 

        Args:
            agreement_config: Configuration for routing agreement module.
            squash_config: Configuration for squashing module.
        """
        super(AgreementRouting, self).__init__()

        self.n_iterations: int = agreement_config.n_iterations
        assert self.n_iterations > 0

        self.num_input_caps: int = agreement_config.num_input_caps
        self.num_output_caps: int = agreement_config.num_output_caps
        self.output_caps_dim: int = agreement_config.output_caps_dim
        self.use_cuda: bool = agreement_config.use_cuda
        self.squash: nn.Module = SquashLayer(squash_config)

    def forward(self, u_ji_predict_5d: TypingFloatTensor) -> TypingFloatTensor:
        """ Applies routing agreement. 

        Args:
            u_ji_predict_5d: The prediction vectors (obtained by multiplying capsule outputs
                u_i by weights W_{ij}).

        Returns:
            Outputs v_j of the capsules for the next capsule layer (squashed).
        """
        u_ji_predict_4d: TypingFloatTensor = u_ji_predict_5d.squeeze(4)
        assert len(u_ji_predict_4d.shape) == 4

        batch_size: Union[int, torch.int32]; num_input_caps: Union[int, torch.int32]
        num_output_caps: Union[int, torch.int32]; output_caps_dim: Union[int, torch.int32]
        batch_size, num_input_caps, num_output_caps, output_caps_dim = u_ji_predict_4d.size()

        b_ij_batch_3d: TypingFloatTensor = torch.zeros(
            batch_size, num_input_caps, num_output_caps)
        if self.use_cuda:
            b_ij_batch_3d = b_ij_batch_3d.cuda()

        v_j_squashed_3d: Optional[TypingFloatTensor] = None
        if self.n_iterations > 0:
            idx_iteration: int
            for idx_iteration in range(self.n_iterations):
                b_i_2d: TypingFloatTensor = b_ij_batch_3d.view(-1, num_output_caps)
                c_ij_2d: TypingFloatTensor = F.softmax(b_i_2d, dim=1)
                c_ij_4d: TypingFloatTensor = c_ij_2d.view(-1, num_input_caps, num_output_caps, 1)
                s_j_3d: TypingFloatTensor = (c_ij_4d * u_ji_predict_4d).sum(dim=1)
                v_j_squashed_3d = self.squash(s_j_3d)
                v_j_aligned_4d: TypingFloatTensor = v_j_squashed_3d.unsqueeze(1)
                b_ij_batch_3d = b_ij_batch_3d + (u_ji_predict_4d * v_j_aligned_4d).sum(-1)

        assert v_j_squashed_3d is not None
        return v_j_squashed_3d


class RecognitionCaps(nn.Module):
    """ Implements recognition capsules. Recognition capsules are those applied after 
    the first layer with primary capsules. Capsules are interpreted as something for 
    finding important objects and measuring properties of those objects. The length of 
    the capsule output corresponds to the probability to have the object in the input. 
    Deeper capsules account for deeper objects.

    Attributes:
        num_input_caps: The number of input capsules.
        input_caps_dim: Dimension of the input capsule vector.
        num_output_caps: The number of output capsules.
        output_caps_dim: Dimension of output capsule vector.
        W_matrix_5d: Weight matrix to compute prediction 
            vectors u[j|i] from input capsule outputs u_i.
        W_hook_handle: Handle for W-related gradient hook. None if no hook applied.
        agreement_routing: Configured module to apply agreement routing.
        dropout: Configured module to apply dropout. 
        use_dropout: Boolean flag whether to apply dropout regularization.
    """
    def __init__(self, recognition_config: ConfigRecognition,
                 agreement_config: ConfigAgreement,
                 squash_config: ConfigSquash) -> None:
        """ Initializer method. Creates weight matrix W_matrix_5d and corresponding gradient
        hook (if configured). Also configures agreement_routing and dropout.
        
        Args:
            recognition_config: Configuration for recognition capsule.
            agreement_config: Configuration for agreement routing module.
            squash_config: Configuration for squashing module.
        """
        super(RecognitionCaps, self).__init__()

        self.num_input_caps: int = recognition_config.num_input_caps
        self.input_caps_dim: int = recognition_config.input_caps_dim
        self.num_output_caps: int = recognition_config.num_output_caps
        self.output_caps_dim: int = recognition_config.output_caps_dim

        self.W_matrix_5d: nn.Parameter = nn.Parameter(
            torch.randn(1, recognition_config.num_input_caps,
            recognition_config.num_output_caps,
            recognition_config.output_caps_dim,
            recognition_config.input_caps_dim))

        self.W_hook_handle: Any = None
        if recognition_config.use_nan_gradient_hook:
            self.W_hook_handle = self.W_matrix_5d.register_hook(fix_nan_gradient_hook)

        self.agreement_routing: nn.Module = AgreementRouting(
            agreement_config, squash_config)

        self.dropout: nn.Module = nn.Dropout(p=recognition_config.dropout_proba)
        self.use_dropout: bool = recognition_config.use_dropout

    def remove_hooks(self) -> None:
        """ Deactivates all applied hooks."""
        if self.W_hook_handle is not None:
            self.W_hook_handle.remove()
            self.W_hook_handle = None

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        """ Applies recognition capsule to input_tensor. Essentially, converts outputs of
        previous capsule layer to ouptut of the next capsule layer. 

        Args:
            input_tensor: The tensor with outputs from the previous capsule layer.

        Returns:
            The tensor with outputs for the further capsule layer.
        """
        batch_size: Union[int, torch.int32] = input_tensor.size(0)
        input_stack: TypingFloatTensor = torch.stack(
            [input_tensor] * self.num_output_caps, dim=2).unsqueeze(4)

        W_batch_5d: TypingFloatTensor = torch.cat([self.W_matrix_5d] * batch_size, dim=0)
        if self.use_dropout:
            W_batch_5d = self.dropout(W_batch_5d)
        u_hat_5d: TypingFloatTensor = torch.matmul(W_batch_5d, input_stack)
        return self.agreement_routing(u_hat_5d).view(
            batch_size, self.num_output_caps, self.output_caps_dim, 1)


class ReconstructionNet(nn.Module):
    """ Module to reconstruct the image of the predicted class 
    (i.e. class with longest capsule vector). This reconstruction 
    is used in the part of loss function computation (as regularizer), 
    and can also provide an opportunity to create reconstructions 
    for visual examination.

    Attributes:
        reconstruction_layers: A sequence with reconstruction layers. Each layer in the
            sequence is fully-connected  with ReLU on top (except the final layer, where 
            the result is activated with Sigmoid). 
        use_cuda: Whether to use CUDA.
        num_output_channels: Number of channels in the output image.
        output_image_size: The size of ouptut image.
        num_classes: The number of classes.
    """
    def __init__(self, config: ConfigReconstruction) -> None:
        """ Initializer method. Creates reconstruction layers. 

        Args:
            config: Configuration for the reconstruction module.
        """
        super(ReconstructionNet, self).__init__()

        modules: List[nn.Module] = [
            nn.Linear(config.linear_input_dim, config.linear_hidden_layers[0])]
        idx: int
        for idx in range(len(config.linear_hidden_layers)):
            modules.append(nn.ReLU(inplace=True))
            if idx + 1 != len(config.linear_hidden_layers):
                modules.append(nn.Linear(config.linear_hidden_layers[idx],
                                         config.linear_hidden_layers[idx + 1]))
            else:
                modules.append(nn.Linear(config.linear_hidden_layers[idx],
                                         config.linear_output_dim))
        modules.append(nn.Sigmoid())
        self.reconstruction_layers: nn.Module = nn.Sequential(*modules)

        self.use_cuda: bool = config.use_cuda
        self.num_output_channels: int = config.output_n_channels
        self.output_image_size: Tuple[int, int] = config.output_img_size
        self.num_classes: int = config.num_classes

    def forward(self, input_tensor: TypingFloatTensor
                ) -> Tuple[TypingFloatTensor, TypingFloatTensor]:
        """ Applies reconstruction module. 

        Args:
            input_tensor: The result from final capsule layer.

        Returns:
            Tuple with two elements. The first contains reconstructions, the
            second contains the mask for the selected class.
        """
        batch_size: Union[int, torch.int32] = input_tensor.size(0)
        logit_classes: TypingFloatTensor = torch.sqrt((input_tensor ** 2).sum(2))
        proba_classes: TypingFloatTensor = F.softmax(logit_classes, dim=1)

        max_length_indices_2d: TypingIntTensor
        _, max_length_indices_2d = proba_classes.max(dim=1)
        assert len(max_length_indices_2d.shape) == 2, f"{max_length_indices_2d.shape}"

        class_mask_2d: TypingFloatTensor = torch.sparse.torch.eye(self.num_classes)
        assert len(class_mask_2d.shape) == 2, f"{class_mask_2d.shape}"

        if self.use_cuda:
            class_mask_2d: TypingFloatTensor = class_mask_2d.cuda()
        selection_mask_2d: TypingFloatTensor = class_mask_2d.index_select(
            dim=0, index=max_length_indices_2d.squeeze(1).data)
        assert len(selection_mask_2d.shape) == 2, f"{selection_mask_2d.shape}"

        reconstructions_2d: TypingFloatTensor = self.reconstruction_layers(
            (input_tensor * selection_mask_2d[:, :, None, None]).view(batch_size, -1))
        reconstructions_4d: TypingFloatTensor = reconstructions_2d.view(
            batch_size, self.num_output_channels,
            self.output_image_size[0], self.output_image_size[1])

        return reconstructions_4d, selection_mask_2d
