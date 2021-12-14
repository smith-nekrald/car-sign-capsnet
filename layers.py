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

from config import ConfigSquash
from config import ConfigConv
from config import ConfigPrimary
from config import ConfigAgreement
from config import ConfigRecognition
from config import ConfigReconstruction

TypingFloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
TypingBoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
TypingIntTensor = Union[torch.IntTensor, torch.cuda.IntTensor]


def fix_nan_gradient_hook(gradient: TypingFloatTensor) -> Optional[TypingFloatTensor]:
    if torch.any(torch.isnan(gradient)):
        logging.info("Fixing NaN gradient.")
        fixed_gradient: TypingFloatTensor = torch.where(
            torch.logical_not(torch.isnan(gradient)), gradient,
            torch.zeros_like(gradient).to(gradient.device))
        return fixed_gradient


def nan_gradient_hook_module(module: nn.Module, in_gradient: TypingFloatTensor,
                      out_gradient: TypingFloatTensor) -> Optional[TypingFloatTensor]:
    rewrite_grads: bool = False
    fixed_list = list()
    for grad_entry in out_gradient:
        fixed_grad = fix_nan_gradient_hook(grad_entry)
        if fixed_grad is not None:
            fixed_list.append(fixed_grad)
            rewrite_grads = True
        else:
            fixed_list.append(grad_entry)
    if rewrite_grads:
        return fixed_list


class SquashLayer(nn.Module):
    def __init__(self, config: ConfigSquash) -> None:
        super(SquashLayer, self).__init__()
        self.eps_denom: float = config.eps_denom
        self.eps_sqrt: float = config.eps_sqrt
        self.eps_input: float = config.eps_input
        self.eps_norm: float = config.eps_norm

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        shifted_tensor: TypingFloatTensor = input_tensor + self.eps_input
        squared_norm: TypingFloatTensor = (shifted_tensor ** 2).sum(
            -1, keepdim=True) + self.eps_norm
        scaling_factor: TypingFloatTensor = squared_norm / (self.eps_denom
            + (1. + squared_norm) * torch.sqrt(squared_norm + self.eps_sqrt))
        output_tensor: TypingFloatTensor = scaling_factor * shifted_tensor
        return output_tensor


class ConvLayer(nn.Module):
    def __init__(self, config: ConfigConv) -> None:
        super(ConvLayer, self).__init__()
        self.conv: nn.Module = nn.Conv2d(in_channels=config.in_channels,
                               out_channels=config.out_channels,
                               kernel_size=config.kernel_size,
                               stride=config.stride)
        self.batch_norm: nn.Module = nn.BatchNorm2d(config.out_channels)
        self.use_batch_norm: bool = config.use_batch_norm

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
        convolved_input: TypingFloatTensor = self.conv(input_tensor)
        to_activate: TypingFloatTensor = convolved_input
        if self.use_batch_norm:
            to_activate = self.batch_norm(convolved_input)
        output_tensor: TypingFloatTensor = F.relu(to_activate)
        return output_tensor


class PrimaryCaps(nn.Module):
    def __init__(self, primary_config: ConfigPrimary,
                 squash_config: ConfigSquash) -> None:
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
        handle: Any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = list()

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
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
    def __init__(self, agreement_config: ConfigAgreement,
                 squash_config: ConfigSquash) -> None:
        super(AgreementRouting, self).__init__()

        self.n_iterations: int = agreement_config.n_iterations
        assert self.n_iterations > 0

        self.num_input_caps: int = agreement_config.num_input_caps
        self.num_output_caps: int = agreement_config.num_output_caps
        self.output_caps_dim: int = agreement_config.output_caps_dim
        self.use_cuda: bool = agreement_config.use_cuda
        self.squash: nn.Module = SquashLayer(squash_config)

    def forward(self, u_ji_predict_5d: TypingFloatTensor) -> TypingFloatTensor:
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
    def __init__(self, recognition_config: ConfigRecognition,
                 agreement_config: ConfigAgreement,
                 squash_config: ConfigSquash) -> None:
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
        if self.W_hook_handle is not None:
            self.W_hook_handle.remove()
            self.W_hook_handle = None

    def forward(self, input_tensor: TypingFloatTensor) -> TypingFloatTensor:
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
    def __init__(self, config: ConfigReconstruction) -> None:
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
