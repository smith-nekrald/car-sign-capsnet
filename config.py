from typing import Optional
from typing import Union
from typing import Tuple
from typing import List

from keys import BenchmarkName
from keys import ColorSchema


class Constants:
    MEAN_CHINESE_GRAYSCALE = 0.4255
    STD_CHINESE_GRAYSCALE = 0.2235


class SetupConfig:
    def __init__(self):
        self.load_checkpoint: Optional[bool] = False
        self.path_to_checkpoint: Optional[str] = None

        self.dump_checkpoints: Optional[bool] = True
        self.checkpoint_root: Optional[str] = './checkpoints'
        self.checkpoint_template: Optional[str] = 'epoch_{}.cpkt'

        self.benchmark: Optional[str] = BenchmarkName.CHINESE
        self.image_color: Optional[str] = ColorSchema.GRAYSCALE
        self.num_channels: Optional[int] = 1

        self.estimate_normalization: bool = False
        self.n_point_to_estimate: int = 1000

        self.use_augmentation: Optional[bool] = True
        self.augment_proba: Optional[float] = 0.5
        self.random_entry_proba: Optional[float] = 0.1
        self.image_size: Optional[Tuple[int, int]] = (28, 28)
        self.mean_normalize: Optional[Union[float, Tuple[float, float, float]]] = Constants.MEAN_CHINESE_GRAYSCALE
        self.std_normalize: Optional[Union[float, Tuple[float, float, float]]] = Constants.STD_CHINESE_GRAYSCALE

        self.use_cuda: Optional[bool] = True
        self.optimizer: Optional[str] = 'adam'
        self.n_epochs: Optional[int] = 30
        self.batch_size: Optional[int] = 16
        self.n_classes: Optional[int] = 58

        self.conv_in_channels: Optional[int] = 1
        self.conv_out_channels: Optional[int] = 256
        self.conv_kernel_size: Optional[int] = 9
        self.conv_stride: Optional[int] = 1

        self.primary_num_capsules: Optional[int] = 8
        self.primary_in_channels: Optional[int] = self.conv_out_channels
        self.primary_out_channels: Optional[int] = 32
        self.primary_kernel_size: Optional[int] = 9
        self.primary_stride: Optional[int] = 2
        self.primary_padding: Optional[int] = 0
        self.primary_num_routes: Optional[int] = 32 * 6 * 6
        self.primary_eps_denom: Optional[float] = 1e-5
        self.primary_eps_sqrt: Optional[float] = 1e-6
        self.primary_eps_input_shift: Optional[float] = 1e-5
        self.primary_eps_squared_shift: Optional[float] = 1e-5

        self.recognition_num_classes: Optional[int] = self.n_classes
        self.recognition_num_routes: Optional[int] = 32 * 6 * 6
        self.recognition_in_channels: Optional[int] = self.primary_num_capsules
        self.recognition_out_channels: Optional[int] = 16
        self.recognition_routing_iterations: Optional[int] = 3
        self.recognition_eps_denom: Optional[float] = 1e-5
        self.recognition_eps_sqrt: Optional[float] = 1e-6
        self.recognition_eps_input_shift: Optional[float] = 1e-5
        self.recognition_eps_squared_shift: Optional[float] = 1e-5

        self.decoder_input_dimension: Optional[int] = self.recognition_num_classes * self.recognition_out_channels
        self.decoder_hidden_layers: List[int] = [512, 1024]
        self.decoder_output_size: int = self.image_size[0] * self.image_size[1]
        self.decoder_n_classes: int = self.recognition_num_classes
        self.decoder_image_size: Tuple[int, int] = self.image_size
        self.decoder_num_channels: int = self.num_channels

        self.net_margin_loss_right_blend: Optional[float] = 0.5
        self.net_margin_loss_square: Optional[bool] = True
        self.net_reconstruction_loss_reg: Optional[float] = 0.0005
        self.net_margin_left: Optional[float] = 0.9
        self.net_margin_right: Optional[float] = 0.1

