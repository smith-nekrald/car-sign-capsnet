from typing import Optional
from typing import Union
from typing import Tuple
from typing import List

from keys import BenchmarkName
from keys import ColorSchema
from keys import Constants


class ConfigSquash:
    def __init__(self) -> None:
        self.eps_denom: float = 1e-5
        self.eps_sqrt: float = 1e-6
        self.eps_input: float = 1e-5
        self.eps_norm: float = 1e-5


class ConfigConv:
    def __init__(self) -> None:
        self.in_channels: int = 1
        self.out_channels: int = 64
        self.kernel_size: int = 9
        self.use_batch_norm: bool = True
        self.stride: int = 1


class ConfigPrimary:
    def __init__(self) -> None:
        self.num_capsules: int = 12
        self.in_conv_channels: int = 64
        self.out_conv_channels: int = 24
        self.conv_kernel_size: int = 9
        self.conv_stride: int = 2
        self.conv_padding: int = 0
        self.capsule_output_dim: int = 24 * 6 * 6
        self.use_dropout: bool = True
        self.dropout_proba: float = 0.4


class ConfigAgreement:
    def __init__(self) -> None:
        self.num_input_caps: int = 12
        self.num_output_caps: int = 58
        self.n_iterations: int = 3
        self.output_caps_dim: int = 16
        self.use_cuda: bool = True


class ConfigRecognition:
    def __init__(self) -> None:
        self.num_output_caps: int = 58
        self.output_caps_dim: int = 16
        self.num_input_caps: int = 12
        self.input_caps_dim: int = 24 * 6 * 6
        self.use_dropout: bool = True
        self.dropout_proba: float = 0.4


class ConfigReconstruction:
    def __init__(self):
        self.linear_input_dim: int = 16 * 58
        self.linear_hidden_layers: List[int] = [512, 1024]
        self.linear_output_dim: int = 1 * 28 * 28
        self.num_classes: int = 58
        self.use_cuda = True
        self.output_img_size: Tuple[int, int] = (28, 28)
        self.output_n_channels: int = 1


class ConfigNetwork:
    def __init__(self) -> None:
        self.conv_config: ConfigConv = ConfigConv()
        self.primary_config: ConfigPrimary = ConfigPrimary()
        self.squash_config: ConfigSquash = ConfigSquash()
        self.agreement_config: ConfigAgreement = ConfigAgreement()
        self.recognition_config: ConfigRecognition = ConfigRecognition()
        self.reconstruction_config: ConfigReconstruction = ConfigReconstruction()

        self.net_reconstruction_loss_reg: float = 0.0005
        self.net_margin_loss_blend: float = 0.5
        self.net_margin_upper: float = 0.9
        self.net_margin_lower: float = 0.1
        self.use_square_in_margin_loss: bool = True


class ConfigBenchmark:
    def __init__(self) -> None:
        self.num_load_workers: int = 1

        self.benchmark: str = BenchmarkName.CHINESE
        self.image_size: Tuple[int, int] = (28, 28)
        self.image_color: str = ColorSchema.GRAYSCALE
        self.num_channels: int = 1

        self.use_augmentation: bool = True
        self.augment_proba: float = 0.5
        self.random_entry_proba: float = 0.1

        self.estimate_normalization: bool = False
        self.n_point_to_estimate: int = 1000
        self.mean_normalize: Union[None, float, Tuple[float, float, float]
            ] = Constants.MEAN_CHINESE_GRAYSCALE
        self.std_normalize: Union[None, float, Tuple[float, float, float]
            ] = Constants.STD_CHINESE_GRAYSCALE

        self.batch_size: int = 16
        self.use_cuda: bool = True


class ConfigTraining:
    def __init__(self):
        self.debug_mode: bool = False

        self.load_checkpoint: bool = False
        self.path_to_checkpoint: Optional[str] = None

        self.dump_checkpoints: bool = True
        self.checkpoint_root: Optional[str] = './traindir/checkpoints'
        self.checkpoint_template: Optional[str] = '{}_epoch_{}.cpkt'

        self.use_cuda: bool = True
        self.n_epochs: int = 30
        self.batch_size: int = 16
        self.n_classes: int = 58

        self.log_frequency: int = 100
        self.checkpoint_frequency: int = 5
        self.n_visualize: int = 6


class SetupConfig:
    def __init__(self) -> None:
        self.benchmark_config: ConfigBenchmark = ConfigBenchmark()
        self.network_config: ConfigNetwork = ConfigNetwork()
        self.training_config: ConfigTraining = ConfigTraining()
