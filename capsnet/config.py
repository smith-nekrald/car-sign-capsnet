""" Specifies configs for entire experiment and provides default values. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2021
# (c) Smith School of Business, 2023

from typing import Optional
from typing import Union
from typing import Tuple
from typing import List

from keys import BenchmarkName
from keys import ColorSchema
from keys import Constants


class ConfigSquash:
    """ Configures squashing module. 

    Attributes:
        eps_denom: Shift for denominator for numeric stability.
        eps_sqrt: Shift for square root for numeric stability.
        eps_input: Shift for input for numeric stability.
        eps_norm: Shift for norm for numeric stability.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.eps_denom: float = 1e-5
        self.eps_sqrt: float = 1e-6
        self.eps_input: float = 1e-5
        self.eps_norm: float = 1e-5


class ConfigConv:
    """ Configures convolution module.

    Attributes:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The kernel size.
        use_batch_norm: Whether to use batch normalization.
        stride: The stride value.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.in_channels: int = 1
        self.out_channels: int = 64
        self.kernel_size: int = 9
        self.use_batch_norm: bool = True
        self.stride: int = 1


class ConfigPrimary:
    """ Configures primary capsules. 

    Attributes:
        num_capsules: The number of primary capsules.
        in_conv_channels: The number of input convolution channels.
        out_conv_channels: The number of output convolution channels.
        conv_kernel_size: The size of convolution kernel.
        conv_stride: The stride of convolution kernel.
        conv_padding: The convolution padding.
        capsule_output_dim: The output dimension of each capsule.
        use_dropout: Whether to use dropout.
        dropout_proba: Dropout probability.
        use_nan_gradient_hook: Whether to use the NaN gradient hook.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.num_capsules: int = 12
        self.in_conv_channels: int = 64
        self.out_conv_channels: int = 24
        self.conv_kernel_size: int = 9
        self.conv_stride: int = 2
        self.conv_padding: int = 0
        self.capsule_output_dim: int = 24 * 6 * 6
        self.use_dropout: bool = True
        self.dropout_proba: float = 0.4
        self.use_nan_gradient_hook: bool = False


class ConfigAgreement:
    """ Configures agreement routing. 

    Attributes:
        num_input_caps: The number of input capsules.
        num_output_caps: The number of output capsules.
        n_iterations: The number of routing cycles.
        output_caps_dim: The dimension of output capsules.
        use_cuda: Whether to use CUDA.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.num_input_caps: int = 12
        self.num_output_caps: int = 58
        self.n_iterations: int = 3
        self.output_caps_dim: int = 16
        self.use_cuda: bool = True


class ConfigRecognition:
    """ Configures recognition capsules. 
    
    Attributes:
        num_output_caps: The number of the output capsules.
        output_caps_dim: The dimensionality of the output capsules.
        num_input_caps: The number of input capsules.
        input_caps_dim: The dimensionality of the input capsules.
        use_dropout: Whether to use dropout.
        dropout_proba: The probability of dropout.
        use_nan_gradient_hook: Whether to use NaN gradient hook.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.num_output_caps: int = 58
        self.output_caps_dim: int = 16
        self.num_input_caps: int = 12
        self.input_caps_dim: int = 24 * 6 * 6
        self.use_dropout: bool = True
        self.dropout_proba: float = 0.4
        self.use_nan_gradient_hook: bool = False


class ConfigReconstruction:
    """ Configures reconstruction network. 

    Attributes:
        linear_input_dim: Input dimensionality.
        linear_hidden_layers: List with sizes of hidden layers.
        linear_ouptut_dim: Output dimensionality.
        num_classes: The number of classes.
        use_cuda: Whether to use CUDA.
        output_img_size: The size of ouptut image.
        output_n_channels: The number of output channels.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.linear_input_dim: int = 16 * 58
        self.linear_hidden_layers: List[int] = [512, 1024]
        self.linear_output_dim: int = 1 * 28 * 28
        self.num_classes: int = 58
        self.use_cuda = True
        self.output_img_size: Tuple[int, int] = (28, 28)
        self.output_n_channels: int = 1


class ConfigNetwork:
    """ Configures Capsule Network. 
    
    Attributes:
        conv_config: Configuration for initial convolution layer.
        primary_config: Configuration for primary capsules.
        squash_config: Configuration for squashing module.
        agreement_config: Configuration for agreement routing.
        recognition_config: Configuration for recognition capsules.
        reconstruction_config: Configuration for reconstruction network.
        net_reconstruction_loss_reg: Regularizer coefficient for reconstruction loss.
        net_margin_loss_blend: Blending coefficient inside margin loss.
        net_margin_upper: Upper margin value.
        net_margin_lower: Lower maring value.
        use_square_in_margin_loss: Whether to use squares in margin loss.
    """
    def __init__(self) -> None:
        """ Initializer method. """
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
    """ Configures benchmarks. 
    
    Attributes:
        num_load_workers: Number of loading workers for DataLoader.
        benchmark: The name of the benchmark.
        image_size: The size of the image.
        image_color: The image color model.
        num_channels: The number of image channels.
        use_augmentation: Whether to use augmentation.
        augment_proba: Probability to augment.
        random_entry_proba: Probability to apply each transformation in the sequence.
        estimate_normalization: Whether to estimate normalization.
        n_point_to_estimate: Number of points to use for normalization estimation.
        mean_normalize: Mean value for image normalization.
        std_normalize: Standard deviation for image normalization.
        batch_size: The batch size.
        use_cuda: Whether to use CUDA.
    """
    def __init__(self) -> None:
        """ Initializer method. """
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
    """ Configures network training process. 

    Args:
        debug_mode: Whether the launch is for debugging. Debug mode makes process slower 
            but outputs more in-process information.
        load_checkpoint: Whether to load checkpoint. 
        path_to_checkpoint: Path to checkpoint for loading.
        dump_checkpoints: Whether to dump checkpoints during training process.
        checkpoint_root: Path to checkpoint root.
        checkpoint_template: The template for saving checkpoint files.
        use_cuda: Whether to use CUDA.
        n_epochs: Number of training epochs.
        batch_size: The size of training batch.
        n_classes: Number of classes.
        use_clipping: Whether to use gradient clipping.
        clipping_threshold: Gradient clipping threshold.
        log_frequency: The frequency of logging.
        checkpoint_frequency: The frequency of creating checkpoints.
        n_visualize: The number of reconstructed images to visualize.
        graph_to_tensorboard: Whether to save graph to TensorBoard.
        use_lime: Whether to use LIME and provide explanations.
    """
    def __init__(self) -> None:
        """ Initializer method. """
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

        self.use_clipping: bool = False
        self.clipping_threshold: Optional[float] = 100.

        self.log_frequency: int = 100
        self.checkpoint_frequency: int = 10
        self.n_visualize: int = 6

        self.graph_to_tensorboard: bool = True
        self.use_lime: bool = True


class SetupConfig:
    """ Configures the entire experiment. 

    Attributes:
        benchmark_config: The benchmark configuration.
        network_config: The capsule network configuration.
        training_config: The training process configuration.
    """
    def __init__(self) -> None:
        """ Initializer method. """
        self.benchmark_config: ConfigBenchmark = ConfigBenchmark()
        self.network_config: ConfigNetwork = ConfigNetwork()
        self.training_config: ConfigTraining = ConfigTraining()

