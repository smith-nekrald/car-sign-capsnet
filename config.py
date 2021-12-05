from typing import Optional


class SetupConfig:
    def __init__(self):
        self.benchmark: Optional[str] = 'chinese'
        self.image_transform: Optional[str] = 'grayscale'
        self.use_cuda: Optional[bool] = True
        self.conv_in_channels: Optional[int] = 1
        self.conv_out_channels: Optional[int] = 256
        self.conv_kernel_size: Optional[int] = 9

        self.primary_num_capsules: Optional[int] = 8
        self.primary_in_channels: Optional[int] = 256
        self.primary_out_channels: Optional[int] = 32
        self.primary_kernel_size: Optional[int] = 9
        self.primary_num_routes: Optional[int] = 32 * 6 * 6
        self.primary_eps_denom: Optional[float] = 1e-5
        self.primary_eps_sqrt: Optional[float] = 1e-6

        self.recognition_num_classes: Optional[int] = 58
        self.recognition_num_routes: Optional[int] = 32 * 6 * 6
        self.recognition_in_channels: Optional[int] = 8
        self.recognition_out_channels: Optional[int] = 16
        self.recognition_routing_iterations: Optional[int] = 3
        self.recognition_eps_denom: Optional[float] = 1e-5
        self.recognition_eps_sqrt: Optional[float] = 1e-6

        self.net_margin_loss_right_blend: Optional[float] = 0.5
        self.net_margin_loss_square: Optional[bool] = True
        self.net_reconstruction_loss_reg: Optional[float] = 0.0005
        self.net_margin_left: Optional[float] = 0.9
        self.net_margin_right: Optional[float] = 0.1

        self.optimizer: Optional[str] = 'adam'
        self.n_epochs: Optional[int] = 30
        self.batch_size: Optional[int] = 16
