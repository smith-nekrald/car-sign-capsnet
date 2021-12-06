from config import SetupConfig
from train import do_training


def perform_launches():
    config: SetupConfig = SetupConfig()
    do_training(config)


