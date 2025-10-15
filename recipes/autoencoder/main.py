import os
import sys

sys.path.append(os.getcwd())

from module.autoencoder_module import AutoEncoderModule
from dataset.cifar10 import Cifar10DataModule
from lightning.pytorch import seed_everything
seed_everything(42)

from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
