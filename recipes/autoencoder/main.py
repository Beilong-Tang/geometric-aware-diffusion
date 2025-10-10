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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_path", default=None, type=str)
    # parser.add_argument("--root_dir", default=None, type=str)
    # parser.add_argument("--devices", default=4, type=int)
    # args = parser.parse_args()
    # main(args)
    cli_main()

from lightning.pytorch.demos.boring_classes import DemoModel
