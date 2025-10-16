import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import torchvision.utils as vutils


class GeometricDiffusionDecoderOnly(L.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
