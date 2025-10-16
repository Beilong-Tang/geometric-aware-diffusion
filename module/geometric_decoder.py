import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils


class GeometricDiffusionDecoderOnly(L.LightningModule):
    def __init__(
        self,
        autoencoder: nn.Module,
    ):
        super().__init__()
        self.autoencoder = autoencoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        latent = self.autoencoder.encode(x)  # [B, D]
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
