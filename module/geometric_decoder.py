import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.utils as vutils

from model.autoencoder import AbstractAutoEncoder
from model.geometric_aware.token_embed import GeometricEmbedding


class GeometricDiffusionDecoderOnly(L.LightningModule):
    def __init__(self, autoencoder: AbstractAutoEncoder, geometric: GeometricEmbedding):
        super().__init__()
        self.autoencoder = autoencoder
        self.geometric = geometric
        self.geometric.setup(in_features=autoencoder.get_emb_dim())

    def training_step(self, batch, batch_idx):
        x, _ = batch
        latent = self.autoencoder.encode(x)  # [B, D]
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
