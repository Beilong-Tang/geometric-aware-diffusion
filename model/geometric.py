## 
import abc
import torch.nn as nn

from model.autoencoder import AbstractAutoEncoder
from model.geometric_aware.token_embed import GeometricEmbedding

class AbstractGeometricDiffusion(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, autoencoder:AbstractAutoEncoder, geometic:GeometricEmbedding):
        super().__init__()
    
    @abc.abstractmethod
    def sample(self, x0):
        ## Given x0, sample points using DDPM forward
        pass

class GeometricDiffusionDecoderOnly(AbstractGeometricDiffusion):
    def __init__(self):
        super().__init__()

    def sample(self, x0):
        pass

    def forward(self):
        pass