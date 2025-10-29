##
import abc
import torch
import torch.nn as nn

from model.autoencoder import AbstractAutoEncoder
from model.geometric_aware.token_embed import GeometricEmbedding
from model.geometric_aware.transformer import (
    DecoderOnlyTransformer,
    GeometricDecoderOnly,
)
from model.diffusion import Diffusion


class AbstractDiffusionDecoderOnly(nn.Module, metaclass=abc.ABCMeta):
    """
    An abstract base class for PyTorch models with an abstract property.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def evaluate(self, z):
        pass

    @abc.abstractmethod
    def test(self, x0):
        pass


class GeometricDiffusionDecoderOnly(AbstractDiffusionDecoderOnly):
    def __init__(
        self,
        autoencoder: AbstractAutoEncoder,
        autoencoder_ckpt: str,
        geometric: GeometricEmbedding,
        decoder_only_transformer: DecoderOnlyTransformer,
        diffusion: Diffusion,
    ):
        super().__init__()
        # Here T stands for the total number of transitions
        self.autoencoder = autoencoder
        self.autoencoder.eval()

        self.geometric = geometric
        self.geometric.setup(autoencoder.get_emb_dim())

        self.diffusion = diffusion

        decoder_only_transformer.setup(d_model=geometric.get_token_emb_dim())
        self.geometric_decoder_only = GeometricDecoderOnly(
            decoder_only_transformer, T=self.diffusion.t_end
        )

        # autoencoder loading ckpt
        ckpt = torch.load(autoencoder_ckpt, map_location="cpu")
        autoencoder.load_state_dict(ckpt["state_dict"])
        # Freeze autoencoder
        for p in autoencoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W], raw images
        Returns:
            MSE loss
        """
        z = self.autoencoder.encode(x)  # [B, D]
        samples_img = self.diffusion.sample(z)  # [B, T+1, D]
        tokens = self.geometric(samples_img)  # [B, T+1, D]
        tokens = torch.flip(tokens, dims=(1,))  # [B, T+1, D]
        loss, result = self.geometric_decoder_only(tokens)
        return loss, result

    def evaluate(self, x):
        """
        Input:
            x: [B, C, H, W]: x0, the original clean image
        """
        loss, result = self.forward(x)

        final_emb = result["output"][:, -1]  # [B, D']
        latent = final_emb[:, : self.autoencoder.get_emb_dim()]
        img = self.autoencoder.decode(latent)  # [B, C, H, W]
        return loss, img

    def test(self, x0):
        z = self.autoencoder.encode(x0)
        samples_img = self.diffusion.sample(z)
        tokens = self.geometric(samples_img)
        xT_token = tokens[:, -1:] # [B, 1, D] the token corresponding to xT
        out = self.geometric_decoder_only.inference(xT_token) # [B, D]
        img = self.autoencoder.decode(out)
        return img


class VanillaDiffusionDecoderOnly(AbstractDiffusionDecoderOnly):
    def __init__(
        self,
        autoencoder: AbstractAutoEncoder,
        autoencoder_ckpt: str,
        decoder_only_transformer: DecoderOnlyTransformer,
        diffusion: Diffusion,
    ):
        super().__init__()
        # Here T stands for the total number of transitions
        self.autoencoder = autoencoder
        self.autoencoder.eval()

        self.diffusion = diffusion

        decoder_only_transformer.setup(d_model=self.autoencoder.get_emb_dim())
        self.geometric_decoder_only = GeometricDecoderOnly(
            decoder_only_transformer, T=diffusion.t_end
        )

        # autoencoder loading ckpt
        ckpt = torch.load(autoencoder_ckpt, map_location="cpu")
        autoencoder.load_state_dict(ckpt["state_dict"])
        # Freeze autoencoder
        for p in autoencoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W], raw images
        Returns:
            MSE loss
        """
        z = self.autoencoder.encode(x)  # [B, D]
        samples_img = self.diffusion.sample(z)  # [B, T+1, D]
        tokens = torch.flip(samples_img, dims=(1,))  # [B, T+1, D]
        loss, result = self.geometric_decoder_only(tokens)
        return loss, result

    def evaluate(self, x):
        """
        Input:
            x: [B, C, H, W]: x0, the original clean image
        """
        loss, result = self.forward(x)

        final_emb = result["output"][:, -1]  # [B, D']
        latent = final_emb[:, : self.autoencoder.get_emb_dim()]
        img = self.autoencoder.decode(latent)  # [B, C, H, W]
        return loss, img

    def test(self, x0):
        z = self.autoencoder.encode(x0)
        samples_img = self.diffusion.sample(z)
        xT_token = samples_img[:, -1:] # [B, 1, D] the token corresponding to xT
        out = self.geometric_decoder_only.inference(xT_token) # [B, D]
        img = self.autoencoder.decode(out)
        return img