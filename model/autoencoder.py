import abc
import torch.nn as nn
import torch


class AbstractAutoEncoder(nn.Module, metaclass=abc.ABCMeta):
    """
    An abstract base class for PyTorch models with an abstract property.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_emb_dim(self) -> int:
        pass

    @abc.abstractmethod
    def encode(self, x):
        pass
    
    @abc.abstractmethod
    def decode(self,z):
        pass


class Autoencoder(AbstractAutoEncoder):
    def __init__(self, img_size=32, emb_dim=128):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.img_size = img_size
        self.emb_dim = emb_dim
        self.latent_dim = int(96 * (img_size / 16) ** 2)
        self.latent_feature_size = int(self.latent_dim / 96 / 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            # nn.ReLU(),
            # nn.Flatten(),  # [batch, 96*2*2]
            # nn.Linear(self.latent_dim, emb_dim),  # [batch, emb_dim]
            # nn.Sigmoid()
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(emb_dim, self.latent_dim), nn.ReLU()  # [batch, 96 * 2 * 2]
        # )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W] -> [B, 3, H, W]
        bb = x.size(0)
        x = self.encoder(x)  # [B, emb_dim]
        # x = self.mlp(x) # [B, ]
        # x = x.view(
        #     bb,
        #     int(self.latent_dim / self.latent_feature_size**2),
        #     self.latent_feature_size,
        #     self.latent_feature_size,
        # )
        x = self.decoder(x)
        return x

    def encode(self, x):
        # [B, 3, H, W] -> [B, emb_dim]
        # raw images to latent embedding
        x = self.encoder(x)  # [B, emb_dim]
        return x

    def decode(self, x):
        # [B, emb_dim]  -> [B, 3, H, W]
        # latent embedding to raw images
        # x = self.mlp(x)
        # bb = x.size(0)
        # x = x.view(
        #     bb,
        #     int(self.latent_dim / self.latent_feature_size**2),
        #     self.latent_feature_size,
        #     self.latent_feature_size,
        # )
        x = self.decoder(x)
        return x

    def get_emb_dim(self):
        return self.emb_dim


if __name__ == "__main__":
    x = torch.randn(5, 3, 32, 32)
    model = Autoencoder()
    emb = model.encode(x)
    out = model.decode(emb)
    out1 = model(x)
    print(f"emb shape {emb.shape}, out shape {out.shape}, out1 shape {out1.shape}")
