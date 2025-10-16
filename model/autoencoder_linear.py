import torch.nn as nn
import torch
from model.autoencoder import AbstractAutoEncoder


class Autoencoder(AbstractAutoEncoder):
    def __init__(self, img_size=32, emb_dim=128):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.img_size = img_size
        self.emb_dim = emb_dim
        self.latent_dim = int(48 * (img_size / 8) ** 2)  #
        self.latent_feature_size = int(self.latent_dim / 48 / 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            # nn.ReLU(),
            nn.Flatten(),  # [batch, 96*2*2]
            nn.Linear(self.latent_dim, emb_dim),  # [batch, emb_dim]
            # nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, self.latent_dim), nn.ReLU()  # [batch, 96 * 2 * 2]
        )

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
        x = self.mlp(x)  # [B, ]
        x = x.view(
            bb,
            int(self.latent_dim / self.latent_feature_size**2),
            self.latent_feature_size,
            self.latent_feature_size,
        )
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
        x = self.mlp(x)
        bb = x.size(0)
        x = x.view(
            bb,
            int(self.latent_dim / self.latent_feature_size**2),
            self.latent_feature_size,
            self.latent_feature_size,
        )
        x = self.decoder(x)
        return x

    def get_emb_dim(self):
        return self.emb_dim


class AdvancedAutoencoder(AbstractAutoEncoder):
    """
    An improved autoencoder structure incorporating Batch Normalization for better
    training stability and performance.
    """

    def __init__(self, img_size=32, emb_dim=128):
        super().__init__()
        if img_size != 32:
            raise ValueError("This specific architecture is designed for 32x32 images.")

        self.emb_dim = emb_dim

        # --- Encoder ---
        # Each convolutional layer is now followed by BatchNorm2d for stabilization.
        # [B, 3, 32, 32] -> [B, emb_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # -> [B, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # -> [B, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # -> [B, 48, 4, 4]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Flatten(),  # -> [B, 48 * 4 * 4 = 768]
            nn.Linear(48 * 4 * 4, emb_dim),  # -> [B, emb_dim]
        )

        # --- Decoder ---
        # The decoder mirrors the encoder's structure, including BatchNorm.
        # [B, emb_dim] -> [B, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 48 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (48, 4, 4)),  # -> [B, 48, 4, 4]
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # -> [B, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # -> [B, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # -> [B, 3, 32, 32]
            # The final layer uses Sigmoid to output pixel values between 0 and 1.
            # No BatchNorm or ReLU is applied here.
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encodes an image into a latent vector."""
        return self.encoder(x)

    def decode(self, z):
        """Decodes a latent vector back into an image."""
        return self.decoder(z)

    def forward(self, x):
        """The full forward pass: encode the image and then decode it."""
        z = self.encode(x)
        return self.decode(z)

    def get_emb_dim(self):
        return self.emb_dim


if __name__ == "__main__":
    x = torch.randn(5, 3, 32, 32)
    model = Autoencoder()
    emb = model.encode(x)
    out = model.decode(emb)
    out1 = model(x)
    print(f"emb shape {emb.shape}, out shape {out.shape}, out1 shape {out1.shape}")
