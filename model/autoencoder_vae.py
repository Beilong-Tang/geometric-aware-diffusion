import torch
from torch import nn

class VAE_CIFAR10(nn.Module):
    """
    A lightweight Variational Autoencoder (VAE) tailored for the CIFAR-10 dataset (32x32 images).

    The encoder compresses the input image into a latent space distribution, and the decoder
    reconstructs the image from a sample of that distribution.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Takes a [batch, 3, 32, 32] image and maps it to the latent space.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> [batch, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> [batch, 128, 4, 4]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0), # -> [batch, 256, 1, 1]
            nn.ReLU(),
            nn.Flatten(), # -> [batch, 256]
        )

        # The flattened feature size after the convolutional layers
        self.conv_output_size = 256

        # Two linear layers to output the mean (mu) and log-variance (log_var)
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_log_var = nn.Linear(self.conv_output_size, latent_dim)

        # --- Decoder ---
        # Takes a sample from the latent space [batch, latent_dim] and maps it back to an image.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.conv_output_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # Input will be reshaped to [batch, 256, 1, 1]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0), # -> [batch, 128, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> [batch, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # -> [batch, 3, 32, 32]
            nn.Sigmoid(), # Use Sigmoid to ensure output pixel values are in [0, 1]
        )

    def encode(self, x):
        """
        Encodes an input image into the mean and log-variance parameters.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, 32, 32].

        Returns:
            torch.Tensor: The mean of the latent distribution (mu).
            torch.Tensor: The log-variance of the latent distribution (log_var).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick: z = mu + epsilon * std.
        Allows gradients to flow through the sampling process.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes a latent vector z back into an image.
        """
        h = self.decoder_fc(z)
        # Reshape to match the input shape for the transposed convolutional layers
        h = h.view(-1, 256, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        """
        Full forward pass of the VAE.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var


def vae_loss_function(recon_x, x, mu, log_var):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    # Use Binary Cross Entropy for reconstruction loss, assuming input is normalized to [0,1]
    recon_loss = nn.functional.binary_cross_entropy(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')

    # KL divergence regularization
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kld


if __name__ == "__main__":
    x = torch.randn(5, 3, 32, 32)
    model =VAE_CIFAR10()
    mu, log_var = model.encode(x)
    out = model.decode(mu)
    print(f"emb shape {mu.shape}, out shape {out.shape}")
