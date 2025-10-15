import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import torchvision.utils as vutils

class AutoEncoderModule(L.LightningModule):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder= autoencoder

    def training_step(self, batch, batch_idx):
        # batch: []
        x, _ = batch
        reconstruction, mu, log_var = self.autoencoder(x)
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld
        self.log("bce", recon_loss, prog_bar=True)
        self.log("kld", kld, prog_bar=True)
        self.log("loss,", loss, prog_bar=True )
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        reconstruction, mu, log_var = self.autoencoder(x)
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld
        self.log("val_bce", recon_loss, prog_bar=True)
        self.log("val_kld", kld, prog_bar=True)
        self.log("val_loss,", loss, prog_bar=True)

        if batch_idx == 0:
            # save an output sample
            save_path = os.path.join(self.trainer.log_dir, "images")
            os.makedirs(save_path, exist_ok=True)
            target_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_target.png"
            )
            output_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_output.png"
            )
            out = self.autoencoder.decode(mu)
            vutils.save_image(x.cpu(), target_path, normalize=True)
            vutils.save_image(out.cpu(), output_path, normalize=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer