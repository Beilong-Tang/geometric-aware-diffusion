from model.autoencoder import AbstractAutoEncoder
import lightning as L
import torch.optim as optim
import torch.nn.functional as F
import os
import torchvision.utils as vutils


class AutoEncoderModule(L.LightningModule, AbstractAutoEncoder):
    def __init__(self, autoencoder: AbstractAutoEncoder, test_output_path):
        super().__init__()
        self.autoencoder = autoencoder

    def training_step(self, batch, batch_idx):
        # batch: []
        x, _ = batch
        out = self.autoencoder(x)
        loss = F.binary_cross_entropy(out, x)
        self.log("bce", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self.autoencoder(x)
        loss = F.binary_cross_entropy(out, x)
        self.log("val_bce", loss)

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
            vutils.save_image(x.cpu(), target_path, normalize=True)
            vutils.save_image(out.cpu(), output_path, normalize=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, x):
        return self.autoencoder.decode(x)

    def get_emb_dim(self):
        return self.autoencoder.get_emb_dim()
