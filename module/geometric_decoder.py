import lightning as L
import torch
import torchvision.utils as vutils
import os

from model.geometric import AbstractDiffusionDecoderOnly

class GeometricDiffusionDecoderOnlyModule(L.LightningModule):
    def __init__(self, geometric_decoder_only: AbstractDiffusionDecoderOnly):
        super().__init__()
        self.geometric_decoder_only = geometric_decoder_only

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, _ = self.geometric_decoder_only(x)
        self.log("mse", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss, img = self.geometric_decoder_only.evaluate(x)
        self.log(
            "val_mse",
            loss,
        )
        if batch_idx == 0:
            # Save the output to a folder
            save_path = os.path.join(self.trainer.log_dir, "images")
            os.makedirs(save_path, exist_ok=True)
            target_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_target.png"
            )
            output_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_output.png"
            )
            vutils.save_image(x.cpu(), target_path, normalize=True)
            vutils.save_image(img.cpu(), output_path, normalize=True)
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-4)
        return optimizer
