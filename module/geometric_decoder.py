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
                save_path, f"epoch_{self.trainer.current_epoch}_target_last.png"
            )
            infer_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_target_infer.png"
            )
            encoded_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_encoded.png"
            )
            output_path = os.path.join(
                save_path, f"epoch_{self.trainer.current_epoch}_output.png"
            )
            vutils.save_image(x.cpu(), target_path, normalize=True)
            vutils.save_image(img.cpu(), output_path, normalize=True)
            # Run an inference
            img, result = self.geometric_decoder_only.test(x)
            vutils.save_image(img.cpu(), infer_path, normalize = True)
            vutils.save_image(result['encoded'].cpu(), encoded_path, normalize = True)


    def test_step(self, batch, batch_idx):
        rank = self.trainer.global_rank
        root_dir = self.trainer.log_dir
        target_dir = os.path.join(root_dir, "target")
        encoded_dir = os.path.join(root_dir, "encoded")
        output_dir = os.path.join(root_dir, "output")
        for p in [target_dir, encoded_dir, output_dir]:
            os.makedirs(p, exist_ok=True)
        with torch.no_grad():
            x, _ = batch
            img, result = self.geometric_decoder_only.test(x)  # [B, C, H, W]
        for i, _img in enumerate(result["encoded"]):
            vutils.save_image(_img, f"{encoded_dir}/{batch_idx}_{i}_r{rank}.png")
        for i, _img in enumerate(img):
            vutils.save_image(_img, f"{output_dir}/{batch_idx}_{i}_r{rank}.png")
        for i, _img in enumerate(x):
            vutils.save_image(_img, f"{target_dir}/{batch_idx}_{i}_r{rank}.png")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0e-4)
        return optimizer
