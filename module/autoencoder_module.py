import lightning as L
import torch.optim as optim
import torch.nn.functional as F

class AutoEncoderModule(L.LightningModule):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder= autoencoder
        print(self.autoencoder.img_size)

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
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer