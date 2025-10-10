import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import os

class ImageGenerationCallback(pl.Callback):
    def __init__(self, num_images=4, log_dir="generated_images"):
        super().__init__()
        self.num_images = num_images
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure the model is in evaluation mode
        pl_module.eval()
        with torch.no_grad():
            # Get a batch of real images from the validation dataloader
            # Or generate images from a latent space if it's a generative model
            # Example for a generative model:
            z = torch.randn(self.num_images, pl_module.latent_dim, device=pl_module.device)
            generated_images = pl_module.generator(z) # Assuming 'generator' is a method in your LightningModule

            # Example if you want to visualize real images from the validation set:
            # val_dataloader = trainer.val_dataloaders[0] # Get the first validation dataloader
            # data_iter = iter(val_dataloader)
            # real_images, _ = next(data_iter)
            # real_images = real_images[:self.num_images].to(pl_module.device)
            # images_to_save = real_images # Or generated_images

            # Normalize images to [0, 1] if needed (e.g., if output is tanh-activated)
            generated_images = (generated_images + 1) / 2 # Example for tanh output

            # Save the images
            epoch = trainer.current_epoch
            save_path = os.path.join(self.log_dir, f"epoch_{epoch:03d}.png")
            vutils.save_image(generated_images, save_path, normalize=True)

        pl_module.train() # Set model back to training mode