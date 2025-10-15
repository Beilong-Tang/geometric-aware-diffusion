# from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.trainer import Trainer
# import torch
# import torchvision.utils as vutils
# import os

# from module.autoencoder_module import AutoEncoderModule


# class ImageGenerationCallback(Callback):
#     def __init__(self):
#         super().__init__()

#     def on_validation_epoch_end(self, trainer: Trainer, pl_module: AutoEncoderModule):
#         # Ensure the model is in evaluation mode
#         pl_module.eval()
#         print(trainer.global_rank)
#         if trainer.global_rank== 0:
#             with torch.no_grad():
#                 val_data = trainer.val_dataloaders
#                 for data in val_data:
#                     x, _ = data
#                     x = x.to(pl_module.device)
#                     out = pl_module.autoencoder(x)  # # [B, ]
#                     break
#             save_path = os.path.join(trainer.log_dir, "images")
#             os.makedirs(save_path, exist_ok=True)
#             target_path = os.path.join(
#                 save_path, f"epoch_{trainer.current_epoch}_target.png"
#             )
#             output_path = os.path.join(
#                 save_path, f"epoch_{trainer.current_epoch}_output.png"
#             )
#             vutils.save_image(x.cpu(), target_path, normalize=True)
#             vutils.save_image(out.cpu(), output_path, normalize=True)
#         torch.distributed.barrier()
#         pl_module.train()  # Set model back to training mode
