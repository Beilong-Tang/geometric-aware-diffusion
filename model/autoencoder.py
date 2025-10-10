import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    def __init__(self, img_size=32, emb_dim=128):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.img_size = img_size
        self.emb_dim = emb_dim
        self.latent_dim = 96 * (img_size / 16) ** 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.Flatten(),  # [batch, 96*2*2]
            nn.Linear(self.latent_dim, emb_dim),  # [batch, emb_dim]
        )

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, self.latent_dim), # [batch, 96 * 2 * 2]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W] -> [B, 3, H, W]
        x_shape = x.shape
        x = self.encoder(x)  # [B, emb_dim]
        x = self.mlp(x) 
        x = x.view(x_shape)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        # [B, 3, H, W] -> [B, emb_dim]
        # raw images to latent embedding
        x = self.encode(x) # [B, emb_dim]
        return x

    def decode(self, x):
        # [B, emb_dim]  -> [B, 3, H, W]
        # latent embedding to raw images
        x = self.mlp(x)
        x = self.decoder(x)
        return x



# def main():
#     parser = argparse.ArgumentParser(description="Train Autoencoder")
#     parser.add_argument("--valid", action="store_true", default=False,
#                         help="Perform validation only.")
#     args = parser.parse_args()

#     # Create model
#     autoencoder = create_model()

#     # Load data
#     transform = transforms.Compose(
#         [transforms.ToTensor(), ])
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
#                                               shuffle=True, num_workers=2)
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                            download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=16,
#                                              shuffle=False, num_workers=2)
#     classes = ('plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#     if args.valid:
#         print("Loading checkpoint...")
#         autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
#         dataiter = iter(testloader)
#         images, labels = dataiter.next()
#         print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
#         imshow(torchvision.utils.make_grid(images))

#         images = Variable(images.cuda())

#         decoded_imgs = autoencoder(images)[1]
#         imshow(torchvision.utils.make_grid(decoded_imgs.data))

#         exit(0)

#     # Define an optimizer and criterion
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(autoencoder.parameters())

#     import tqdm

#     pbar = tqdm.tqdm(trainloader)

#     for epoch in range(100):
#         running_loss = 0.0
#         for i, (inputs, _) in enumerate(pbar, 0):
#             inputs = get_torch_vars(inputs)

#             # ============ Forward ============
#             encoded, outputs = autoencoder(inputs)
#             loss = criterion(outputs, inputs)
#             # ============ Backward ============
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             pbar.set_postfix(loss=f"{loss.item():.3f}")

#             # ============ Logging ============
#             running_loss += loss.data
#             if i % 2000 == 1999:
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0

#     print('Finished Training')
#     print('Saving Model...')
#     if not os.path.exists('./weights'):
#         os.mkdir('./weights')
#     torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")


# if __name__ == '__main__':
#     main()
