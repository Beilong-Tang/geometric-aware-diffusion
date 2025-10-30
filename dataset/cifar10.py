from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import lightning as L


class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        pass

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor()])
        if stage == "fit":
            self.trainset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            self.valset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        if stage == "test":
            self.valset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
