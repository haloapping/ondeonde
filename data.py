import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


class DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str="dataset", batch_size: int=64) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def prepare_data(self) -> None:
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage: str) -> tuple:
        self.mnist_test = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )
        
        self.mnist_predict = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=False
        )
        
        mnist_full = datasets.MNIST(
            self.data_dir, transform=transforms.ToTensor(), train=True
        )
        
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)
    