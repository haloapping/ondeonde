import lightning as L
import torch.nn.functional as F
from typing import Any
from torch import Tensor, optim
from torchmetrics.functional import accuracy


class LightningModel(L.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        
    def forward(self, x) -> Any:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> Tensor:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=10)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx) -> None:
        features, true_labels = batch
        probs = self(features)
        loss = F.cross_entropy(probs, true_labels)
        acc = accuracy(probs.argmax(1), true_labels, task="multiclass", num_classes=10)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(self.parameters())
        
        return optimizer
