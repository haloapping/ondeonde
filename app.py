from data import DataModule
from model import Model
from lightning_model import LightningModel
from lightning.pytorch.loggers import CSVLogger

import lightning as L
from datetime import datetime


if __name__ == "__main__":
    # load data
    dm = DataModule()

    # model
    model = Model()
    lightning_model = LightningModel(model)

    # trainer
    pathname = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="cpu",
        deterministic=True,
        logger=CSVLogger(save_dir="logs/", name=pathname)
    )

    trainer.fit(
        model=lightning_model,
        datamodule=dm,
    )