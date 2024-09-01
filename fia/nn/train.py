from pathlib import Path
from datetime import datetime

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from fia.nn.classifier import FraudDetectionModel
from fia.nn.model import LightningFraudClassifier
from fia.nn.data import DataModule

from typing import List, Tuple

import gc
from pytorch_lightning.callbacks import Callback



class TrainerManager:
    def __init__(
        self,
        pl_model: LightningFraudClassifier,
        pl_datamodule: DataModule,
        run_datadir: str = f"./model_trainer",
    ):

        self.pl_model = pl_model
        self.pl_datamodule = pl_datamodule
        self.run_datadir = Path(run_datadir)

        self._logger = MLFlowLogger(
            experiment_name="FraudDetection",
            tracking_uri=str(self.run_datadir / "mlflow"),
            run_name=datetime.now().strftime("%Y%m%d_%H%M"),
        )

        self._callback_list = [
            EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=Path(self.run_datadir, "checkpoints"),
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]
        self.trainer = None

    @property
    def logger(self) -> TensorBoardLogger:
        return self._logger

    @property
    def callback(self) -> List[Callback]:
        return self._callback_list

    @classmethod
    def set_seed(cls, seed: int = 42):
        seed_everything(seed=seed)

    def train(
        self, epochs: int = 10, use_gpu: bool = False
    ) -> Tuple[FraudDetectionModel, dict]:

        self.set_seed()

        self.trainer = Trainer(
            logger=self.logger,
            callbacks=self.callback,
            max_epochs=epochs,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            devices="auto",
            accelerator="gpu" if use_gpu else "cpu",
            accumulate_grad_batches=1,
        )

        self.trainer.fit(model=self.pl_model, datamodule=self.pl_datamodule)

        return self.pl_model, self.trainer.logged_metrics
    
    def load_best_model(self) -> FraudDetectionModel:
        """
        Load the best model from the checkpoint.
        """
        checkpoint_path = self._callback_list[1].best_model_path  # The ModelCheckpoint is the second callback in the list
        if checkpoint_path == "":
            raise ValueError("No checkpoint found. Ensure that training has been completed and a checkpoint has been saved.")
        
        best_model = LightningFraudClassifier.load_from_checkpoint(checkpoint_path)
        return best_model

    def test(self):

        if self.trainer is None:
            raise ValueError("The model has not been trained, Please call train first")

        results = self.trainer.test(dataloaders=self.pl_datamodule, ckpt_path="best")

        return results
