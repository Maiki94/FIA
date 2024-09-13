import os
import pandas as pd
from typing import Optional, Any

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Recall,
    Precision,
)
from pytorch_lightning import LightningModule

from fia.nn.classifier import FraudDetectionModel


class LightningFraudClassifier(LightningModule):
    model: FraudDetectionModel
    metrics: MetricCollection
    critirion: _Loss

    def __init__(
        self,
        model: FraudDetectionModel,
        num_classes: int,
        *args,
        metrics: MetricCollection | None = None,
        criterion: _Loss | None = None,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        metrics = metrics or self.initialize_metrics(num_classes=num_classes)

        self.metrics = metrics
        self.model = model
        self.criterion = criterion

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def initialize_metrics(self, num_classes: int) -> MetricCollection:

        metrics = MetricCollection(
            {
                "accuracy_weighted": Accuracy(
                    average="weighted", task="binary", num_classes=num_classes
                ),
                "recall_weighted": Recall(
                    average="weighted", task="binary", num_classes=num_classes
                ),
                "precision_weighted": Precision(
                    average="weighted",
                    task="binary",
                    num_classes=num_classes,
                ),
            }
        )

        return metrics

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        inputs, targets = batch
        logits = self.forward(inputs).squeeze()
        targets = targets.long()

        loss = self.criterion(logits, targets.float())
        probs = torch.sigmoid(logits)

        return loss, probs, targets

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        loss, probs, targets = self.step(batch=batch)
        self.log(name="train_loss", value=loss, on_step=False, on_epoch=True)
        self.train_metrics.update(probs, targets)

        return loss
    
    def on_train_epoch_end(self):
        # Compute and log metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        loss, probs, targets = self.step(batch=batch)
        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True)
        self.val_metrics.update(probs, targets)
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        loss, probs, targets = self.step(batch=batch)
        self.log(name="test_loss", value=loss, on_step=False, on_epoch=True)

        self.test_metrics.update(probs, targets)

        return {
            "loss": loss,
            "probs": probs.detach(),
            "target": targets,
        }
    
    def on_test_epoch_end(self):
        # Compute and log metrics
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True)
        self.test_metrics.reset()


    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters())