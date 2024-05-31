import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import Tensor
from hydra.utils import instantiate


class Solver(pl.LightningModule):
    def __init__(self, model, config: DictConfig) -> None:
        super(Solver, self).__init__()
        self.model = model
        self.config = config
        self.loss = instantiate(self.config.optimizer.loss)
        self.metric = instantiate(self.config.optimizer.metrics)
        self.best_epoch = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = instantiate(
            self.config.optimizer.optimizer,
            params=params
        )

        scheduler = instantiate(
            self.config.optimizer.scheduler,
            optimizer=optimizer
        )

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val/loss'
        }

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss(pred, target)

    @torch.no_grad()
    def metric_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.metric(pred, target)

    def training_step(self, batch: Tensor, batch_idx: int):
        image, _, target_regr = batch
        pred = self.model(image)
        loss = self.loss_fn(pred, target_regr)
        metrics = self.metric_fn(pred, target_regr)

        self.log("train/loss", loss)
        return {
            "loss": loss,
            "metrics": metrics
        }

    def training_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([
            output['metrics']["l1_loss"] for output in outputs
        ])
        self.log("train/l1_loss_mean", torch.mean(l1_losses))
        self.log("train/l1_loss_std", torch.std(l1_losses))

    def validation_step(self, batch: Tensor, batch_idx: int):
        image, _, target_regr = batch
        pred = self.model(image)
        loss = self.loss_fn(pred, target_regr)
        metrics = self.metric_fn(pred, target_regr)
        self.log("val/loss", loss)

        return {
            "loss": loss,
            "metrics": metrics
        }

    def test_step(self, batch: Tensor, batch_idx: int):
        image, _, target_regr = batch
        pred =  self.model(image)
        loss = self.loss_fn(pred, target_regr)
        metrics = self.metric_fn(pred, target_regr)

        self.log("test/loss", loss)

        return {"metrics": metrics, "loss": loss}

    def test_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([output['metrics']["l1_loss"] for output in outputs])
        self.log("test/l1_loss_mean", torch.mean(l1_losses))
        self.log("test/l1_loss_std", torch.std(l1_losses))

    def validation_epoch_end(self, outputs) -> None:
        l1_losses = torch.tensor([output['metrics']["l1_loss"] for output in outputs])
        self.best_epoch.append(torch.mean(l1_losses))
        self.log("val/l1_loss_mean", torch.mean(l1_losses))
        self.log("val/l1_loss_std", torch.std(l1_losses))
        self.log("val/best_loss", min(self.best_epoch))


    def fit(self):
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.trainer.fit(
            self,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
