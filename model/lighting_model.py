# writer : shiyu
# code time : 2022/10/23

import math

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy

import pytorch_lightning as pl

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU}


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config, basic_model):
        super().__init__()
        self.basic_model = basic_model
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.lr = config['lr']
        self.momentum = config['momentum']

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.basic_model(x)

        loss = F.mse_loss(z, y)
        accuracy = self.train_acc(z, y.int())

        self.log('ptl/train_acc', accuracy, on_step=False, on_epoch=True)
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
    #     avg_acc = torch.stack([x["train_accuracy"] for x in outputs]).mean()
    #     self.log("ptl/train_loss", avg_loss)
    #     self.log("ptl/train_accuracy", avg_acc)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.basic_model(x)
        loss = F.mse_loss(z, y)
        accuracy = self.valid_acc(z, y.int())

        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        self.log('ptl/val_lr', self.lr)
        self.log('ptl/momentum', self.momentum)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, nesterov=False)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-1 / 60))
        return [optimizer], [lr_scheduler]
