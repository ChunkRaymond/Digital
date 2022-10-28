# writer : shiyu
# code time : 2022/10/25
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_mnist(config, Data_loader, lightnting_model):

    train_set = Data_loader['train_set']
    val_set = Data_loader['val_set']
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    tunner = TuneReportCallback(metrics, on="validation_end")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    Callbacks = [tunner, lr_monitor]

    epoches = 2

    # Train
    trainer = pl.Trainer(max_epochs=epoches,
                         profiler="simple",
                         default_root_dir="./",
                         gpus=-1,
                         auto_select_gpus=True,
                         callbacks=Callbacks,
                         enable_progress_bar=False
                         # enable_progress_bar=False
                         )
    trainer.fit(model=lightnting_model,
                train_dataloaders=train_set,
                val_dataloaders=val_set,
                )
