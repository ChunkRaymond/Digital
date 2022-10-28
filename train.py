# writer : shiyu
# code time : 2022/10/27
import argparse
import parser
import torch

import torch.utils.data as data
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

from model.base_model import BaseNetwork
from model.lighting_model import LitAutoEncoder
from utils.Dataset import MyDataset, load_data


def train_mnist(config,
                max_epoch=None,
                train_set=None,
                val_set=None,
                show_bar=False,
                Callbacks=None):
    model = LitAutoEncoder(config, basic_model=BaseNetwork())

    trainer = pl.Trainer(max_epochs=max_epoch,
                         callbacks=Callbacks,
                         enable_progress_bar=show_bar,
                         logger=TensorBoardLogger(
                             save_dir=tune.get_trial_dir(), name="", version="."),
                         )

    trainer.fit(model=model,
                train_dataloaders=train_set,
                val_dataloaders=val_set,
                )


def run(if_tune=False):
    x_train, y_train, x_test = load_data('./data')
    train_set = MyDataset(x_train, y_train)
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_set = DataLoader(train_set, batch_size=32)
    val_set = DataLoader(val_set, batch_size=32)

    if if_tune:
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.choice([0.8, 0.7, 0.6]),
        }

        tune_report = TuneReportCallback(
            {
                "loss": "ptl/val_loss",
                "mean_accuracy": "ptl/val_accuracy"
            },
            on="validation_end")

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        reporter = CLIReporter(
            parameter_columns=["lr", "momentum"],
            metric_columns=["loss", "mean_accuracy"])
        Callbacks = [tune_report, lr_monitor]

        train_fn_with_parameters = tune.with_parameters(train_mnist,
                                                        max_epoch=10,
                                                        train_set=train_set,
                                                        val_set=val_set,
                                                        show_bar=False,
                                                        Callbacks=Callbacks
                                                        )
        # resources_per_trial = {"cpu": 1, "gpu": 0}

        analysis = tune.run(train_fn_with_parameters,
                            # resources_per_trial=resources_per_trial,
                            metric="loss",
                            mode="min",
                            config=config,
                            num_samples=5,
                            max_concurrent_trials=2,
                            # scheduler=scheduler,
                            progress_reporter=reporter,
                            name="Digital_tune",
                            local_dir='./ray_results',

                            )

        print("Best hyperparameters found were: ", analysis.best_config)

        pass
    else:
        config = {
            "lr": 1e-3,
            "momentum": 0.8
        }
        train_mnist(config=config,
                    max_epoch=10,
                    train_set=train_set,
                    val_set=val_set,
                    show_bar=True,
                    Callbacks=[LearningRateMonitor(logging_interval='epoch')],
                    )
        pass

    pass


def main():
    description = 'if use ray tool for fine tune'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--tune', type=str, default='False', help='open ray tool fine tune the parameters.')
    parser.add_argument('--epoches', type=int, default=60, help='epoches of train.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size of the ')
    parser.add_argument('--momentume', type=float, default=0.8, help='momentume of sgd')
    args = parser.parse_args()

    if args.tune == 'True':
        run(True)
    elif args.tune == 'False':
        run(False)

    pass


if __name__ == '__main__':
    main()
