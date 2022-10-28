# This is a sample Python script.

# Press Ctrl+Alt+Shift+R to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from model.grid_model import train_mnist

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from utils.Dataset import MyDataset
from model.base_model import BaseNetwork
from model.lighting_model import LitAutoEncoder
import tempfile

data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")

from ray import tune

from argparse import ArgumentParser




def run(li_mode=False):
    x_train, y_train, x_test = load_data('./data')
    train_set = MyDataset(x_train, y_train)
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_set = DataLoader(train_set, batch_size=32)
    val_set = DataLoader(val_set, batch_size=32)

    # build lightning model
    base_model = BaseNetwork()
    autoencoder = LitAutoEncoder(base_model)
    config_l = {'epoches': 20}
    if li_mode:
        train_mnist(config_l, {'train_set': train_set, 'val_set': val_set}, autoencoder)
        pass
    else:
        x_train, y_train, x_test = load_data('./data')
        train_set = MyDataset(x_train, y_train)
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

        train_set = DataLoader(train_set, batch_size=32)
        val_set = DataLoader(val_set, batch_size=32)

        # build lightning model
        base_model = BaseNetwork()
        autoencoder = LitAutoEncoder(base_model)

        config = {
            # "epoches": tune.choice([5, 2]),
            'ids': tune.choice([1, 2])
        }

        trainable = tune.with_parameters(
            train_mnist,
            Data_loader={'train_set': train_set, 'val_set': val_set},
            lightnting_model=autoencoder
        )

        analysis = tune.run(
            trainable,
            metric="loss",
            mode="min",
            config=config,
            name="tune_mnist",
            max_concurrent_trials=2,

        )

        print(analysis.best_config)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--layer_1_dim", type=int, default=128)
    args = parser.parse_args()
    run()
