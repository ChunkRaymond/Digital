# writer : shiyu
# code time : 2022/10/23

import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.x_data = images
        self.y_data = labels
        self.transform = transform
        self.length = len(self.y_data)
        pass

    def __getitem__(self, index):
        return torch.tensor(np.expand_dims(self.x_data[index], axis=1), dtype=torch.float), torch.tensor(
            self.y_data[index], dtype=torch.float)

    def __len__(self):
        return self.length


def load_data(root_path):
    train_data = pd.read_csv(os.path.join(root_path, 'train.csv')).to_numpy()
    test_data = pd.read_csv(os.path.join(root_path, 'test.csv')).to_numpy()
    y_train = []
    x_train = []
    for i in train_data:
        y_train.append(i[0])
        x_train.append(np.array(i[1:]))

    y_train = np.array(y_train)
    x_train = np.array(x_train)
    y_tmp_train = []
    for y in y_train:
        y_tmp_train.append(np.eye(10, dtype='uint8')[y])
    y_train = np.array(y_tmp_train)

    x_test = []
    for i in test_data:
        x_test.append(np.array(i))

    x_test = np.array(x_test)
    return x_train, y_train, x_test
