# writer : shiyu
# code time : 2022/10/23

import os
from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128]):
        """
        Args:
            act_fn: Object of the activation function that should be used as non-linearity in the network.
            input_size: Size of the input images in pixels
            num_classes: Number of classes we want to predict
            hidden_sizes: A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()

        self.config = {
            # "act_fn": act_fn.__class__.__name__,
            # "input_size": input_size,
            # "num_classes": num_classes,
            # "hidden_sizes": hidden_sizes,
        }
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes

        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), nn.ReLU(inplace=True),
                       nn.Dropout(0.1)]
        layers += [nn.Linear(layer_sizes[-1], num_classes), nn.Softmax()]
        # A module list registers a list of modules as submodules (e.g. for parameters)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
