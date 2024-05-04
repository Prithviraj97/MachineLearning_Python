from abc import ABC, abstractmethod 
import numpy as np

from neural_networks.losses import initialize_loss 
from neural_networks.optimizers import initialize_optimizer 
from neural_networks.layers import initialize_layer 
from collections import OrderedDict 
import pickle 
from tqdm import tqdm 
import pandas as pd
from neural_networks.utils import AttrDict 
from neural_networks.datasets import Dataset 
from typing import Any, Dict, List, Sequence, Tuple

import losses

def initialize_model(name, loss, layer_args, optimizer_args, logger=None, seed=None):
    return NeuralNetwork(
    loss=loss,
    layer_args=layer_args,
    optimizer_args=optimizer_args,
    logger=logger,
    seed=seed,
)

class NeuralNetwork(ABC): 
    def __init__( self, loss: str, layer_args: Sequence[AttrDict], optimizer_args: AttrDict, logger=None, seed: int = None, ) -> None:
        self.n_layers = len(layer_args)
        self.layer_args = layer_args
        self.logger = logger
        self.epoch_log = {"loss": {}, "error": {}}

        self.loss = initialize_loss(loss)
        self.optimizer = initialize_optimizer(**optimizer_args)
        self._initialize_layers(layer_args)

    def _initialize_layers(self, layer_args: Sequence[AttrDict]) -> None:
        self.layers = []
        for l_arg in layer_args[:-1]:
            l = initialize_layer(**l_arg)
            self.layers.append(l)

    def _log(self, loss: float, error: float, validation: bool = False) -> None:

        if self.logger is not None:
            if validation:

                self.epoch_log["loss"]["validate"] = round(loss, 4)
                self.epoch_log["error"]["validate"] = round(error, 4)
                self.logger.push(self.epoch_log)
                self.epoch_log = {"loss": {}, "error": {}}
            else:
                self.epoch_log["loss"]["train"] = round(loss, 4)
                self.epoch_log["error"]["train"] = round(error, 4)

    def save_parameters(self, epoch: int) -> None:
        parameters = {}
        for i, l in enumerate(self.layers):
            parameters[i] = l.parameters
        if self.logger is None:
            raise ValueError("Must have a logger")
        else:
            with open(
                self.logger.save_dir + "parameters_epoch{}".format(epoch), "wb"
            ) as f:
                pickle.dump(parameters, f)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through all the layers of the neural network.

        Parameters
        ----------
        X  design matrix whose must match the input shape required by the
        first layer

        Returns
        -------
        forward pass output, matches the shape of the output of the last layer
        """
        ### YOUR CODE HERE ###
        input = X.copy()

        # Iterate through the network's layers.
        for l in self.layers:
            input = l.forward(input)

        return input

    def backward(self, target: np.ndarray, out: np.ndarray) -> float:
        """One backward pass through all the layers of the neural network.
        During this phase we calculate the gradients of the loss with respect to
        each of the parameters of the entire neural network. Most of the heavy
        lifting is done by the `backward` methods of the layers, so this method
        should be relatively simple. Also make sure to compute the loss in this
        method and NOT in `self.forward`.

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on training data

        Returns
        -------
        the loss of the model given the training inputs and targets
        """
        ### YOUR CODE HERE ###
        # Compute the loss.
        
        # Backpropagate through the network's layers.
        
        return 
