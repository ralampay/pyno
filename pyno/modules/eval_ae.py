import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import plotext as plt
from torch import tensor
import torch
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from lib.autoencoder import Autoencoder
from lib.utils import performance_metrics

class EvalAe:
    def __init__(self, params=None):
        self.model_file = params.get('model_file')
        self.test_file  = params.get('test_file')
        self.chunk_size = params.get('chunk_size')

        self.data = pd.DataFrame()

    def execute(self):
        # load the model file
        state   = torch.load(self.model_file)
        params  = state['params']

        # Autoencoder parameters
        self.layers             = params.get('layers')
        self.device             = params.get('device')
        self.h_activation       = params.get('h_activation')
        self.o_activation       = params.get('o_activation')
        self.anomaly_threshold  = params.get('anomaly_threshold')
        self.error_type         = params.get('error_type')

        self.autoencoder  = Autoencoder(
                              layers=self.layers, 
                              h_activation=self.h_activation,
                              o_activation=self.o_activation,
                              device=self.device
                            )

        # Load the saved model
        self.autoencoder.load(self.model_file)

        for i, chunk in enumerate(pd.read_csv(self.test_file, header=None, chunksize=self.chunk_size)):
            self.data = self.data.append(chunk)

        # Dimensionality is the length of the column - 1
        self.input_dim = len(self.data.columns) - 1

        self.validation_data  = self.data.iloc[:, :self.input_dim]
        self.labels           = self.data.iloc[:,-1].values

        x     = torch.tensor(self.validation_data.values).float().to(self.device)
        x_hat = self.autoencoder.forward(x)

        if self.error_type == "mse":
            err = (x_hat - x).pow(2).sum(dim=1).sqrt()
        else:
            raise Exception("Invalid error_type: {}".format(self.error_type))

        err_values = err.detach().cpu().numpy()
        bool_array = err_values >= self.anomaly_threshold

        self.predictions  = np.array([-1 if elem else 1 for elem in bool_array])
        self.metrics      = performance_metrics(self.labels, self.predictions)

        print("Test File: {}".format(self.test_file))
        print("Model File: {}".format(self.model_file))
        print(
            tabulate(
                [
                    ["True Positive", self.metrics['tp']],
                    ["True Negative", self.metrics['tn']],
                    ["False Positive", self.metrics['fp']],
                    ["False Negative", self.metrics['fn']],
                    ["True Positive Rate", "{}%".format(round(self.metrics['tpr'] * 100, 2))],
                    ["True Negative Rate", "{}%".format(round(self.metrics['tnr'] * 100, 2))],
                    ["False Positive Rate", "{}%".format(round(self.metrics['fpr'] * 100, 2))],
                    ["False Negative Rate", "{}%".format(round(self.metrics['fnr'] * 100, 2))],
                    ["PPV", "{}%".format(round(self.metrics['ppv'] * 100, 2))],
                    ["NPV", "{}%".format(round(self.metrics['npv'] * 100, 2))],
                    ["TS", "{}%".format(round(self.metrics['ts'] * 100, 2))],
                    ["PT", "{}%".format(round(self.metrics['pt'] * 100, 2))],
                    ["Accuracy", "{}%".format(round(self.metrics['acc'] * 100, 2))],
                    ["F1", "{}%".format(round(self.metrics['f1'] * 100, 2))],
                    ["MCC", round(self.metrics['mcc'], 2)]
                ],
                headers=["Metric", "Value"],
                tablefmt="fancy_grid"
            )
        )
