import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import plotext as plt
from torch import tensor
import torch

from lib.autoencoder import Autoencoder
from lib.auto_threshold_re import AutoThresholdRe

class TrainAe:
    def __init__(self, params=None):
        # Autoencoder parameters
        self.layers       = params.get('layers')
        self.epochs       = params.get('epochs')
        self.lr           = params.get('lr')
        self.batch_size   = params.get('batch_size')
        self.device       = params.get('device')
        self.gpu_index    = params.get('gpu_index')
        self.h_activation = params.get('h_activation')
        self.o_activation = params.get('o_activation')

        # Training configuration
        self.training_file      = params.get('training_file')
        self.chunk_size         = params.get('chunk_size')
        self.output_model_file  = params.get('output_model_file')

        self.with_autothresholding  = params.get('with_autothresholding')

    def execute(self):
        print("Training using device {}...".format(self.device))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))

            # Concatenate index of cuda machine specified
            self.device = "cuda:{}".format(self.gpu_index)

        self.autoencoder =  Autoencoder(
                                layers=self.layers, 
                                h_activation=self.h_activation,
                                o_activation=self.o_activation,
                                device=self.device
                            )

        # Read data as data frame by chunk_size
        data = pd.DataFrame()

        for i, chunk in enumerate(pd.read_csv(self.training_file, header=None, chunksize=self.chunk_size)):
            data = data.append(chunk)

        # Represent data as a tensor for training
        print("Storing data to tensor...")
        X = torch.tensor(data.values).float().to(self.device)

        self.autoencoder.fit(
            X,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size
        )

        # Display the error in a plot
        y = self.autoencoder.errs
        plt.plot(range(1, self.epochs), y)
        plt.title("Training Error per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.axes_color("none")
        plt.canvas_color("none")
        plt.ticks_color("white")
        plt.show()

        if self.with_autothresholding:
            autothreshold_ops = AutoThresholdRe(X, self.autoencoder)
            autothreshold_ops.execute()

            anomaly_threshold = autothreshold_ops.optimal_threshold
        else:
            anomaly_threshold = None

        print("Saving file to {}...".format(self.output_model_file))

        self.autoencoder.save(self.output_model_file, anomaly_threshold=anomaly_threshold)

        print("Done.")
