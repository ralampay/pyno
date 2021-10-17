import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import plotext as plt
from torch import tensor
import torch

from lib.autoencoder import Autoencoder

class Compress:
    def __init__(self, params=None):
        self.model_file             = params.get('model_file')
        self.chunk_size             = params.get('chunk_size')
        self.input_file             = params.get('input_file')
        self.output_file            = params.get('output_file')
        self.compress_with_labels   = params.get('compress_with_labels')
        self.device                 = params.get('device')
        self.gpu_index              = params.get('gpu_index')

        self.data = pd.DataFrame()
   
        for i, chunk in enumerate(pd.read_csv(self.input_file, header=None, chunksize=self.chunk_size)):
            self.data = self.data.append(chunk)

        if self.compress_with_labels:
            self.input_dim = len(self.data.columns) - 1

            self.labels = []

            label_values = self.data.iloc[:,-1:].values

            for y in label_values:
                self.labels.append(y[0])
        else:
            self.input_dim = len(self.data.columns)

        self.X = self.data.iloc[:,:self.input_dim].values

    def execute(self):
        # Load device
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))

            # Concatenate index of cuda machine specified
            self.device = "cuda:{}".format(self.gpu_index)

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

        # Represent data as a tensor for training
        print("Storing data to tensor...")
        self.X = torch.tensor(self.X).float().to(self.device)

        print("Encoding...")
        self.Z = self.autoencoder.encode(self.X)

        result = self.Z.detach().numpy()

        data_to_csv = pd.DataFrame()

        for r in result:
            data_to_csv = data_to_csv.append([r])

        if self.compress_with_labels:
            data_to_csv['label'] = self.labels

        print("Saving to {}...".format(self.output_file))
        data_to_csv.to_csv(self.output_file, header=None, index=None)

        print("Done.")
