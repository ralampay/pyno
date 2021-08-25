import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from torch import tensor
import torch

from lib.autoencoder import Autoencoder

class TrainAe:
  def __init__(self, params=None):
    # Autoencoder parameters
    self.layers       = params.get('layers')
    self.epochs       = params.get('epochs')
    self.lr           = params.get('lr')
    self.batch_size   = params.get('batch_size')
    self.device       = params.get('device')
    self.h_activation = params.get('h_activation')
    self.o_activation = params.get('o_activation')

    # Training configuration
    self.training_file      = params.get('training_file')
    self.chunk_size         = params.get('chunk_size')
    self.output_model_file  = params.get('output_model_file')

  def execute(self):
    self.autoencoder  = Autoencoder(
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
    X = torch.tensor(data.values).float()

    self.autoencoder.fit(
      X,
      epochs=self.epochs,
      lr=self.lr,
      batch_size=self.batch_size
    )

    self.autoencoder.save(self.output_model_file)
