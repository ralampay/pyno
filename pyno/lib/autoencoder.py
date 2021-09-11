import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from abstract_dataset import AbstractDataset

class Autoencoder(nn.Module):
  def __init__(self, layers=[], h_activation="relu", o_activation="sigmoid", device=torch.device("cpu"), error_type="mse", optimizer_type="adam"):
    super().__init__()

    self.device = device

    self.h_activation = h_activation
    self.o_activation = o_activation

    self.encoding_layers  = nn.ModuleList([])
    self.decoding_layers  = nn.ModuleList([])
    
    self.error_type     = error_type
    self.optimizer_type = optimizer_type

    self.layers = layers
    self.reconstruction_threshold = -1

    reversed_layers = list(reversed(layers))

    for i in range(len(layers) - 1):
      self.encoding_layers.append(nn.Linear(layers[i], layers[i+1]))
      self.decoding_layers.append(nn.Linear(reversed_layers[i], reversed_layers[i+1]))

    self.errs = []

    # Initialize model to device
    self.to(self.device)

  def encode(self, x):
    for i in range(len(self.encoding_layers)):
      if self.h_activation == "relu":
        x = F.relu(self.encoding_layers[i](x))
      else:
        raise Exception("Invalid hidden activation {}".format(self.h_activation))

    return x

  def decode(self, x):
    for i in range(len(self.decoding_layers)):
      if i != len(self.decoding_layers) - 1:
        if self.h_activation == "relu":
          x = F.relu(self.decoding_layers[i](x))
        else:
          raise Exception("Invalid hidden activation {}".format(self.h_activation))
      else:
        if self.o_activation == "sigmoid":
          x = torch.sigmoid(self.decoding_layers[i](x))
        else:
          raise Exception("Invalid output activation {}".format(self.o_activation))

    return x

  def forward(self, x):
    x = self.encode(x)
    x = self.decode(x)

    return x

  def save(self, filename, anomaly_threshold=None):

    state = {
      'params': {
        'error_type':         self.error_type,
        'o_activation':       self.o_activation,
        'h_activation':       self.h_activation,
        'layers':             self.layers,
        'device':             self.device,
        'anomaly_threshold':  anomaly_threshold
      },
      'state_dict': self.state_dict(),
      'optimizer':  self.optimizer.state_dict()
    }

    torch.save(state, filename)

  def load(self, filename):
    state = torch.load(filename)

    self.load_state_dict(state['state_dict'])
    self.optimizer    = state['optimizer']

  def fit(self, x, epochs=100, lr=0.005, batch_size=5):
    # Reset errors to empty list
    self.errs = []

    data        = AbstractDataset(x)
    dataloader  = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=False)

    if self.optimizer_type == "adam":
      self.optimizer = optim.Adam(self.parameters(), lr=lr)
    else:
      raise Exception("Invalid optimizer_type: {}".format(self.optimizer_type))

    num_iterations = data.n_samples / batch_size

    for epoch in range(epochs):
      curr_loss = 0
      self.reconstruction_threshold = -1

      for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        output = self.forward(inputs)

        if self.error_type == "mse":
          loss = (output - labels).pow(2).sum(dim=1).sqrt().mean()
        else:
          raise Exception("Invalid error_type: {}".format(self.error_type))

        curr_loss += loss

        if loss > self.reconstruction_threshold:
          self.reconstruction_threshold = loss

        loss.backward()

        self.optimizer.step()

      curr_loss = curr_loss / num_iterations

      print("Epoch: %i\tLoss: %0.5f" % (epoch + 1, curr_loss.item()))

      self.errs.append(curr_loss.item())
