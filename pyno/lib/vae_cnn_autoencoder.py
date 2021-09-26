import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from abstract_dataset import AbstractDataset

class VaeCnnAutoencoder(nn.Module):
    def __init__(self, scale=2, channel_maps=[], padding=1, kernel_size=3, num_channels=3, img_width=500, img_height=500, device=torch.device("cpu"), criterion=nn.BCELoss(), h_activation="relu", o_activation="sigmoid", z_dim=256):
        super().__init__()

        if img_width != img_height:
            raise ValueError("Image width ({}) and height ({}) should be equal".format(img_width, img_height))

        self.scale          = scale
        self.channel_maps   = channel_maps
        self.padding        = padding
        self.kernel_size    = kernel_size
        self.num_channels   = num_channels
        self.img_width      = img_width
        self.img_height     = img_height
        self.device         = device
        self.criterion      = criterion
        self.z_dim          = z_dim

        self.h_activation = h_activation
        self.o_activation = o_activation

        self.reversed_channel_maps = list(reversed(channel_maps))

        # Build convolutional layers
        self.convolutional_layers = nn.ModuleList([])

        for i in range(len(self.channel_maps) - 1):
            self.convolutional_layers.append(
                nn.Conv2d(
                    self.channel_maps[i], 
                    self.channel_maps[i+1], 
                    kernel_size=self.kernel_size, 
                    padding=self.padding
                )
            )

        # Set feature dimension
        self.inner_depth = img_width

        for i in range(len(self.channel_maps) - 1):
            self.inner_depth = int(self.inner_depth / 2)

        self.feature_dimension = int(self.channel_maps[-1] * self.inner_depth * self.inner_depth)

        # Layer for mu
        self.conv_mu_layer = nn.Linear(self.feature_dimension, self.z_dim)

        # Layer for stdev
        self.conv_std_layer = nn.Linear(self.feature_dimension, self.z_dim)

        # Deconv layer for deconv
        self.deconv_layer = nn.Linear(self.z_dim, self.feature_dimension)

        # Build deconvolutional layers
        self.deconvolutional_layers = nn.ModuleList([])

        for i in range(len(self.reversed_channel_maps) - 1):
            self.deconvolutional_layers.append(
                nn.ConvTranspose2d(
                    self.reversed_channel_maps[i], 
                    self.reversed_channel_maps[i+1], 
                    kernel_size=self.kernel_size, 
                    padding=self.padding
                )
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.criterion = criterion

        self.errs = []

        # Initialize model to device
        self.to(self.device)

    def conv(self, x):
        for i in range(len(self.convolutional_layers)):
            conv_layer = self.convolutional_layers[i]

            if self.h_activation == "relu":
                x = F.relu(conv_layer(x))
            else:
                raise Exception("Invalid hidden activation {}".format(self.h_activation))

            x = self.pool(x)

        x = x.view(-1, self.feature_dimension)

        mu      = self.conv_mu_layer(x)
        log_var = self.conv_std_layer(x)


        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)

        return mu + std * eps

    def deconv(self, z):
        x = F.relu(self.deconv_layer(z))
        x = x.view(-1, self.channel_maps[-1], self.inner_depth, self.inner_depth)

        for i in range(len(self.deconvolutional_layers)):
            deconv_layer = self.deconvolutional_layers[i]
            x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
            x = deconv_layer(x)

            if i != len(self.deconvolutional_layers) - 1:
                if self.h_activation == "relu":
                    x = F.relu(x)
                else:
                    raise Exception("Invalid hidden activation {}".format(self.h_activation)) 
            else:
                if self.o_activation == "sigmoid":
                    x = torch.sigmoid(x)
                else:
                    raise Exception("Invalid output activation {}".format(self.o_activation))

        return x

    def forward(self, x):
        mu, log_var = self.conv(x)
        z = self.reparameterize(mu, log_var)
        out = self.deconv(z)

        return out, mu, log_var

    def save(self, filename):
        state = {
            'params': {
                'o_activation':   self.o_activation,
                'h_activation':   self.h_activation,
                'channel_maps':   self.channel_maps,
                'device':         self.device,
                'scale':          self.scale,
                'padding':        self.padding,
                'kernel_size':    self.kernel_size,
                'num_channels':   self.num_channels,
                'img_width':      self.img_width,
                'img_height':     self.img_height
            },
            'state_dict': self.state_dict(),
            'optimizer':  self.optimizer.state_dict()
        }

        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)

        self.load_state_dict(state['state_dict'])

        self.optimizer  = state['optimizer']

        # other parameters
        params = state['params']

        self.o_activation   = params['o_activation']
        self.h_activation   = params['h_activation']
        self.channel_maps   = params['channel_maps']
        self.device         = params['device']
        self.scale          = params['scale']
        self.padding        = params['padding']
        self.kernel_size    = params['kernel_size']
        self.num_channels   = params['num_channels']
        self.img_width      = params['img_width']
        self.img_height     = params['img_height']

    def fit(self, x, epochs=100, lr=0.001, batch_size=5, optimizer_type="adam"):
        # Reset errors to empty list
        self.errs = []

        data        = AbstractDataset(x)
        dataloader  = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=False)

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            raise Exception("Invalid optimizer_type: {}".format(optimizer_type))

        num_iterations = len(x) / batch_size

        for epoch in range(epochs):
            curr_loss = 0

            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                output = self.forward(inputs)

                loss = self.criterion(output, labels)

                curr_loss += loss
                loss.backward()
                self.optimizer.step()

            curr_loss = curr_loss / num_iterations

            print("Epoch: %i\tLoss: %0.5f" % (epoch + 1, curr_loss.item()))

            self.errs.append(curr_loss.detach())
