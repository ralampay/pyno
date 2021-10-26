import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import plotext as plt
from torch import tensor
import torch

from lib.cnn_autoencoder import CnnAutoencoder
from lib.cv.utils import load_image_tensors

class CompressCnn:
    def __init__(self, params=None):
        self.model_file     = params.get('model_file')
        self.input_img_dir  = params.get('input_img_dir')
        self.output_file    = params.get('output_file')

    def execute(self):
        """
        load the model file
        """
        state   = torch.load(self.model_file)
        params  = state['params']

        """
        CNN Autoencoder parameters
        """
        self.scale          = params.get('scale')
        self.channel_maps   = params.get('channel_maps')
        self.padding        = params.get('padding')
        self.kernel_size    = params.get('kernel_size')
        self.num_channels   = params.get('num_channels')
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.h_activation   = params.get('h_activation')
        self.o_activation   = params.get('o_activation')
        self.device         = params.get('device')

        self.autoencoder    = CnnAutoencoder(
                                scale=self.scale,
                                channel_maps=self.channel_maps,
                                padding=self.padding,
                                kernel_size=self.kernel_size,
                                num_channels=self.num_channels,
                                img_width=self.img_width,
                                img_height=self.img_height,
                                h_activation=self.h_activation,
                                o_activation=self.o_activation,
                                device=self.device
                              )

        """
        Load the saved model
        """
        self.autoencoder.load(self.model_file)

        """
        Load images with proper dimensions as input data
        """
        print("Loading image tensors from {}...".format(self.input_img_dir))
        x = load_image_tensors(self.input_img_dir, self.img_width, self.img_height)
        x = x.to(self.device)

        """
        Encode the images
        """
        print("Encoding tensors...")
        z = self.autoencoder.encode(x)

        print("Building data frame...")
        data_to_csv = pd.DataFrame()

        for row in z:
            data_to_csv = data_to_csv.append([row])

        print("Saving to {}...".format(self.output_file))
        data_to_csv.to_csv(self.output_file, header=None, index=None)
