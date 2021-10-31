import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import plotext as plt
import glob
from torch import tensor
import torch
import cv2

from lib.cnn_autoencoder import CnnAutoencoder
from lib.cv.utils import load_image_tensors

class TrainCnnAe:
    def __init__(self, params=None):
        # CNN Autoencoder parameters
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
        self.gpu_index      = params.get('gpu_index')

        # Training configuration
        self.input_img_dir      = params.get('input_img_dir')
        self.output_model_file  = params.get('output_model_file')
        self.batch_size         = params.get('batch_size')
        self.epochs             = params.get('epochs')
        self.lr                 = params.get('lr')
        self.cont               = params.get('cont')
        self.model_file         = params.get('model_file')

    def execute(self):
        print("Training using device {}...".format(self.device))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))

            # Concatenate index of cuda machine specified
            self.device = "cuda:{}".format(self.gpu_index)

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

        if self.cont and self.model_file:
            print("Continuing training from model file {}".format(self.model_file))
            self.autoencoder.load(self.model_file)


        """
        Load images with proper dimensions as input data
        """
        x = load_image_tensors(self.input_img_dir, self.img_width, self.img_height)
        x = x.to(self.device)

        self.autoencoder.fit(
            x,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size
        )

        print("Latent Dimensionality: {}".format(self.autoencoder.latent_dim))

        print("Saving file to {}...".format(self.output_model_file))

        self.autoencoder.save(self.output_model_file)

        print("Done.")

