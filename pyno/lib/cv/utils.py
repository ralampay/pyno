import torch
from torch import tensor
import cv2
import glob
import numpy as np

def fetch_patches(img, cell_width, cell_height):
  if len(img.shape) == 2:
    height, width = img.shape
  else:
    height, width, channels = img.shape

  if width % cell_width != 0:
    raise Exception("Invalid cell_width %d for width %d" % (cell_width, width))

  if height % cell_height != 0:
    raise Exception("Invalid cell_height %d for height %d" % (cell_height, height))

  cells = []

  for x in range(0, width, cell_width):
    for y in range(0, height, cell_height):
      roi = img[y:y+cell_height, x:x+cell_width]

      cells.append(roi)

  return cells

def load_image_tensors(input_img_dir, img_width, img_height):
    images = []

    ext = ['png', 'jpg', 'gif', 'tiff', 'tif'] # Add supported image formats here

    files = []
    [files.extend(glob.glob(input_img_dir + '/*.' + e)) for e in ext]

    dim = (img_width, img_height)

    images = np.array([cv2.resize(cv2.imread(file), dim) for file in files])

    images = images / 255 # Normalize the images

    x = []

    for img in images:
        result = img.transpose((2, 0, 1))
        x.append(result)

    return torch.tensor(x).float()
