import cv2
import numpy as np
import pandas as pd
from .utils import fetch_patches

def covariance_grid_vector(img, cell_width, cell_height):
  features = []

  if len(img.shape) != 2:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  else:
    gray = img.copy()

  gray = gray / 255

  cells = fetch_patches(gray, cell_width, cell_height)

  for cell in cells:
    df = pd.DataFrame(cell, columns=None)

    features.append(np.trace(df.cov().values))

  norm = [float(i)/max(features) for i in features]

  return np.array(norm)
