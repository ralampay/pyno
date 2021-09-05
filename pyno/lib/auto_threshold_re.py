import sys
import os
import numpy as np
from scipy.stats import iqr

class AutoThresholdRe():
  def __init__(self, x, autoencoder, error_type="mse"):
    self.autoencoder        = autoencoder
    self.error_type         = error_type
    self.x                  = x
    self.optimal_threshold  = -1

  def execute(self):
    self.set_optimal_threshold()

  def set_optimal_threshold(self):
    self.errors()

    self.bin_width  = 2 * iqr(self.errs) / np.power(len(self.errs), (1/3))
    self.num_bins   = (np.max(self.errs) - np.min(self.errs)) / self.bin_width

    hist, bins  = self.create_histogram(self.errs, num_bins=self.num_bins, step=self.bin_width)

    breaks  = self.htb(hist)

    possible_thresholds = []

    for b in breaks:
      t = self.fetch_threshold(bins, hist, b)
      possible_thresholds.append(t)

      self.optimal_threshold  = max(possible_thresholds)

    return self.optimal_threshold

  # TODO: Iterate over array efficiently
  def predict(self, x):
    if not self.optimal_threshold:
      raise Exception("No optimal threshold set")

    re = self.diff(x)

    bool_arr = re >= self.optimal_threshold

    return np.array([-1 if elem else 1 for elem in bool_arr])

  def diff(self, x):
    x_hat = self.autoencoder.forward(x)

    if self.error_type == "mse":
      err = (x_hat - x).pow(2).sum(dim=1).sqrt()
    else:
      raise Exception("Invalid error_type: {}".format(self.error_type))

    return err.detach().cpu().numpy()

  def errors(self, x=None):
    """ Computes the errors on a per dimensional basis

    Raises
    ------
    Exception
      If the error_type is not one of the following values:
        - mse (mean squared errors)
    """

    _x = (self.x if x == None else x).to(self.autoencoder.device)

    if x == None:
      x_hat = self.autoencoder.forward(_x)
    else:
      x_hat = self.autoencoder.forward(_x)

    if self.error_type == "mse":
      self.errs = (x_hat - _x).pow(2).sum(dim=1).sqrt()
    else:
      raise Exception("Invalid error_type: {}".format(self.error_type))

    self.errs = self.errs.detach().cpu().numpy()

    return self.errs

  def fetch_threshold(self, bins, counts, break_point):
    index       = 0
    latest_min  = 999999
    threshold   = -1

    for i in range(len(counts)):
      diff = abs(counts[i] - break_point)

      if diff <= latest_min:
        latest_min  = diff
        index       = i
        threshold   = ((bins[i + 1] - bins[i]) / 2) + bins[i]

    return threshold

  def create_histogram(self, data, num_bins=100, step=-1):
    min_bin = np.min(data)
    max_bin = np.max(data) + min_bin

    if step < 0:
      step = (max_bin - min_bin) / num_bins

    bins  = np.arange(min_bin, max_bin, step)

    (hist, bins)  = np.histogram(data, bins=bins)

    return (hist, bins)

  def htb(self, data):
    outp  = []

    def htb_inner(data):
      """
      Inner ht breaks function for recursively computing the break points.
      """
      data_length = float(len(data))
      data_mean   = sum(data) / data_length
      head  = [_ for _ in data if _ > data_mean]
      outp.append(data_mean)

      while len(head) > 1 and len(head) / data_length < 0.40:
        return htb_inner(head)

    htb_inner(data)

    return outp
