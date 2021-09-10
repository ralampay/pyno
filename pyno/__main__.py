import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train_ae import TrainAe
from modules.eval_ae import EvalAe

def main():
  parser = argparse.ArgumentParser(description="PyNO: Neural network outlier detector")

  parser.add_argument("--mode", help="Mode to be used", choices=["train-ae", "eval-ae"], type=str, required=True)
  parser.add_argument("--training-file", help="CSV file for training autoencoder (classes should not be included)", type=str)
  parser.add_argument("--test-file", help="CSV file for testing (classes should be at the last column)", type=str)
  parser.add_argument("--model-file", help="Model file for audoencoder to be loaded", type=str)
  parser.add_argument("--output-model-file", help="Output file for model", type=str, default="model.pth")
  parser.add_argument("--device", help="Device used for training/evaluation/prediction", choices=["cpu", "cuda"], type=str, default="cpu")
  parser.add_argument("--gpu-index", help="GPU index", type=int, default=0)
  parser.add_argument("--layers", help='Layers for autoencoder', type=int, nargs='+')
  parser.add_argument("--lr", help='Learning rate', type=float, default=0.001)
  parser.add_argument("--epochs", help='Number of epochs', type=int, default=100)
  parser.add_argument("--cont", help='Continue from model file', type=bool, default=False)
  parser.add_argument("--h-activation", help='Hidden activation method', choices=["relu", "sigmoid"], type=str, default="relu")
  parser.add_argument("--o-activation", help='Output activation method', choices=["relu", "sigmoid"], type=str, default="sigmoid")
  parser.add_argument("--chunk-size", help='Chunk size for reading training data', type=int, default=100)
  parser.add_argument("--batch-size", help='Batch size for training data', type=int, default=100)
  parser.add_argument("--with-autothresholding", help='Save an autothreshold value for AE training', type=bool, default=True)
  parser.add_argument("--train-unsupervised", help='Train in an unsupervised manner', type=bool, default=False)

  args                  = parser.parse_args()
  mode                  = args.mode
  model_file            = args.model_file
  test_file             = args.test_file
  device                = args.device
  gpu_index             = args.gpu_index
  training_file         = args.training_file
  output_model_file     = args.output_model_file
  layers                = args.layers
  lr                    = args.lr
  epochs                = args.epochs
  cont                  = args.cont
  h_activation          = args.h_activation
  o_activation          = args.o_activation
  chunk_size            = args.chunk_size
  batch_size            = args.batch_size
  with_autothresholding = args.with_autothresholding
  train_unsupervised    = args.train_unsupervised


  if mode == "train-ae":
    params = {
      'layers':                 layers,
      'epochs':                 epochs,
      'lr':                     lr,
      'batch_size':             batch_size,
      'device':                 device,
      'gpu_index':              gpu_index,
      'h_activation':           h_activation,
      'o_activation':           o_activation,
      'training_file':          training_file,
      'chunk_size':             chunk_size,
      'output_model_file':      output_model_file,
      'with_autothresholding':  with_autothresholding,
      'train_unsupervised':     train_unsupervised
    }

    cmd = TrainAe(params)

    cmd.execute()

  elif mode == "eval-ae":
    if not model_file or not test_file:
      raise ValueError('model_file and test_file required for mode eval-ae')

    params = {
      'model_file':   model_file,
      'test_file':    test_file,
      'chunk_size':   chunk_size
    }

    cmd = EvalAe(params)

    cmd.execute()

if __name__ == '__main__':
  main()
