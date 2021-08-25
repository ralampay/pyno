import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train_ae import TrainAe

def main():
  parser = argparse.ArgumentParser(description="PyNO: Neural network outlier detector")

  parser.add_argument("--mode", help="Mode to be used", choices=["train-ae"], type=str, required=True)
  parser.add_argument("--training-file", help="CSV file for training autoencoder (classes should not be included)", type=str)
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

  args              = parser.parse_args()
  mode              = args.mode
  device            = args.device
  gpu_index         = args.gpu_index
  training_file     = args.training_file
  output_model_file = args.output_model_file
  layers            = args.layers
  lr                = args.lr
  epochs            = args.epochs
  cont              = args.cont
  h_activation      = args.h_activation
  o_activation      = args.o_activation
  chunk_size        = args.chunk_size
  batch_size        = args.batch_size


  if mode == "train-ae":
    print("Training using device {}...".format(device))

    if device == 'cuda':
      print("CUDA Device: {}".format(torch.cuda.get_device_name(gpu_index)))

      # Concatenate index of cuda machine specified
      device = "cuda:{}".format(gpu_index)

    params = {
      'layers':             layers,
      'epochs':             epochs,
      'lr':                 lr,
      'batch_size':         batch_size,
      'device':             device,
      'h_activation':       h_activation,
      'o_activation':       o_activation,
      'training_file':      training_file,
      'chunk_size':         chunk_size,
      'output_model_file':  output_model_file
    }

    cmd = TrainAe(params)

    cmd.execute()

    print("Saved file to {}.".format(output_model_file))
    print("Done.")

if __name__ == '__main__':
  main()
