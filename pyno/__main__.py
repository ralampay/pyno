import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train_ae import TrainAe
from modules.train_filtered_ae import TrainFilteredAe
from modules.train_neural_filtered_ae import TrainNeuralFilteredAe
from modules.train_cnn_ae import TrainCnnAe
from modules.eval_ae import EvalAe
from modules.predict_ae import PredictAe
from modules.compress import Compress
from modules.compress_cnn import CompressCnn

def main():
    mode_choices = [
        "train-ae",
        "train-filtered-ae",
        "train-neural-filtered-ae",
        "train-cnn-ae",
        "eval-ae",
        "compress",
        "compress-cnn",
        "predict-ae"
    ]

    parser = argparse.ArgumentParser(description="PyNO: Neural network outlier detector")

    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--training-file", help="CSV file for training autoencoder (classes should not be included)", type=str)
    parser.add_argument("--test-file", help="CSV file for testing (classes should be at the last column)", type=str)
    parser.add_argument("--model-file", help="Model file for audoencoder to be loaded", type=str)
    parser.add_argument("--output-model-file", help="Output file for model", type=str, default="model.pth")
    parser.add_argument("--input-file", help="Input file for mode compress", type=str, required=False)
    parser.add_argument("--output-file", help="Output file for mode compress", type=str, default="output.csv")
    parser.add_argument("--compress-with-labels", help="Compress data and include labels from file (assumed to be last column)", type=bool, default=True)
    parser.add_argument("--compress-with-errors", help='Include residual errors with AE compress', type=bool, default=False)
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
    parser.add_argument("--channel-maps", help='Channel maps for CNN', type=int, nargs='+')
    parser.add_argument("--img-width", help='Image width for CNN', type=int, default=500)
    parser.add_argument("--img-height", help='Image height for CNN', type=int, default=500)
    parser.add_argument("--scale", help='Scale for CNN', type=int, default=2)
    parser.add_argument("--kernel-size", help='Kernel size for CNN', type=int, default=3)
    parser.add_argument("--num-channels", help='Number of channels for CNN', type=int, default=3)
    parser.add_argument("--input-img-dir", help='Input image directory for CNN', type=str)
    parser.add_argument("--padding", help='Padding value for CNN', type=int, default=1)

    args                  = parser.parse_args()
    mode                  = args.mode
    model_file            = args.model_file
    test_file             = args.test_file
    device                = args.device
    gpu_index             = args.gpu_index
    training_file         = args.training_file
    output_model_file     = args.output_model_file
    input_file            = args.input_file
    output_file           = args.output_file
    layers                = args.layers
    lr                    = args.lr
    epochs                = args.epochs
    cont                  = args.cont
    h_activation          = args.h_activation
    o_activation          = args.o_activation
    chunk_size            = args.chunk_size
    batch_size            = args.batch_size
    with_autothresholding = args.with_autothresholding
    channel_maps          = args.channel_maps
    scale                 = args.scale
    kernel_size           = args.kernel_size
    num_channels          = args.num_channels
    input_img_dir         = args.input_img_dir
    compress_with_labels  = args.compress_with_labels
    compress_with_errors  = args.compress_with_errors
    padding               = args.padding
    img_width             = args.img_width
    img_height            = args.img_height


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
            'with_autothresholding':  with_autothresholding
        }

        cmd = TrainAe(params)

        cmd.execute()

    elif mode == "train-filtered-ae":
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
            'with_autothresholding':  with_autothresholding
        }

        cmd = TrainFilteredAe(params)

        cmd.execute()

    elif mode == "train-neural-filtered-ae":
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
            'with_autothresholding':  with_autothresholding
        }

        cmd = TrainNeuralFilteredAe(params)

        cmd.execute()

    elif mode == "compress":
        if not model_file:
            raise ValueError('model_file required for mode compress')

        if not input_file:
            raise ValueError('input_file required for mode compress')

        params = {
            'model_file':           model_file,
            'chunk_size':           chunk_size,
            'output_file':          output_file,
            'input_file':           input_file,
            'compress_with_labels': compress_with_labels,
            'compress_with_errors': compress_with_errors,
            'device':               device,
            'gpu_index':            gpu_index
        }

        cmd = Compress(params)

        cmd.execute()

    elif mode == "compress-cnn":
        if not model_file:
            raise ValueError('model_file required for mode compress-cnn')

        if not output_file:
            raise ValueError('output_file required for mode compress-cnn')

        params = {
            'model_file':       model_file,
            'input_img_dir':    input_img_dir,
            'output_file':      output_file
        }

        cmd = CompressCnn(params)

        cmd.execute()

    elif mode == "train-cnn-ae":
        if not input_img_dir:
            raise ValueError("{} required for CNN AE Training".format(input_img_dir))

        params = {
            'scale':                  scale,
            'channel_maps':           channel_maps,
            'padding':                padding,
            'kernel_size':            kernel_size,
            'num_channels':           num_channels,
            'img_width':              img_width,
            'img_height':             img_height,
            'h_activation':           h_activation,
            'o_activation':           o_activation,
            'input_img_dir':          input_img_dir,
            'output_model_file':      output_model_file,
            'epochs':                 epochs,
            'lr':                     lr,
            'batch_size':             batch_size,
            'device':                 device,
            'gpu_index':              gpu_index,
            'cont':                   cont,
            'model_file':             model_file
        }

        cmd = TrainCnnAe(params=params)

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

    elif mode == "predict-ae":
        if not model_file or not test_file:
            raise ValueError('model_file and test_file required for mode predict-ae')

        params = {
            'model_file':   model_file,
            'test_file':    test_file,
            'chunk_size':   chunk_size
        }

        cmd = PredictAe(params)

        cmd.execute()
  

if __name__ == '__main__':
    main()
