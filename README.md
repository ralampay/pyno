# Python Neural Outliers (PyNO)

PyNO is a neural network approach in detecting outliers for multivariate data.

## Installation and Setup

1. Install dependencies

For `pip` users use:

```
pip install -r requirements.txt
```

2. Activate the environment

For `venv` users:

```
source env/bin/activate
```

## Sample Usage

1. Train an autoencoder with 144 input size and 140 hidden neurons which will produce a model file called `model.pth` in the current directory (defaults to using `cpu` for training. To use gpu, pass `--device cuda:0` as a command line argument.

```
python -m pyno --mode train-ae --training-file data.csv --layers 144 140 --output-model-file model.pth
```

## Notes

* Code uses tabs as spaces with 2 spaces = 1 tab. (Might screw up code when used with other editors not using this format)
