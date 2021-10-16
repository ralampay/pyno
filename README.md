# Python Neural Outliers (PyNO)

PyNO is a neural network approach in detecting outliers for multivariate data.

## Installation and Setup

1. Install dependencies

For `pip` users use:

```
pip install -r requirements.txt
```

Install `pytorch` manually (since currently it's not in the `pip` repositories:

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

2. Activate the environment

For `venv` users:

```
source env/bin/activate
```

## Sample Usage

### Training and Saving an Autoencoder Model

Train an autoencoder with 144 input size and 140 hidden neurons which will produce a model file called `model.pth` in the current directory. Defaults to using `cpu` for training. To use gpu, pass `--device cuda --gpu-index 0` as a command line argument (`--gpu-index` defaults to `0` when not specified).

```
python -m pyno --mode train-ae --training-file data.csv --layers 144 140 --output-model-file model.pth
```

### Using the Data Parser Utility

`data-parser` is a utility that will split an initial dataset

python -m data-parser --input-file https://happy-research.s3.ap-southeast-1.amazonaws.com/kddcup99.csv --training-file data/kddcup99-unsupervised-training.csv --validation-file data/kddcup99-unsupervised-validation.csv --outlier-ratio 0.1

## Notes
* A dataset used by `data-parser` is assumed to be normalized, no headers and will have labels at the last column where `1` represents normal / benign points and `-1` represents outliers.
* Code uses tabs as spaces with 2 spaces = 1 tab. (Might screw up code when used with other editors not using this format)
* When committing, make sure to clear all output from notebooks via `jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb` command
