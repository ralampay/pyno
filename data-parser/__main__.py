import sys
import argparse
import os
import uuid
import pandas as pd
from tabulate import tabulate

def range_limit_normalized(arg):
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be floating point number between 0 and 1")

    if f <= 0 or f >= 1:
        raise argparse.ArgumentTypeError("Range of number must be between 0 and 1")

    return f

def main():
    parser = argparse.ArgumentParser(description="Parser: Utility for partitioning data for unsupervised learning tasks")

    parser.add_argument("--input-file", help="CSV file for parsing (classes should be included as either 1 or -1 in the last column", type=str, required=True)
    parser.add_argument("--outlier-ratio", type=range_limit_normalized, help="Percentage of outliers present", default=0.05)
    parser.add_argument("--training-ratio", type=range_limit_normalized, help="Percentage of training data", default=0.7)
    parser.add_argument("--training-file", help="CSV output training file (doesn't contain labels)", type=str, default="{}-training.csv".format(str(uuid.uuid4())))
    parser.add_argument("--validation-file", help="CSV output validation file (contains labels 1 for normal and -1 for outlier in last column", type=str, default="{}-validation.csv".format(str(uuid.uuid4())))
    parser.add_argument("--chunk-size", help="Chunk size for reading data from CSV", type=int, default=1000)

    args            = parser.parse_args()
    outlier_ratio   = args.outlier_ratio
    training_ratio  = args.training_ratio
    input_file      = args.input_file
    training_file   = args.training_file
    validation_file = args.validation_file
    chunk_size      = args.chunk_size

    """ 
    Read the data chunk by chunk according to chunk_size to avoid memory exhaustion for large files
    """
    data = pd.DataFrame()

    for i, chunk in enumerate(pd.read_csv(input_file, header=None, chunksize=chunk_size)):
        print("Reading chunk {} from file {}...".format(i, input_file))
        data = data.append(chunk)

    input_dim = len(data.columns) - 1

    """
    Compute for validation data according to outlier ratio
    """
    training_data           = data.sample(frac=training_ratio)
    num_training_normal     = len(training_data[training_data[input_dim] == 1])
    num_training_outlier    = len(training_data[training_data[input_dim] == -1])
    training_data           = training_data.iloc[:,:input_dim]
    validation_data         = data.drop(training_data.index)

    num_validation_normal   = len(validation_data[validation_data[input_dim] == 1])
    num_validation_outlier  = len(validation_data[validation_data[input_dim] == -1])

    num_outlier_drop = int((1 - outlier_ratio) * len(validation_data))
    dropped_outliers = validation_data[validation_data[input_dim] == -1].sample(frac=outlier_ratio)

    validation_data = validation_data.drop(dropped_outliers.index)

    """
    Save files
    """
    print("Saving training file to {}...".format(training_file))
    training_data.to_csv(training_file, header=False, index=False)

    print("Saving validation file to {}...".format(validation_file))
    validation_data.to_csv(validation_file, header=False, index=False)

    """
    Print out statistics
    """
    num_validation_outlier      = len(validation_data[validation_data[input_dim] == -1])
    outlier_ratio_in_validation = num_validation_outlier / len(validation_data)

    print(
        tabulate(
            [
                ["Input File", input_file],
                ["Training File", training_file],
                ["Validation File", validation_file],
                ["Dimensionality", input_dim],
                ["# Training Data", len(training_data)],
                ["# Outliers in Training", num_training_outlier],
                ["# Normal in Training", num_training_normal],
                ["# Outliers in Validation", num_validation_outlier],
                ["# Normal in Validation", num_validation_normal],
                ["Outlier Ratio in Validation", outlier_ratio_in_validation * 100]
            ],
            headers=["Metric", "Value"],
            tablefmt="fancy_grid"
        )
    )

    print("Done.")

if __name__ == '__main__':
    main()
