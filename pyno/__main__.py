import sys
import argparse
import os

import yaml

def main():
  parser = argparse.ArgumentParser(description="PyNO: Neural network outlier detector")

  parser.add_argument("--config", help="Config file to use (yaml)", type=str, required=True)

  args = parser.parse_args()

  config_file = args.config

  with open(config_file) as stream:
    try:
      config = yaml.safe_load(stream)
    except yaml.YAMLError as e:
      print("Error in parsing yaml file {}".format(config_file))
      print(e)
      sys.exit()

  print("Input size: {}".format(config.get('input_size')))
  print("Done...")

if __name__ == '__main__':
  main()
