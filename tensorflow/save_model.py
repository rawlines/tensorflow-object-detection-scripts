import os
import argparse

import utils.model_utils as mu


parser = argparse.ArgumentParser(description="Create frozen graph from model",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '-o', '--output',
    help='Output file for store the frozen graph.',
    type=str,
    required=True
)
parser.add_argument(
    '-p', '--pipeline',
    help='Path to model\'s pipeline.',
    type=str,
    required=True
)
parser.add_argument(
    '-c', '--checkpoint',
    help='Path to model\'s checkpoint.',
    type=str,
    required=True
)

def main():
    args = parser.parse_args()
    mu.save_model(args)
    

if __name__ == "__main__":
    main()