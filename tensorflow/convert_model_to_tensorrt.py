import argparse

import utils.tensorrt_utils as tu


parser = argparse.ArgumentParser(description="Converts a Tensorflow model to TensorRT model",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '-m', '--model_dir',
    help='Path to model.',
    type=str,
    required=True
)
parser.add_argument(
    '-o', '--output_model_dir',
    help='Output dir for converted model.',
    type=str,
    required=True
)

def main():
    args = parser.parse_args()
    tu.conver_to_tensorrt_model(args)


if __name__ == "__main__":
    main()