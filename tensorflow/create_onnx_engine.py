import argparse
import os

import onnx.onnx_engine as engine

parser = argparse.ArgumentParser(description="Builds a onnx engine out of model",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '-m', '--onnx_model',
    help='Path to onnx model.',
    type=str,
    required=True
)
parser.add_argument(
    '-o', '--output',
    help='Output to save the engine.',
    type=str,
    required=True
)
parser.add_argument(
    '--input_shape',
    help='Input shape of the model.',
    nargs='+',
    required=False,
    default=[1, 320, 320, 3]
)


def main():
    args = parser.parse_args()

    onnx_model = os.path.normpath(args.onnx_model)
    output = os.path.normpath(args.onnx_model)

    eng = engine.build_engine(onnx_model, args.input_shape)
    engine.save_engine(eng, output)


if __name__ == "__main__":
    main()