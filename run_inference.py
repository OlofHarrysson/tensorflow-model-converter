import argparse
from pathlib import Path
import numpy as np

from utils import tf_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('-m',
                   '--model_path',
                   type=str,
                   required=True,
                   help='Path to input model')

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(420)
    tf_utils.enable_eager()
    model_path = Path(args.model_path)
    model = tf_utils.load_model(model_path)

    #
    inputs = tf_utils.prepare_input(model)

    outputs = model(inputs)
    print(outputs)


if __name__ == '__main__':
    main()
