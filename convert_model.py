import argparse
from pathlib import Path

from utils import tf_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('-m',
                   '--model_path',
                   type=str,
                   required=True,
                   help='Path to input model')

    p.add_argument('-o',
                   '--model_out_dir',
                   type=str,
                   required=True,
                   help='Directory where model will be saved')

    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    model = tf_utils.load_model(model_path)
    # model.summary()

    model_dir = Path(args.model_out_dir)
    tf_utils.save_model(model, model_dir / model_path.name)


if __name__ == '__main__':
    main()
