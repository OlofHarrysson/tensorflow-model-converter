import argparse
from pathlib import Path
from collections import namedtuple

from utils import meta_utils
from utils import test_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('-m',
                   '--model_path',
                   type=str,
                   required=True,
                   help='Path to input model')

    # Stable releases from tensorflows docker. Could also work with non-stable releases
    tf_1 = [
        '1.11.0', '1.12.0', '1.12.3', '1.13.1', '1.13.2', '1.14.0', '1.15.0'
    ]
    tf_2 = ['2.0.0', '2.1.0']
    tf_versions = tf_1 + tf_2

    p.add_argument('-i',
                   '--in_version',
                   type=str,
                   required=True,
                   choices=tf_versions,
                   help='Tensorflow version for input model')

    p.add_argument('-o',
                   '--out_version',
                   type=str,
                   required=True,
                   choices=tf_versions,
                   help='Tensorflow version for output model')

    p.add_argument(
        '-e',
        '--epsilon',
        type=float,
        default=1e-6,
        help=
        "How closely the models' outputs need to match. Generally 1e-6 for float32 models"
    )

    p.add_argument("-q",
                   "--quiet",
                   action='store_false',
                   help='Disable print output from Docker')

    return p.parse_args()


def main():
    args = parse_args()
    in_version, out_version = args.in_version, args.out_version

    assert Path(args.model_path).exists(), f'File not found: {args.model_path}'

    verbose = args.quiet
    org_model_path = Path(args.model_path)
    inp_model_path = convert_model(org_model_path, in_version, verbose)
    out_model_path = convert_model(inp_model_path, out_version, verbose)

    Model = namedtuple('Model', ['path', 'tf_version', 'name'])
    models = [
        Model(org_model_path, in_version, 'original'),
        Model(inp_model_path, in_version, 'resaved'),
        Model(out_model_path, out_version, 'final')
    ]
    test_utils.test_models(models, verbose, args.epsilon)


def convert_model(model_path, tf_version, verbose):
    ''' Saves a model into a specific tensorflow version.
    Returns the path to the saved model '''
    model_outdir = meta_utils.get_model_outdir(model_path, tf_version)
    command = f'python convert_model.py -m {model_path} -o {model_outdir}'
    docker_image = f'tensorflow/tensorflow:{tf_version}-py3'
    meta_utils.run_docker(docker_image, command, verbose)
    print(f"Resaved model @ '{model_outdir}'")
    return model_outdir / Path(model_path).name


if __name__ == '__main__':
    main()
