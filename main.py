import subprocess
import argparse
from pathlib import Path
from collections import namedtuple

from utils import meta_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-m',
        '--model_path',
        type=str,
        # required=True,
        # default='input_models/mobilenet12.h5',
        default='input_models/mymodel210.h5',
        help='Path to input model')

    tf_versions = ['1.12.0', '1.15.0', '2.1.0']

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

    p.add_argument("-v",
                   "--verbose",
                   action="store_true",
                   help='Prints output from Docker')

    return p.parse_args()


def main():
    args = parse_args()
    in_version, out_version = args.in_version, args.out_version

    assert Path(args.model_path).exists(), f'File not found: {args.model_path}'

    inp_model_path = convert_model(args.model_path, in_version)
    out_model_path = convert_model(inp_model_path, out_version)

    Model = namedtuple('Model', ['path', 'tf_version'])
    models = [
        Model(args.model_path, in_version),
        Model(inp_model_path, in_version),
        Model(out_model_path, out_version)
    ]
    test_models(models)


def convert_model(model_path, tf_version, verbose=False):
    ''' Saves a model into a specific tensorflow version.
    Returns the path to the saved model '''
    model_outdir = meta_utils.get_model_outdir(model_path, tf_version)
    command = f'python convert_model.py -m {model_path} -o {model_outdir}'
    docker_image = f'tensorflow/tensorflow:{tf_version}-py3'
    # run_docker(docker_image, command, verbose)
    print(f"Resaved model @ '{model_outdir}'")
    return model_outdir / Path(model_path).name


def run_docker(docker_image, command, verbose):
    ''' Runs a docker container with the specified command '''
    project_root = meta_utils.get_project_root()
    workdir = '/convert'
    args = f"docker run -it --rm --name convert-tf -v {project_root}:{workdir} -w {workdir} {docker_image} {command}"

    subprocess.run(args.split(), check=True, capture_output=not verbose)


def test_models(models):
    for model in models:
        tf_version = model.tf_version
        docker_image = f'tensorflow/tensorflow:{tf_version}-py3'
        command = f'python run_inference.py -m {model.path}'
        run_docker(docker_image, command, True)
        # break


if __name__ == '__main__':
    main()
