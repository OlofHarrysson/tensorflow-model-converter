import subprocess
import argparse
from pathlib import Path

from utils import meta_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-m',
        '--model_path',
        type=str,
        # required=True,
        default='input_models/mymodel210.h5',
        help='Path to input model')

    tf_versions = ['1.12.0', '1.5.0', '2.1.0']

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

    return p.parse_args()


def main():
    args = parse_args()

    # Activate docker for input version and run inference on model with test inp.
    # Save model in both h5 and json+weights

    # Activate output docker and load model. Run inference on model with test inp. resave model in new version.

    # Run some kind of test that the two outputs are close.

    # Run docker with tf_2. Load model -> save both h5, weights and json
    # Run docker with tf_1. Load recently saved model -> save both h5, weights and json

    model_outdir = meta_utils.get_model_outdir(args.model_path,
                                               args.in_version)
    command = '/bin/bash'
    command = f'python convert_model.py -m {args.model_path} -o {model_outdir}'
    docker_image = f'tensorflow/tensorflow:{args.in_version}-py3'
    run_docker(docker_image, command)
    print("Resaved input model succesfully...")

    model_path = model_outdir / Path(args.model_path).name
    model_outdir = meta_utils.get_model_outdir(args.model_path,
                                               args.out_version)
    command = f'python convert_model.py -m {model_path} -o {model_outdir}'
    docker_image = f'tensorflow/tensorflow:{args.out_version}-py3'
    run_docker(docker_image, command)


def run_docker(docker_image, command):
    project_root = meta_utils.get_project_root()
    workdir = '/convert'
    args = f"docker run -it --rm --name convert-tf -v {project_root}:{workdir} -w {workdir} {docker_image} {command}"

    completed = subprocess.run(args.split())
    print('returncode:', completed.returncode)


if __name__ == '__main__':
    main()
