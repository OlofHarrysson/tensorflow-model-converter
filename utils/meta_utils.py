import subprocess
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def get_model_outdir(model_path, tf_version):
    mp = Path(model_path).stem
    tf_v = tf_version.replace('.', '_')
    out_dir = Path('output_models') / '{}_tensorflow_{}'.format(mp, tf_v)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_docker(docker_image, command, verbose):
    ''' Runs a docker container with the specified command '''
    project_root = get_project_root()
    workdir = '/convert'
    args = f"docker run -it --rm --name convert-tf -v {project_root}:{workdir} -w {workdir} {docker_image} {command}"

    subprocess.run(args.split(), check=True, capture_output=not verbose)

