import subprocess
import functools
from pathlib import Path

from .multipledispatch import dispatch


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


def remove_keys(obj, key2remove):
    ''' Recursively deletes key from input dictionary '''

    if isinstance(obj, dict):
        new = obj.__class__()
        for k, v in obj.items():
            if k == key2remove:
                continue
            new[k] = remove_keys(v, key2remove)

    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(remove_keys(v, key2remove) for v in obj)
    else:
        return obj
    return new


@functools.total_ordering
class TensorflowVersion():
    def __init__(self, version):
        err_msg = "Expected version to be a string. Was '{}' with type '{}'".format(
            version, type(version))
        assert isinstance(version, str), err_msg

        allowed_chars = '-.0123456789'
        v = version.replace('.', '-')

        err_msg = "Version contained illegal characters. Expected version to consist of '{}' but was '{}'".format(
            allowed_chars, version)
        assert all([c in allowed_chars for c in v]), err_msg
        self._version = version

    @property
    def version(self):
        return self._version.replace('.', '-')

    def __str__(self):
        return self.version

    @dispatch(object)
    def __eq__(self, other):
        if not isinstance(other, TensorflowVersion):
            return NotImplemented

        other_v = str(other).split('-')
        this_v = str(self).split('-')
        for o, t in zip(other_v, this_v):
            if o != t:
                return False
        return True

    @dispatch(str)
    def __eq__(self, other: str):
        return TensorflowVersion(other) == self

    @dispatch(object)
    def __lt__(self, other):
        if not isinstance(other, TensorflowVersion):
            return NotImplemented

        other_v = str(other).split('-')
        this_v = str(self).split('-')
        for o, t in zip(other_v, this_v):
            if o != t:
                return int(o) < int(t)

        return False

    @dispatch(str)
    def __lt__(self, other: str):
        return TensorflowVersion(other) < self
