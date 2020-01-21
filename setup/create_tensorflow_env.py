import subprocess
import os


class TensorflowInstaller():
    def __init__(self):
        pass
        # args = 'conda init bash'
        # completed = subprocess.run(args.split())

        # args = 'source /root/.bashrc'
        # completed = subprocess.run(args.split())

    def install_tensorflow(self, tf_version):
        env_name = f'tf-{tf_version}'
        self._create_env(env_name)
        # self.activate_env(env_name)
        self._pip_tensorflow(env_name, tf_version)

    def _create_env(self, name, python=3.6):
        # args = f'conda create -n {name} python={python}'

        # args = f'/bin/bash conda create -n {name} python={python}'
        args = f'conda create -n {name} python={python}'
        run_cmd(args)

    def activate_env(self, name):
        pass
        print("WHATS MY VERSION 111111")
        args = f'bash -c "source activate {name}; python -V"'
        # args = eval "$(conda shell.bash hook)"
        run_cmd(args)

    def _pip_tensorflow(self, name, version):
        pre_args = f'bash -c "source activate {name}; '
        post_args = '; source deactivate"'
        args = f'pip install tensorflow=={version}'
        args = pre_args + args + post_args
        print(args)
        run_cmd(args)


def run_cmd(args):

    subprocess.run(args, shell=True)


if __name__ == '__main__':
    # shell = os.environ['SHELL']
    supported_versions = ['1.12.0', '2.1.0']
    # print(shell)
    subprocess.run('bash -c "conda activate base; python -V"', shell=True)

    # installer = TensorflowInstaller()
    # installer.install_tensorflow(tf_version='2.1.0')
