import numpy as np
from itertools import combinations
from collections import namedtuple
from pathlib import Path

from . import meta_utils


def validate_outputs(model_outputs, verbose=True, eps=1e-05):
    ''' Checks wether a list of model outputs match each other '''

    # Print model outputs
    if verbose:
        for model in model_outputs:
            print(f"\nModel {model.name} output: shape={model.data.shape} data=\n\n{model.data}")

    # Compare models
    failed_models = []
    combs = combinations(model_outputs, 2)
    for model1, model2 in combs:
        m1_output, m2_output = model1.data, model2.data
        m1_name, m2_name = model1.name, model2.name

        err_msg = f"Output shapes differ for models '{m1_name}={m1_output.shape} & {m2_name}={m2_output.shape}'"
        assert m1_output.shape == m2_output.shape, err_msg

        correct = np.allclose(m1_output, m2_output, rtol=0, atol=eps)

        if not correct:
            failed_models.append((m1_name, m2_name))

    # Print result
    if failed_models:
        model_combs = [f"'{m1} & {m2}'" for m1, m2 in failed_models]

        print(
            f"Models {' and '.join(model_combs)} didn't return the same answer."
            "You can run the program again with an increased --epsilon value"
        )

    models_passed = not failed_models
    return models_passed


def test_models(models, verbose, epsilon):
    ''' Loads the models tests them '''
    save_dir = Path('inference_output')
    save_dir.mkdir(exist_ok=True)

    Model = namedtuple('Model_output', ['name', 'data'])
    model_outputs = []
    for model in models:
        tf_version = model.tf_version
        docker_image = f'tensorflow/tensorflow:{tf_version}-py3'
        save_path = save_dir / f'{model.name}_{model.path.stem}'
        command = f'python run_inference.py -m {model.path} -s {save_path}'
        meta_utils.run_docker(docker_image, command, verbose)
        load_path = f"{save_path}_tf_{tf_version.replace('.', '-')}.npy"
        model_output = np.load(load_path)
        model_outputs.append(Model(model.name, model_output))

    passed_test = validate_outputs(model_outputs, verbose, epsilon)
    if passed_test:
        out_model_path = models[-1].path.stem
        print(
            f"All tests passed. You can use your new model @ '{out_model_path}'"
        )
