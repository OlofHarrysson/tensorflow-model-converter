import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import json
import numpy as np

from . import meta_utils


def tf_version():
    return meta_utils.TensorflowVersion(tf.__version__)


def enable_eager():
    if tf_version() <= '1.15':
        tf.enable_eager_execution()


def save_model(model, path):
    assert path.suffix == '.h5', 'Only supports .h5 models'
    keras.models.save_model(model, path)
    h5_weights_path = str(path).replace('.h5', '_weights.h5')
    tf_weights_path = str(path).replace('.h5', '_weights_tf')
    model.save_weights(h5_weights_path)
    model.save_weights(tf_weights_path, save_format='tf')

    json_config = json.loads(model.to_json())

    with open(str(path).replace('.h5', '.json'), 'w') as outfile:
        json.dump(json_config, outfile, indent=2)


def load_model(path):
    try:
        return keras.models.load_model(path)
    except ValueError:
        return _load_json_model(path)


def _load_json_model(path_h5):
    assert path_h5.suffix == '.h5', 'Only supports .h5 models'
    path_json = str(path_h5).replace('.h5', '.json')

    with open(path_json) as infile:
        json_model = json.load(infile)

    try:
        model = tf.keras.models.model_from_json(json.dumps(json_model))
    except ValueError:
        json_model = _remove_ragged(json_model)
        model = tf.keras.models.model_from_json(json.dumps(json_model))

    h5_weights_path = str(path_h5).replace('.h5', '_weights.h5')
    tf_weights_path = str(path_h5).replace('.h5', '_weights_tf')

    try:
        model.load_weights(h5_weights_path)
    except ValueError:
        model.load_weights(tf_weights_path)

    return model


def _remove_ragged(json_model):
    ''' Tensorflow 1.4 and below doesn't support ragged tensors '''
    return meta_utils.remove_keys(json_model, 'ragged')


def prepare_input(model):
    ''' Prepares random data for model input to run inference on '''
    inp_shape = model.input_shape
    if len(inp_shape) == 1:  # Shape differs for tensorflow version
        inp_shape = inp_shape[0]
    inp_shape = (1, ) + inp_shape[1:]

    inp = np.random.rand(*inp_shape).astype(np.float32)
    return inp
