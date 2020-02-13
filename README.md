# tensorflow-model-converter

## Intro

Tensorflow has support to read models from multiple versions but lacks export functionality to save models to a different version. For example, one can not read a tensorflow 2.x model into 1.x due to the introduction of "ragged tensors". 

This repo tries to fill that gap. It does so by loading a model and re-saving it into a target tensorflow version.

The program tests the models on a sample batch in an attempt to validate that the conversion was successful.


## Known limitations
- Testing only works for models with one input branch
- Small numerical differences can occur when resaving the model (due to Tensorflow versions i believe). Make sure to benchmark the new model.



## Requirements

- Python 3.7+
- Pip (very lightweight)
- Docker [(website)](https://www.docker.com/)



## Install 

```bash
git clone https://github.com/OlofHarrysson/tensorflow-model-converter.git
cd tensorflow-model-converter
pip install -r requirements.txt
python main.py --help
```



## Model compatibility
The project is new so haven't been tested very much. The plan is to have a compatibility matrix here for successfully/failed model conversions. For now, a list will do.

- [x] Functional keras models all versions
- [x] Nested functional keras model that was created through tensorflow.js 

## After conversion
Because some models have issues with normal saving/loading we save the model in multiple formats. Once the model is converted you can play around this code to figure out how to load your model.

```python
from tensorflow import keras
import json

# Load via keras. If this works, nothing below is needed.
model_h5_path = 'model.h5'
model = keras.models.load_model(model_h5_path)

# Load model from json
model_json_path = 'model.json'
with open(model_json_path) as infile:
    json_model = json.load(infile)
model = keras.models.model_from_json(json.dumps(json_model))

# Load weights saved with model.save_weights(path/to/model)
h5_weights_path = 'model_weights.h5'
model.load_weights(h5_weights_path)

# Loading weights that are saved in the "tf-format" https://www.tensorflow.org/guide/keras/save_and_serialize#saving_subclassed_models
# For this the model has to be saved with model.save_weights('path_to_my_weights', save_format='tf')
tf_weights_path = 'model_weights_tf'
model.load_weights(tf_weights_path)

model.summary()
```


## Feature requests

Vote for features or add suggestions you'd like to see at [https://internetport.hellonext.co/b/tensorflow-model-converter](https://internetport.hellonext.co/b/tensorflow-model-converter)  

Pull requests are welcome :)
