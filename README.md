# tensorflow-model-converter

## Intro

Tensorflow has support to read models from multiple versions but lacks export functionality to save models to a different version. For example, one can not read a tensorflow 2.x model into 1.x due to the introduction of "ragged tensors". 

This repo tries to fill that gap. It does so by loading a model and re-saving it into a target tensorflow version.



## Requirements

Python 3.6+

Docker [(website)](https://www.docker.com/)



## Install 

```bash
git clone https://github.com/OlofHarrysson/tensorflow-model-converter.git
cd tensorflow-model-converter
python main.py -h
```



## Feature requests

Vote for features or add suggestions you'd like to see at [https://internetport.hellonext.co/b/tensorflow-model-converter](https://internetport.hellonext.co/b/tensorflow-model-converter)  

Pull requests are welcome :)