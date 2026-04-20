# RoPOI
![](https://img.shields.io/badge/python-3.9.23-green)
![](https://img.shields.io/badge/pytorch-2.7.1+cu118-green)
![](https://img.shields.io/badge/cudatoolkit-11.8-green)
![](https://img.shields.io/badge/torch__geometric-2.6.1-green)

This is the PyTorch implementation of the paper "RoPOI: Robust POI Recommendation based on Modality Disentanglement and Missing Representation Generation"

![model-structure](figures/framework.png)

## Requirements

The code was tested with `python 3.9.23`, `pytorch 2.7.1+cu118`, `cudatoolkit 11.8`, and `torch_geometric 2.6.1`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name RoPOI python=3.9.23

# activate environment
conda activate RoPOI

# install pytorch & cudatoolkit
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# install torch_geometric
pip install torch_geometric==2.6.1

# install other requirements
conda install numpy==1.26.4 pandas==2.3.2 tqdm==4.58.0
pip install scikit-learn==1.0.2
```

## Train
- Open the "RoPOI" folder
cd RoPOI

- Download dataset from https://drive.google.com/file/d/1xLXcRriNpMF9rAqsQOc4csjJBAeBuziP/view?usp=sharing.

- Download checkpoint from https://drive.google.com/file/d/1T83jRs-iGZ6zEDNhpKZpJXMYQFOnBY2A/view?usp=sharing.

- Unzip `NYC.zip` to `dataset/`.

- Run `build_graph.py` to construct the sequence graphs from the training data.

- Train the model using `python run.py`. All hyper-parameters are defined in `param_parser.py`

## Test
python run.py --test_only --checkpoint RoPOI_best_20260415_152057.pth

## Raw Dataset 

NYC: https://drive.google.com/file/d/1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC/view?usp=sharing

TKY: https://drive.google.com/file/d/1Cpnp2iEmHGfvUkOL-8myeQmyOuq4LmV7/view?usp=sharing

Please cite our paper if you use these datasets, thank you very much!
