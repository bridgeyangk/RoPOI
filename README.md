# RoPOI

This is the PyTorch implementation of the paper "RoPOI: Robust POI Recommendation based on Modality Disentanglement and Missing Representation Generation"

![model-structure](figures/framework_.png)


## Train

- Download dataset from https://drive.google.com/file/d/17FbNvkO74xub6AeT2fpm938qDqUDB-04/view?usp=sharing.

- Unzip `NYC.zip` to `dataset/`.

- Run `build_graph.py` to construct the sequence graphs from the training data.

- Train the model using `python train.py`. All hyper-parameters are defined in `param_parser.py`

## Test
python run.py --test_only --checkpoint RoPOI_best_20260415_152057.pth

## Raw Dataset 

NYC: https://drive.google.com/file/d/1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC/view?usp=sharing

TKY: https://drive.google.com/file/d/1Cpnp2iEmHGfvUkOL-8myeQmyOuq4LmV7/view?usp=sharing

Please cite our paper if you use these datasets, thank you very much!