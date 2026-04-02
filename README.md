# RoPOI
![](https://img.shields.io/badge/python-3.8.13-green)
![](https://img.shields.io/badge/pytorch-1.12.1-green)
![](https://img.shields.io/badge/cudatoolkit-11.6.0-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)

This folder provides a reference implementation of paper -- RoPOI: Robust POI Recommendation based on Modality Disentanglement and Missing Representation Generation

## Basic Usage

### Requirements

The code was tested with `python 3.8.13`, `pytorch 1.12.1`,  `cudatoolkit 11.6.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name RoPOI python=3.8.13

# activate environment
conda activate RoPOI

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# install other requirements
conda install numpy pandas
pip install scikit-learn
```

### Run the code
