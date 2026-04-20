<div align="center">

# 🗺️ RoPOI
### Robust POI Recommendation via Modality Disentanglement & Missing Representation Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9.23-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.7.1+cu118-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyG-2.6.1-3C2179?style=for-the-badge&logo=pyg&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square"/>
</p>

> 📄 PyTorch implementation of **"RoPOI: Robust POI Recommendation based on Modality Disentanglement and Missing Representation Generation"**

</div>

---

## 🏗️ Framework

<div align="center">
<img src="figures/framework.png" width="85%" alt="RoPOI Framework"/>
</div>

---

## ✨ Highlights

- 🔀 **Modality Disentanglement** — Information Bottleneck-based general/specific feature decoupling
- 🧩 **Missing Modality Generation** — Robust reconstruction under incomplete multimodal inputs
- 📍 **POI Recommendation** — Sequential trajectory modeling with geo-aware graph embeddings
- ⚡ **Efficient Training** — GatedGraphConv + Transformer encoder architecture

---

## 🛠️ Requirements

```shell
# Create & activate environment
conda create --name RoPOI python=3.9.23
conda activate RoPOI

# Install PyTorch (CUDA 11.8)
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric==2.6.1

# Install other dependencies
conda install numpy==1.26.4 pandas==2.3.2 tqdm==4.58.0
pip install scikit-learn==1.0.2
```

---

## 🚀 Quick Start

### 1. Prepare Data

```shell
cd RoPOI-main
```

- 📦 Download dataset: [NYC.zip](https://drive.google.com/file/d/1xLXcRriNpMF9rAqsQOc4csjJBAeBuziP/view?usp=sharing) → unzip to `data/NYC/`
- 💾 Download checkpoint: [RoPOI_best.pth](https://drive.google.com/file/d/1T83jRs-iGZ6zEDNhpKZpJXMYQFOnBY2A/view?usp=sharing)

### 2. Build Graphs

```shell
python build_graph.py
```

### 3. Train

```shell
python run.py
```

> All hyper-parameters are defined in `param_parser.py`

### 4. Test

```shell
python run.py --test_only --checkpoint RoPOI_best_20260415_152057.pth
```

---

## 📊 Datasets

| Dataset | Link |
|---------|------|
| NYC | [Download](https://drive.google.com/file/d/1v0BvKs46ixUf1CgjRlk9MY5JsFL7ILxC/view?usp=sharing) |
| TKY | [Download](https://drive.google.com/file/d/1Cpnp2iEmHGfvUkOL-8myeQmyOuq4LmV7/view?usp=sharing) |

---

<div align="center">
If you use our datasets or code, please cite our paper. Thank you! 🙏
</div>
