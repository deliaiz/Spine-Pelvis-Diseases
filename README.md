# Gait-Based Diagnostic Network for Localization and Pathological Characterization of Spine and Pelvis Diseases 🧠✨
This is the official code for the paper "Gait-Based Diagnostic Network for Localization and Pathological
Characterization of Spine and Pelvis Diseases", published in Medical Image Analysis, Accepted 10 August 2026.
# 💡 Overview
![Overview](images/framework.png)
# 🛠️ Method


# Software Requirements
## Hardware requirements
The package development version is tested on Linux operating systems.
Linux: Ubuntu 16.04
window: window 10
CUDA/cudnn:10.1
## Python Dependencies
> - Python
> - PyTorch-cuda
> - torchvision
> - opencv
> - numpy
> - json
> - os
>
...
# 🚀 Quick Start
## Prepare dataset
1. Prepare an txt file containing video names(*.mp4) and the Label to complete video sequences as the data input for training.
# Training and Evaluation example
Training and evaluation are on a single GPU. A GPU with approximately 10 GB of memory is sufficient for training and inference.
## Train
python train.py
## Evaluation
python test.py
