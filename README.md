# Shared Weights, Separate Losses: Alternate and Independent Task-Specific Optimization for Multi-Task Learning
This code was made under Pytorch v1.2.0.

Each provided folder contains code for its respective dataset (Celeb-A, CityScapes and NYUv2). Inside of these, a single 'train.py' training script is provided for all of the baselines described in the submission paper.

## Usage
In the 'train.py' training files, the different arguments can be used to reproduce the benchmark results in the paper. Instructions about their usage can be found in the respective folders.

## Data
We use the official releases of each dataset:
* NYUv2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
* Cityscapes: https://www.cityscapes-dataset.com/
* Celeb-A: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
