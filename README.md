# A Divide and Conquer Training Strategy for Multi-Task Learning
This code was made under Pytorch v1.2.0.

Each provided folder contains code for its respective dataset (Celeb-A, CityScapes and NYUv2). Inside of these, a single 'train.py' training script is provided for all of the baselines described in the submission paper.

## Usage
In the 'train.py' training files, the different arguments can be used to reproduce the benchmark results in the paper. In particular:
* '--method': used to choose different multi-task baselines (MTL, MGDA, GradNorm, MR, PCGrad)
* '--per_batch_step': used to choose if task should be optimized altogether (MTL_SUS) or separately (MTL-IUS/MTL-IO). Only supported for MTL.
* '--one_optim_per_task': if 'per_batch_step' is True, used to choose whether to use MTL-IO (True) or MTL-IUS (False). Only supported for MTL.

## Data
For the Celeb-A dataset, we use the official release : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

For CityScapes and NYUv2, we use the preprocessed data made available here : https://github.com/lorenmt/mtan
