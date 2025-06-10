# Learning with Reputation Reward (LR2)
Learning with Reputation Reward (LR2) is a multi-agent reinforcement learning (MARL) method designed to foster cooperative behavior  through rewards shaping based on bottom-up reputation. This repository contains the implementation of LR2, based on the paper: [Bottom-Up Reputation Promotes Cooperation with Multi-Agent Reinforcement Learning](https://dl.acm.org/doi/10.5555/3709347.3743810), AAMAS2025

## Installation
This repository is built using PyTorch and supports both CPU and GPU environments. The instructions below are tailored for a Linux environment but should work on other operating systems as well.

**Step 1: Create a Conda Environment**

```bash
conda create -n lr2 python=3.10
conda activate lr2
```
**Step 2: Install Dependencies**
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
**Step 3: Install the LR2 Package**
```bash
cd LR2
pip install -e .
```

## Overview of Files
* `algorithms`: Contains the training algorithm for the LR2 agent, adapted from [Stable-Baselines3's PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) implemnetation. 
* `configs/config.py`: Contains configuration files specifying hyperparameters and environment settings.
* `scripts/train_pd_scripts`: Includes executable bash scripts for training with preset hyperparameters.
* `env/matrix_dilemma/_md_utils/lattice_env.py`: Implements the main interaction rules for agents, including reward and reputation calculation.
* `runner`: Contains the main training and evaluation logic.


## Running an Experiment
To start a training experiment, run the following bash script:
```bash
 ./lr2/scripts/train_pd_scripts/train_lattice.sh
```
All experiment results will be saved in the `Results` folder.

## Logging and Visualize the experiment results
We integrate [Weights and biases](https://wandb.ai/) to track the training process, visualize performance metrics, and manage experiment results.

**1. Login to W&B**

Before running an experiment, log into W&B by executing:
```bash
wandb.login()
```

**2. Activate W&B**

Enable W&B logging by setting the `--user_name username` and `--use_wandb` flags in the `lr2/scripts/train_pd_scripts/train_lattice.sh` script.

**3. Rendering**

To render the training process, add the `--use_render` flag. The rendered video will be automatically saved to W&B.


## Paper Citation
```bibtex
@inproceedings{ren2025bottom,
author = {Ren, Tianyu and Yao, Xuan and Li, Yang and Zeng, Xiao-Jun},
title = {Bottom-Up Reputation Promotes Cooperation with Multi-Agent Reinforcement Learning},
year = {2025},
address = {Richland, SC},
booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
pages = {1745–1754},
numpages = {10},
series = {AAMAS '25}
}
```
