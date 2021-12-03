# Deep Reinforcement Learning 

Hello! Before diving in, I would recommend getting familiarized with basic Reinforcement Learning. 
Here is a link to my blog post on Reinforcement Learning to get you started: 
[RL Primer](https://shyamalanadkat.medium.com/reinforcement-learning-a-primer-29116d487e42)

**Also, if you're an AIPI530 student, and if this repo helps I don't mind some beer money/bitcoin LOL** 

## _TLDR_ - Example Usage 

Customized to training CQL on a custom dataset in d3rlpy, and training OPE (FQE) to 
evaluate the trained policy. 

---

1. Execute `cql_train.py` found at the root of the project
 - Default dataset is `hopper-bullet-mixed-v0`. For example if we want to run for 10 epochs: `python cql_train.py --epochs_cql 10 --epochs_fqe 10`
2. Important Logs:
 - Estimated Q values vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/init_value.csv`
 - Average reward vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/environment.csv`
 - True Q values vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/true_q_value.csv`
3. Examples speak more so:  [Jupyter notebook example](https://colab.research.google.com/drive/1S5RDTwaqVjA4wAJISxApra_G0ewSuS0R?usp=sharing) 

---
# Getting Started 

## Why d3rlpy?
d3rlpy is an offline deep reinforcement learning library for practitioners and researchers.

## How do I use it? 
```py
import d3rlpy

dataset, env = d3rlpy.datasets.get_dataset("hopper-medium-v0")

# prepare algorithm
sac = d3rlpy.algos.SAC()

# train offline
sac.fit(dataset, n_steps=1000000)

# train online
sac.fit_online(env, n_steps=1000000)

# ready to control
actions = sac.predict(x)
```

- Documentation: https://d3rlpy.readthedocs.io
- Paper: https://arxiv.org/abs/2111.03788

## How do I instal d3rlpy? 

d3rlpy supports Linux, macOS and Windows. There are several ways: 

### PyPI (recommended)
[![PyPI version](https://badge.fury.io/py/d3rlpy.svg)](https://badge.fury.io/py/d3rlpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/d3rlpy)
```
$ pip install d3rlpy
```
### Anaconda
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/d3rlpy/badges/version.svg)](https://anaconda.org/conda-forge/d3rlpy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/d3rlpy/badges/platforms.svg)](https://anaconda.org/conda-forge/d3rlpy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/d3rlpy/badges/downloads.svg)](https://anaconda.org/conda-forge/d3rlpy)
```
$ conda install -c conda-forge d3rlpy
```

## Show me some more examples ? 

### MuJoCo
```py
import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# train
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```
See more datasets at [d4rl](https://github.com/rail-berkeley/d4rl).

### Atari 2600
```py
import d3rlpy
from sklearn.model_selection import train_test_split

# prepare dataset
dataset, env = d3rlpy.datasets.get_atari('breakout-expert-v0')

# split dataset
train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQL(n_frames=4, q_func_factory='qr', scaler='pixel', use_gpu=True)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```
See more Atari datasets at [d4rl-atari](https://github.com/takuseno/d4rl-atari).

### PyBullet
<p align="center"><img align="center" width="160px" src="assets/hopper.gif"></p>

```py
import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_pybullet('hopper-bullet-mixed-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# start training
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```

See more PyBullet datasets at [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet).

## How about some tutorials? 
Try a cartpole example on Google Colaboratory:

- offline RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb)


## Citation
The paper is available [here](https://arxiv.org/abs/2111.03788).
```
@InProceedings{seno2021d3rlpy,
  author = {Takuma Seno, Michita Imai},
  title = {d3rlpy: An Offline Deep Reinforcement Library},
  booktitle = {NeurIPS 2021 Offline Reinforcement Learning Workshop},
  month = {December},
  year = {2021}
}
```
## Acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
