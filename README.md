# Machine Learning course project

**Implementation of Proximal Policy Optimization algorithm to control model of hopper robot**  
Daniel Khan R4235C  
ITMO University

 Report to the project [link](https://drive.google.com/file/d/1FNQtdQviKqAiIJrQJx1gZVXa6RHDkBIy/view?usp=sharing).

The following describes how to set up the project and reproduce the result.

## Install

Download repo:

```shell
$ git clone https://github.com/Danyachello/Hopper-robot-control-bu-PPO-algorithm.git
$ cd Hopper-robot-control-bu-PPO-algorithm
```
This project was performed on ubuntu os, so it is crutial to run it om ubuntu.
All following commands are expected to be run in the project root directory.

[Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
, create a new environment, *NAME*, and activate it:

```shell

$ conda env create --name NAME --file myenvironments.yaml
$ conda activate NAME
```

## Perform training of the model


File PPO_continious,py perform the training and validation of the model. After training, when total return will be grater, then desired reward, it will save weights of the networks in the file checkpoint and also rewrite file train.png in folder \media. During test programm will render video from simulator for 5 epispdes and saved them as gif in folder media. Also it will rcreate files with plots of applyid torques for each test episode in the folder media.



Train a model:

```shell
$ python PPO_continuous.py --mode train
```

Test model

```shell
$ python PPO_continuous.py --render
```

