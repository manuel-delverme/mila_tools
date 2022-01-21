`experiment_buddy` aims to reduce the overhead to deploy experiments on servers.

Aims to reducing the cognitive load of running experiments remotely: with minimal changes it removes the need to think about experiment deployment for common use cases.

buddy is a work in progress, if you are intrerested in using it in your workflow, ping me in slack @delvermm

It's important to reduce cognitive overload for the researcher, as measured in seconds-to-first-tensorboard-datapoint.

Right now its responsibilities cover:
1. Deployment on servers
1. Handling of Sweeps
1. Tracking of hyperparameters
1. Code versioning
1. Notifications
1. Wandb integration.

Example: (Updated)

# Quick start with Buddy cli

## Requirements

1. You now need to setup the cluster SSH key. how create and add your SSH keys can be found [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) 
1. Add cluster public-key to: `https://github.com/settings/keys` if you don't know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `pip install wandb && wandb init` there are two ways to set up wandb: 
    - set up wandb init from the cmd
    - or set export  WANDB_API_KEY = your wandb key which can be found here: https://wandb.ai/settings


## Installation
### Directly from pip
```shell
pip install experiment-buddy
```

## Usage
### Interactive
Buddy will help you get started with the mila-cluster. Make sure you are using a virtual env, and run:
```shell
buddy-init
```
Once the setup is done you can also login with your Mila account on the cluster with `ssh mila`

# To run your code locally

Start by forking the example repo https://github.com/ministry-of-silly-code/examples

```shell
# 1 - Go to your 
cd <your_awesome_project>

# Run your experiments
python ./mnist_classifier.py
```

More details on experiment-buddy:
1. experiment-buddy will add tagged commits to a dangling branch 
2. Supports: Unix based OS

# To run your code on the cluster (Mila users)
Set host as `mila`:
```python
experiment_buddy.deploy(host='mila')
```
Run your code:
```shell
python main.py
```
## Git connection refused
If the ssh-agent is unable to connect to git, add the ssh-keys to the current ssh-agent:
```bash
eval `ssh-agent -s`
ssh-add
```
