~`mila_tools` aims to reduce the overhead to deploy experiments on mila clusters.~

`experiment_buddy` aims to reduce the overhead to deploy experiments on servers.

buddy is a work in progress, if you are intrerested in using it in your workflow, ping me in slack @delvermm

It's important to reduce cognitive overload for the researcher measured as seconds-to-first-tensorboard datapoint

Right now it's responsabilities cover:
1. Deployment on servers
1. Handling of Sweeps
1. Tracking of hyperparameters
1. Code versioning
1. Notifications
1. Wandb integration.

Example: (Updated)

1. `python3 -m pip install virtualenv`
1. `python3 -m virtualenv venv --python=python3.8`
1. `source venv/bin/activate`
1. Add cluster private-key to: `https://github.com/settings/keys` if you dont know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `pip install wandb && wandb init` there are two ways to set up wandb: 
    - set up wandb init from the cmd
    - or set export  WANDB_API_KEY = your wandb key which can be found here: https://wandb.ai/settings
1. `python -m pip install jax jaxlib` # Local requirements
1. `python -m examples.mnist_classifier` 

More details on experiment-buddy:
1. experiment-buddy will commit to a branch called "experiment-buddy" to the first git repo found in the reverse os walk tree
2. Supports: Unix based OS


