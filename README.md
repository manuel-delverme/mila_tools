~`mila_tools` aims to reduce the overhead to deploy experiments on mila clusters.~

`experiment_buddy` aims to reduce the overhead to deploy experiments on servers.


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
1. `cd examples/`
1. `git init .`
1. `git remote add origin https://github.com/user/repo.git`
1. Add mila pkey to: `https://github.com/settings/keys` if you dont know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `cd .. & pip install wandb && wandb init` then copy apy wandb key to set up the wandb inyour local repo
1. `python -m pip install jax jaxlib` # Local requirements
1. `python -m examples.mnist_classifier`


