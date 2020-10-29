`mila_tools` aims to reduce the overhead to deploy experiments on mila clusters.

Right now it's responsabilities cover:
1. Deployment on servers
1. Handling of Sweeps
1. Tracking of hyperparameters
1. Code versioning
1. Notifications
1. Wandb integration.

Example:

1. `python3 -m pip install virtualenv`
1. `python3 -m virtualenv venv --python=python3.8`
1. `source venv/bin/activate`
1. `python -m pip install -r requirements.txt` # tools requirements
1. `cd examples/`
1. `git init .`
1. `git remote add origin https://github.com/user/repo.git`
1. Add mila pkey to: `https://github.com/settings/keys`
1. `python -m pip install jax jaxlib` # Local requirements
1. `python -m examples.mnist_classifier`
