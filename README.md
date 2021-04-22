~`mila_tools` aims to reduce the overhead to deploy experiments on mila clusters.~
@Mila users, the documentation is currently out of date,
ping me on slack @delvermm to understand how to get started.


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

1. Add cluster private-key to: `https://github.com/settings/keys` if you dont know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `pip install wandb && wandb init` there are two ways to set up wandb: 
    - set up wandb init from the cmd
    - or set export  WANDB_API_KEY = your wandb key which can be found here: https://wandb.ai/settings
1. `python main.py`

More details on experiment-buddy:
1. experiment-buddy will add tagged commits to a dangling branch 
2. Supports: Unix based OS
