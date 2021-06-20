`mila_tools` aims to reduce the overhead to deploy experiments on mila clusters.

## How to develop
Quick start

`experiment_buddy` aims to reduce the overhead to deploy experiments on servers.

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

### Usage

#### Local

```shell
# 1 - Clone [or create] your project, in this case I'm using a basic mnist_classifier 
git clone https://github.com/ministry-of-silly-code/examples.git
cd examples

# 2 - Create a new virtual env
python3 -m venv buddy-env
source ./buddy-env/bin/activate

# 3 - Install Buddy
# (and add it to you your requirements.txt if you are not using the given examples)
pip install -e git+https://github.com/ministry-of-silly-code/experiment_buddy.git#egg=experiment_buddy

# Run your experiments
python ./mnist_classifier.py
```

#### Cluster
Same as the local one, but you have to add to your `~/.ssh/config` the following configuration 

```shell
##   Mila
Host mila1   login-1.login.server.mila.quebec
    Hostname login-1.login.server.mila.quebec
Host mila2   login-2.login.server.mila.quebec
    Hostname login-2.login.server.mila.quebec
Host mila3   login-3.login.server.mila.quebec
    Hostname login-3.login.server.mila.quebec
Host mila4   login-4.login.server.mila.quebec
    Hostname login-4.login.server.mila.quebec
Host mila            login.server.mila.quebec
    Hostname         login.server.mila.quebec
Host mila* *login.server.mila.quebec
    Port 2222

Match host *.mila.quebec,*.umontreal.ca
    User $MILA_USER
    PreferredAuthentications publickey,keyboard-interactive
    Port 2222
    ServerAliveInterval 120
    ServerAliveCountMax 5
```

And you need to have your cluster private key enabled

How create and add your SSH keys can be found [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) 

1. Add cluster public-key to: `https://github.com/settings/keys` if you don't know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `pip install wandb && wandb init` there are two ways to set up wandb: 
    - set up wandb init from the cmd
    - or set export  WANDB_API_KEY = your wandb key which can be found here: https://wandb.ai/settings
1. `python main.py`

More details on experiment-buddy:
1. experiment-buddy will add tagged commits to a dangling branch 
2. Supports: Unix based OS
