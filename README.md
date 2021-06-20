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

# Quick start

# To run your code locally

Start by forking the example repo https://github.com/ministry-of-silly-code/examples

```shell
# 1 - Create a new virtual env
python3 -m venv venv
source ./venv/bin/activate

# 2 - Clone [or create] your project, in this case I'm using a basic mnist_classifier 
git clone https://github.com/[your-username]/examples.git
cd examples

# 3 - Install Buddy
# (and add it to you your requirements.txt if you are not using the given examples)
pip install -e git+https://github.com/ministry-of-silly-code/experiment_buddy.git#egg=experiment_buddy

pip install --upgrade -r requirements.txt

# Run your experiments
python ./mnist_classifier.py
```

#### Run an experiment remotely
We need to tell buddy where to deploy the experiments, first add to the file `~/.ssh/config` the following configuration:

For Mila users (replace $MILA_USER with your username)
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
Thanks to [obilaniu](https://github.com/obilaniu) for the config.

You now need to setup the cluster SSH key. how create and add your SSH keys can be found [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account) 

1. Add cluster public-key to: `https://github.com/settings/keys` if you don't know this yet, check this out [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account)
1. `pip install wandb && wandb init` there are two ways to set up wandb: 
    - set up wandb init from the cmd
    - or set export  WANDB_API_KEY = your wandb key which can be found here: https://wandb.ai/settings
1. Set the host parameter [here](https://github.com/ministry-of-silly-code/examples/blob/master/config.py#L28) to the ssh hostname you want to deploy to .e.g. "mila"
1. `python main.py`

More details on experiment-buddy:
1. experiment-buddy will add tagged commits to a dangling branch 
2. Supports: Unix based OS
