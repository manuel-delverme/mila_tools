import argparse
import datetime
import logging
import os
import subprocess
import sys
import types
import warnings
from multiprocessing import Pool
from typing import Dict, Optional

import cloudpickle
import git
import matplotlib.pyplot as plt
import tensorboardX
import tqdm
import wandb
import wandb.cli
import yaml

import experiment_buddy.utils
from experiment_buddy import executors

try:
    import torch
except ImportError:
    TORCH_ENABLED = False
    torch = None
else:
    TORCH_ENABLED = True

logging.basicConfig(level=logging.INFO)

tb = tensorboard = None
if os.path.exists("buddy_scripts/"):
    SCRIPTS_PATH = "buddy_scripts/"
else:
    SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "../scripts/")
ARTIFACTS_PATH = "runs/"
DEFAULT_WANDB_KEY = os.path.join(os.environ["HOME"], ".netrc")


# def register(config_params):
#     warnings.warn("Use register_defaults() instead")
#     return register_defaults(config_params)


# def register_defaults(config_params, allow_overwrite=False):
#     global hyperparams
#     # TODO: fails on nested config object
#     if allow_overwrite:
#         hyperparams = None
#
#     if hyperparams is not None:
#         raise RuntimeError("refusing to overwrite registered parameters")
#
#     if isinstance(config_params, argparse.Namespace):
#         raise Exception("Need a dict, use var() or locals()")
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('_ignored', nargs='*')
#
#     for k, v in config_params.items():
#         if _is_valid_hyperparam(k, v):
#             parser.add_argument(f"--{k}", type=type(v), default=v)
#             if "_" in k:
#                 k = k.replace("_", "-")
#                 parser.add_argument(f"--{k}", type=type(v), default=v)
#
#     try:
#         parsed = parser.parse_args()
#     except TypeError as e:
#         print("Type mismatch between registered hyperparameters defaults and actual values,"
#               " it might be a float argument that was passed as an int (e.g. lr=1 rather than lr=1.0) but set as a float (--lr 0.1)")
#         raise e
#
#     for k, v in vars(parsed).items():
#         config_params[k] = v
#
#     hyperparams = config_params.copy()


def _is_valid_hyperparam(key, value):
    if key.startswith("__") and key.endswith("__"):
        return False
    if key == "_":
        return False
    if isinstance(value, (types.FunctionType, types.MethodType, types.ModuleType)):
        return False
    return True


class WandbWrapper:
    def __init__(self, experiment_id, debug, wandb_kwargs):
        wandb_kwargs["mode"] = wandb_kwargs.get("mode", "offline" if debug else "online")
        if not debug:
            wandb_kwargs["settings"] = wandb_kwargs.get("settings", wandb.Settings(start_method="fork"))

        self.run = wandb.init(name=experiment_id, **wandb_kwargs)

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)


def deploy(url: str = "", sweep_definition: str = "", proc_num: int = 1, wandb_kwargs=None,
           extra_slurm_headers="", extra_modules=None, disabled=False, wandb_run_name=None,
           conda_env="base") -> WandbWrapper:
    """
    :param url: The host to deploy to.
    :param sweep_definition: Either a yaml file or a string containing the sweep id to resume from
    :param proc_num: The number of parallel jobs to run.
    :param wandb_kwargs: Kwargs to pass to wandb.init
    :param extra_slurm_headers: Extra slurm headers to add to the job script
    :param extra_modules: Extra modules to module load
    :param disabled: If true does not run jobs in the cluster and invokes wandb.init with disabled=True.
    :param run_per_agent: If set to a number, each agent will run `run_per_agent` experiments and then exit.
    :param wandb_run_name: If set, will use this name for the wandb run.
    :return: A tensorboard-like object that can be used to log data.
    """
    if wandb_kwargs is None:
        wandb_kwargs = {}
    if extra_modules is None:
        extra_modules = [
            "python/3.7",
            "pytorch/1.7",
        ]
    if not any("python" in m for m in extra_modules):
        warnings.warn("No python module found, are you sure?")

    extra_slurm_headers = extra_slurm_headers.strip()
    debug = sys.gettrace() is not None and not os.environ.get('BUDDY_DEBUG_DEPLOYMENT', False)
    running_on_cluster = "SLURM_JOB_ID" in os.environ.keys() or "BUDDY_IS_DEPLOYED" in os.environ.keys()
    local_run = not url and not running_on_cluster

    try:
        git_repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise ValueError(f"Could not find a git repo")

    if "project" in wandb_kwargs:
        project_name = wandb_kwargs["project"]
    else:
        project_name = experiment_buddy.utils.get_project_name(git_repo)
        wandb_kwargs["project"] = project_name

    common_kwargs = dict(debug=debug, wandb_kwargs=wandb_kwargs)
    dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    if disabled:
        wandb_kwargs["mode"] = "disabled"
        logger = WandbWrapper(f"buddy_disabled_{dtm}", **common_kwargs)
    elif running_on_cluster:
        print("using wandb")
        experiment_id = f"{git_repo.head.commit.message.strip()}"
        jid = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        jid += os.environ.get("SLURM_JOB_ID", "")
        # TODO: turn into a big switch based on scheduler
        logger = WandbWrapper(f"{experiment_id}_{jid}", **common_kwargs)
    elif debug:
        experiment_id = "DEBUG_RUN"
        logger = WandbWrapper(f"{experiment_id}_{dtm}", **common_kwargs)
    else:
        experiment_id = wandb_run_name if wandb_run_name is not None else ask_experiment_id(url, sweep_definition)
        print(f"experiment_id: {experiment_id}")

        if local_run:
            if sweep_definition:
                raise NotImplementedError(
                    "Local sweeps are not supported.\n"
                    f"SLURM_JOB_ID is {os.environ.get('SLURM_JOB_ID', 'KeyError')}\n"
                    f"BUDDY_IS_DEPLOYED is {os.environ.get('BUDDY_IS_DEPLOYED', 'KeyError')}\n"
                )

            return WandbWrapper(f"{experiment_id}_{dtm}", **common_kwargs)
        else:
            entrypoint = os.path.relpath(sys.argv[0], git_repo.working_dir)
            extra_modules = "@".join(extra_modules)

            sweep_id = None
            if sweep_definition:
                entity = wandb_kwargs.get("entity", None)
                sweep_id = _load_sweep(entrypoint, experiment_id, project_name, sweep_definition, entity)
                sweep_path = [entity, project_name, sweep_id]
                sweep_id = "/".join(sweep_path)

            hash_commit = git_sync(experiment_id, git_repo)
            git_url = git_repo.remotes[0].url

            if proc_num == -1:
                if sweep_id is None:
                    raise ValueError("proc_num is -1, but this is not a sweep")
                api = wandb.Api()
                sweep = api.sweep(sweep_id)
                proc_num = sweep.expected_run_count
                if proc_num is None:
                    raise ValueError("proc_num is None, is this a grid search?")

            SEQUENTIAL = sys.gettrace() is not None
            if SEQUENTIAL:
                for _ in tqdm.trange(proc_num):
                    send_job(entrypoint, extra_modules, extra_slurm_headers, git_repo, git_url, hash_commit, sweep_id,
                             url, conda_env)
            else:
                args = [(entrypoint, extra_modules, extra_slurm_headers, git_repo, git_url, hash_commit, sweep_id,
                         url, conda_env)] * proc_num
                with Pool(min(proc_num, 3)) as p:
                    p.starmap(send_job, args)

            sys.exit()

    return logger


def send_job(entrypoint, extra_modules, extra_slurm_headers, git_repo, git_url, hash_commit, sweep_id, url, conda_env):
    executor: executors.SSHSLURMExecutor = executors.get_executor(url)
    executor.setup_remote(extra_slurm_headers, git_repo.working_dir)
    if sweep_id:
        executor.sweep_agent(git_url, hash_commit, extra_modules, sweep_id, conda_env=conda_env)

    else:
        executor.launch_job(git_url, entrypoint, hash_commit, extra_modules, conda_env=conda_env)


def ask_experiment_id(cluster, sweep):
    title = f'{"[CLUSTER" if cluster else "[LOCAL"}'
    if sweep:
        title = f"{title}-SWEEP"
    title = f"{title}]"

    try:
        import tkinter.simpledialog  # fails on the server or colab
        logging.info("Name your run in the pop-up window!")
        root = tkinter.Tk()
        root.withdraw()
        experiment_id = tkinter.simpledialog.askstring(title, "experiment_id")
        root.destroy()
    except Exception as e:
        if os.environ.get('BUDDY_CURRENT_TESTING_BRANCH', ''):
            import uuid
            experiment_id = f'TESTING_BRANCH-{os.environ["BUDDY_CURRENT_TESTING_BRANCH"]}-{uuid.uuid4()}'
        else:
            experiment_id = input(f"Running on {title}\ndescribe your experiment (experiment_id):\n")

    experiment_id = (experiment_id or "no_id").replace(" ", "_")
    if cluster:
        experiment_id = f"[CLUSTER] {experiment_id}"
    return experiment_id


def log_cmd(cmd, retr):
    print("################################################################")
    print(f"## {cmd}")
    print("################################################################")
    print(retr)
    print("################################################################")


def _load_sweep(entrypoint, experiment_id, project, sweep_yaml, entity):
    with open(sweep_yaml, 'r') as stream:
        sweep_dict = yaml.safe_load(stream)
    sweep_dict["name"] = experiment_id

    if sweep_dict["program"] != entrypoint:
        warnings.warn(f'YAML {sweep_dict["program"]} does not match the entrypoint {entrypoint}')

    sweep_id = wandb.sweep(sweep_dict, project=project, entity=entity)
    return sweep_id


def git_sync(experiment_id, git_repo):
    if any(url.lower().startswith('https://') for url in git_repo.remote('origin').urls):
        raise Exception("Can't use HTTPS urls for your project, please, switch to GIT urls\n"
                        "Look here for more infos https://docs.github.com/en/github/getting-started-with-github/"
                        "getting-started-with-git/managing-remote-repositories#changing-a-remote-repositorys-url")

    active_branch = git_repo.active_branch.name
    try:
        subprocess.check_output(f"git checkout --detach", shell=True)  # move changest to snapshot branch
        subprocess.check_output(f"git add .", shell=True)

        try:
            subprocess.check_output(f"git commit --no-verify -m '{experiment_id}'", shell=True)
        except subprocess.CalledProcessError as e:
            git_hash = git_repo.commit().hexsha
            # Ensure the code is remote
            subprocess.check_output(f"git push {git_repo.remotes[0]} {active_branch}", shell=True)
        else:
            git_hash = git_repo.commit().hexsha
            tag_name = f"snapshot/{active_branch}/{git_hash}"
            subprocess.check_output(f"git tag {tag_name}", shell=True)
            subprocess.check_output(f"git push {git_repo.remotes[0]} {tag_name}", shell=True)  # send to online repo
            subprocess.check_output(f"git reset HEAD~1", shell=True)  # untrack the changes
    finally:
        subprocess.check_output(f"git checkout {active_branch}", shell=True)
    return git_hash
