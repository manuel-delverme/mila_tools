import ast
import concurrent.futures
import datetime
import os
import subprocess
import sys
import time
import types
import warnings

import fabric
import git
import matplotlib.pyplot as plt
import tensorboardX
import yaml

try:
    import torch
except ImportError:
    USE_TORCH = False
else:
    USE_TORCH = True
import wandb
import wandb.cli
from paramiko.ssh_exception import SSHException

wandb_escape = "^"
hyperparams = None
tb = None
SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "../slurm_scripts/")
ARTIFACTS_PATH = "runs/"
PROFILE = False


def timeit(method):
    if not PROFILE:
        return method

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__!r}  {(te - ts):2.2f} s')
        return result

    return timed


def register(config_params):
    global hyperparams
    # overwrite CLI parameters
    # fails on nested config object
    for k in config_params.keys():
        if k.startswith(wandb_escape):
            raise NameError(f"{wandb_escape} is a reserved prefix")

    for arg in sys.argv[1:]:
        assert arg[:2] == "--"
        k, v = arg[2:].split("=")
        k = k.lstrip(wandb_escape)
        v = _cast_param(v)

        if k not in config_params.keys():
            raise ValueError(f"Trying to set {k}, but that's not one of {list(config_params.keys())}")
        config_params[k] = v
    # TODO: should only register valid_hyperparams()
    hyperparams = config_params.copy()


def _cast_param(v):
    try:
        return ast.literal_eval(v)
    except ValueError:
        return v


def _valid_hyperparam(key, value):
    if key.startswith("__") and key.endswith("__"):
        return False
    if key == "_":
        return False
    if isinstance(value, (types.FunctionType, types.MethodType, types.ModuleType)):
        return False
    return True


class WandbWrapper:
    def __init__(self, experiment_id, project_name, local_tensorboard=None):
        # proj name is git root folder name
        print(f"wandb.init(project={project_name}, name={experiment_id})")

        # Calling wandb.method is equivalent to calling self.run.method
        # I'd rather to keep explicit tracking of which run this object is following
        self.run = wandb.init(project=project_name, name=experiment_id)

        self.tensorboard = local_tensorboard
        self.objects_path = os.path.join(ARTIFACTS_PATH, "objects/", self.run.name)
        os.makedirs(self.objects_path, exist_ok=True)

        def register_param(name, value, prefix=""):
            if not _valid_hyperparam(name, value):
                return

            if name == "_extra_modules_":
                for module in value:
                    for __k in dir(module):
                        __v = getattr(module, __k)
                        register_param(__k, __v, prefix=module.__name__.replace(".", "_"))
            else:
                name = prefix + wandb_escape + name
                # if the parameter was not set by a sweep
                if not name in wandb.config._items:
                    print(f"setting {name}={str(value)}")
                    setattr(wandb.config, name, str(value))
                else:
                    print(
                        f"not setting {name} to {str(value)}, str because its already {getattr(wandb.config, name)}, {type(getattr(wandb.config, name))}")

        for k, v in hyperparams.items():
            register_param(k, v)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        scalar_value = float(scalar_value)  # silently remove extra data such as torch gradients
        self.run.log({tag: scalar_value}, step=global_step, commit=False)
        if self.tensorboard:
            self.tensorboard.add_scalar(tag, scalar_value, global_step=global_step)

    def add_scalar_dict(self, scalar_dict, global_step):
        raise NotImplementedError
        # This is not a tensorboard funciton
        self.run.log(scalar_dict, step=global_step, commit=False)

    def add_figure(self, tag, figure, global_step=None, close=True):
        self.run.log({tag: figure}, step=global_step, commit=False)
        if close:
            plt.close(figure)

        if self.tensorboard:
            self.tensorboard.add_figure(tag, figure, global_step=None, close=True)

    def add_histogram(self, tag, values, global_step=None):
        wandb.log({tag: wandb.Histogram(values)}, step=global_step, commit=False)
        if self.tensorboard:
            self.tensorboard.add_histogram(tag, values, global_step=None)

    ###########################################################################
    # THE FOLLOWING METHODS ARE NOT IMPLEMENTED IN TENSORBOARD (can they be?) #
    ###########################################################################

    def add_object(self, tag, obj, global_step):
        if not USE_TORCH:
            raise NotImplementedError
        # This is not a tensorboard function
        local_path = os.path.join(self.objects_path, f"{tag}-{global_step}.pt")
        with open(local_path, "wb") as fout:
            try:
                torch.save(obj, fout)
            except Exception as e:
                raise e

        self.run.save(local_path)
        return local_path

    def watch(self, *args, **kwargs):
        self.run.watch(*args, **kwargs)


@timeit
def deploy(host: str = "", sweep_yaml: str = "", proc_num: int = 1, use_remote="", extra_slurm_headers="") -> WandbWrapper:
    if use_remote and not host:
        warnings.warn("use_remote is deprecated, use host instead")
        host = use_remote

    debug = '_pydev_bundle.pydev_log' in sys.modules.keys() and not os.environ.get('BUDDY_DEBUG_DEPLOYMENT', False)
    is_running_remotely = "SLURM_JOB_ID" in os.environ.keys()
    local_run = not host

    try:
        git_repo = git.Repo()
    except git.InvalidGitRepositoryError:
        raise ValueError(f"Could not find a git repo")

    project_name = git_repo.remotes.origin.url.split('.git')[0].split('/')[-1]

    if local_run and sweep_yaml:
        raise NotImplemented("Local sweeps are not supported")

    if is_running_remotely:
        print("using wandb")
        experiment_id = f"{git_repo.head.commit.message.strip()}"
        if sweep_yaml:
            jid = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        else:
            jid = os.environ["SLURM_JOB_ID"]
        return WandbWrapper(f"{experiment_id}_{jid}", project_name=project_name)

    dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    if debug:
        experiment_id = "DEBUG_RUN"
        tb_dir = os.path.join(git_repo.working_dir, ARTIFACTS_PATH, "tensorboard/", experiment_id, dtm)
        return WandbWrapper(f"{experiment_id}_{dtm}", project_name=project_name,
                            local_tensorboard=_setup_tb(logdir=tb_dir))

    experiment_id = _ask_experiment_id(host, sweep_yaml)
    print(f"experiment_id: {experiment_id}")
    if local_run:
        tb_dir = os.path.join(git_repo.working_dir, ARTIFACTS_PATH, "tensorboard/", experiment_id, dtm)
        return WandbWrapper(f"{experiment_id}_{dtm}", project_name=project_name, local_tensorboard=_setup_tb(logdir=tb_dir))
    else:
        _commit_and_sendjob(host, experiment_id, sweep_yaml, git_repo, project_name, proc_num, extra_slurm_headers)
        sys.exit()


def _ask_experiment_id(cluster, sweep):
    title = f'{"[CLUSTER" if cluster else "[LOCAL"}'
    if sweep:
        title = f"{title}-SWEEP"
    title = f"{title}]"

    try:
        import tkinter.simpledialog  # fails on the server or colab
        root = tkinter.Tk()
        root.withdraw()
        experiment_id = tkinter.simpledialog.askstring(title, "experiment_id")
        root.destroy()
    except:
        experiment_id = input(f"Running on {title} \ndescribe your experiment (experiment_id):\n")

    experiment_id = (experiment_id or "no_id").replace(" ", "_")
    if cluster:
        experiment_id = f"[CLUSTER] {experiment_id}"
    return experiment_id


def _setup_tb(logdir):
    print("http://localhost:6006")
    return tensorboardX.SummaryWriter(logdir=logdir)


def _open_ssh_session(hostname):
    """ TODO: move this to utils.py or to a class (better)
        TODO add time-out for unknown host
     """
    kwargs_connection = {
        "host": hostname
    }
    try:
        kwargs_connection["connect_kwargs"] = {"password": os.environ["BUDDY_PASSWORD"]}
    except KeyError:
        pass

    try:
        ssh_session = fabric.Connection(**kwargs_connection, connect_timeout=10)
        ssh_session.run("")
    except SSHException as e:
        raise SSHException("SSH connection failed!,"
                           f"Make sure you can successfully run `ssh {hostname}` with no parameters, any parameters should be set in the ssh_config file"
                           "If you need a password to authenticate set the Environment variable BUDDY_PASSWORD.")

    return ssh_session


def _ensure_scripts(hostname, extra_headers):
    ssh_session = _open_ssh_session(hostname)
    retr = ssh_session.run("mktemp -d -t experiment_buddy-XXXXXXXXXX")
    tmp_folder = retr.stdout.strip()
    for file_path in os.listdir(SCRIPTS_PATH):
        script_path = SCRIPTS_PATH + file_path
        if extra_headers and file_path in ("localenv_sweep.sh", "srun_python.sh"):
            with open(SCRIPTS_PATH + file_path) as fin:
                rows = fin.readlines()

            script_path = "/tmp/" + file_path
            with open(script_path, "w") as fout:
                for flag_idx in range(1, len(rows)):
                    old = rows[flag_idx - 1].strip()
                    new = rows[flag_idx].strip()
                    if old[:7] in ("#SBATCH", "") and new[:7] not in ("#SBATCH", ""):
                        rows.insert(flag_idx, "\n" + extra_headers + "\n")
                        break
                fout.write("".join(rows))

        ssh_session.put(script_path, tmp_folder + "/")
    return tmp_folder, ssh_session


def log_cmd(cmd, retr):
    print("################################################################")
    print(f"## {cmd}")
    print("################################################################")
    print(retr)
    print("################################################################")


@timeit
def _commit_and_sendjob(hostname, experiment_id, sweep_yaml: str, git_repo, project_name, proc_num, extra_slurm_header):
    git_url = git_repo.remotes[0].url
    # _ensure_scripts(hostname, extra_slurm_header)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        scripts_folder = executor.submit(_ensure_scripts, hostname, extra_slurm_header)
        hash_commit = git_sync(experiment_id, git_repo)

        entrypoint = os.path.relpath(sys.argv[0], git_repo.working_dir)
        if sweep_yaml:
            with open(sweep_yaml, 'r') as stream:
                data_loaded = yaml.safe_load(stream)

            if data_loaded["program"] != entrypoint:
                raise ValueError(f'YAML {data_loaded["program"]} does not match the entrypoint {entrypoint}')

            wandb_stdout = subprocess.check_output(["wandb", "sweep", "--name", experiment_id, "-p", project_name, sweep_yaml], stderr=subprocess.STDOUT).decode("utf-8")
            # sweep_id = wandb_stdout.split("/")[-1].strip()
            row, = [row for row in wandb_stdout.split("\n") if "Run sweep agent with:" in row]
            print([row for row in wandb_stdout.split("\n") if "View" in row][0])
            sweep_id = row.split()[-1].strip()

            ssh_args = (git_url, sweep_id, hash_commit)
            ssh_command = "/opt/slurm/bin/sbatch {0}/localenv_sweep.sh {1} {2} {3}"
            num_repeats = 1  # this should become > 1 for parallel sweeps
        else:
            ssh_args = (git_url, entrypoint, hash_commit)
            ssh_command = "bash -l {0}/run_experiment.sh {1} {2} {3}"
            num_repeats = 1  # this should become > 1 for parallel sweeps
            print("monitor your run on https://wandb.ai/")

    # TODO: assert -e git+git@github.com:manuel-delverme/mila_tools.git#egg=mila_tools is in requirements.txt
    scripts_folder, ssh_session = timeit(lambda: scripts_folder.result())()
    ssh_command = ssh_command.format(scripts_folder, *ssh_args)
    print(ssh_command)
    for proc_num in range(num_repeats):
        if proc_num > 0:
            time.sleep(1)
            raise NotImplemented
        if proc_num > 1:
            priority = "long"
            raise NotImplemented("localenv_sweep.sh does not handle this yet")
        ssh_session.run(ssh_command)


@timeit
def git_sync(experiment_id, git_repo):
    active_branch = git_repo.active_branch.name
    os.system(f"git checkout --detach")  # move changest to snapshot branch
    os.system(f"git add .")
    os.system(f"git commit -m '{experiment_id}'")
    git_hash = git_repo.commit().hexsha
    tag_name = f"snapshot/{active_branch}/{git_hash}"
    os.system(f"git tag {tag_name}")
    os.system(f"git push {git_repo.remote()} {tag_name}")  # send to online repo
    os.system(f"git reset HEAD~1")  # untrack the changes
    os.system(f"git checkout {active_branch}")
    return git_hash
