import datetime
import os
import subprocess
import sys
import types

import git
import matplotlib.pyplot as plt
import tensorboardX
import wandb
import wandb.cli

wandb_escape = "^"
hyperparams = None


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
    if "." in v:
        v = float(v)
    else:
        try:
            v = int(v)
        except ValueError:
            pass
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
    _global_step = 0

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def __init__(self, experiment_id, project_name):
        # proj name is git root folder name
        print(f"wandb.init(project={project_name}, name={experiment_id})")
        wandb.init(project=project_name, name=experiment_id)

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
                    print(f"not setting {name} to {str(value)}, str because its already {getattr(wandb.config, name)}, {type(getattr(wandb.config, name))}")

        for k, v in hyperparams.items():
            register_param(k, v)

    def add_scalar(self, tag, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step, commit=False)

    def add_figure(self, tag, figure, global_step=None, close=True):
        wandb.log({tag: figure}, step=global_step, commit=False)
        if close:
            plt.close(figure)

    def add_histogram(self, tag, values, global_step=None):
        wandb.log({tag: wandb.Histogram(values)}, step=global_step, commit=False)


def deploy(cluster, sweep):
    debug = '_pydev_bundle.pydev_log' in sys.modules.keys() or __debug__
    debug = False  # TODO: removeme

    git_repo = git.Repo(os.path.dirname(hyperparams["__file__"]))
    project_name = git_repo.remotes.origin.url.split('.git')[0].split('/')[-1]

    if (not cluster) and sweep:
        raise ValueError("?!?!?")

    if "SLURM_JOB_ID" in os.environ.keys():
        print("using wandb")
        experiment_id = f"{git_repo.head.commit.message.strip()}"
        dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        tb = WandbWrapper(f"{experiment_id}_{dtm}", project_name=project_name)
    else:
        experiment_id = _ask_experiment_id(cluster, debug)

        if not cluster or debug:
            dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + ".pt/"
            tb = _setup_tb(logdir=os.path.join("../tensorboard/", experiment_id, dtm))
        else:
            _commit_and_sendjob(experiment_id, sweep, git_repo, project_name)
            sys.exit()

    print(f"experiment_id: {experiment_id}", dtm)
    tb.global_step = 0

    if any((debug,)):
        print(r'''
                                        _.---"'"""""'`--.._
                                 _,.-'                   `-._
                             _,."                            -.
                         .-""   ___...---------.._             `.
                         `---'""                  `-.            `.
                                                     `.            \
                                                       `.           \
                                                         \           \
                                                          .           \
                                                          |            .
                                                          |            |
                                    _________             |            |
                              _,.-'"         `"'-.._      :            |
                          _,-'                      `-._.'             |
                       _.'                              `.             '
            _.-.    _,+......__                           `.          .
          .'    `-"'           `"-.,-""--._                 \        /
         /    ,'                  |    __  \                 \      /
        `   ..                       +"  )  \                 \    /
         `.'  \          ,-"`-..    |       |                  \  /
          / " |        .'       \   '.    _.'                   .'
         |,.."--"""--..|    "    |    `""`.                     |
       ,"               `-._     |        |                     |
     .'                     `-._+         |                     |
    /                           `.                        /     |
    |    `     '                  |                      /      |
    `-.....--.__                  |              |      /       |
       `./ "| / `-.........--.-   '              |    ,'        '
         /| ||        `.'  ,'   .'               |_,-+         /
        / ' '.`.        _,'   ,'     `.          |   '   _,.. /
       /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
      /... _.`:.________,.'              `._,.-..|        "'
     `.__.'                                 `._  /
                                               "'
      ''')


def _ask_experiment_id(cluster, debug):
    if debug:
        return f"DEBUG_RUN"

    import tkinter.simpledialog

    root = tkinter.Tk()
    title = "[CLUSTER]" if cluster else "[LOCAL]"
    experiment_id = tkinter.simpledialog.askstring(title, "experiment_id")
    experiment_id = (experiment_id or "no_id").replace(" ", "_")
    root.destroy()

    if cluster:
        experiment_id = f"[CLUSTER] {experiment_id}"
    return experiment_id


def _setup_tb(logdir):
    print("http://localhost:6006")
    return tensorboardX.SummaryWriter(logdir=logdir)


def _ensure_scripts():
    retr1 = os.popen("ssh mila md5sum localenv_sweep.sh run_experiment.sh srun_python.sh").read()
    retr2 = os.popen("md5sum slurm_scripts/localenv_sweep.sh slurm_scripts/run_experiment.sh slurm_scripts/srun_python.sh").read()
    if retr1.split()[::2] != retr2.split()[::2]:
        os.system("ssh mila mkdir '~/mila_tools_backup/'")
        os.system("ssh mila mv '~/localenv_sweep.sh ~/run_experiment.sh ~/srun_python.sh ~/mila_tools_backup/'")
        os.system("scp slurm_scripts/* mila:")


def _commit_and_sendjob(experiment_id, sweep, git_repo, project_name):
    code_version = git_repo.commit().hexsha
    url = git_repo.remotes[0].url
    # 2) commits everything to git with the name as message (so i r later reproduce the same experiment)
    os.system(f"git add .")
    os.system(f"git commit -m '{experiment_id}'")
    # 3) pushes the changes to git
    os.system("git push")

    _ensure_scripts()  # TODO: the check requires an extra ssh, this should be skipped

    if sweep:
        wandb_stdout = subprocess.check_output(["wandb", "sweep", "--name", experiment_id, "-p", project_name, sweep], stderr=subprocess.STDOUT).decode("utf-8")
        sweep_id = wandb_stdout.split("/")[-1].strip()
        # ctx = dict(default_map=wandb.cli.api.settings())
        # wandb.cli.sweep(ctx, project_name, entity=None, controller=False, verbose=False, name=project_name, program=False, settings=False, update=None, config_yaml=sweep)
        # with open(sweep, 'r') as stream:
        #     # sweep_id = wandb.sweep(yaml.safe_load(stream), entity=experiment_id, project=project_name)

        command = f"ssh mila /opt/slurm/bin/sbatch ./localenv_sweep.sh {url} {sweep_id} {code_version}"
        num_repeats = 1  # this should become > 1 for parallel sweeps
    else:
        _, entrypoint = os.path.split(sys.argv[0])
        # assert entrypoint is in $SWEEPCONFIG
        command = f"ssh mila bash -l ./run_experiment.sh {url} {entrypoint} {code_version}"
        num_repeats = 1  # this should become > 1 for parallel sweeps

    # command = f"ssh mila ./run_experiment.sh {next(git_repo.remote().urls)} {entrypoint} {git_repo.commit().hexsha}"
    print(command)
    for proc_num in range(num_repeats):
        if proc_num > 0:
            time.sleep(1)
            raise NotImplemented
        if proc_num > 1:
            priority = "long"
            raise NotImplemented("localenv_sweep.sh does not handle this yet")

        os.system(command)
    with open("ssh.log", 'w') as fout:
        fout.write(command)
    # 4) logs on the server and pulls the latest version
    # 5) runs the experiment
    # 7) writes me on slack/telegram/email a link for the tensorboard
