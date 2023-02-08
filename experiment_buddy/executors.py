import abc
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import warnings
from typing import Optional

import docker
import fabric
import wandb
import wandb.env
from invoke import UnexpectedExit
from paramiko.ssh_exception import SSHException

import experiment_buddy.utils

wandb_escape = "^"
tb = tensorboard = None
if os.path.exists("buddy_scripts/"):
    SCRIPTS_PATH = "buddy_scripts/"
else:
    SCRIPTS_PATH = os.path.join(os.path.dirname(__file__), "../scripts/")
ARTIFACTS_PATH = "runs/"
DEFAULT_WANDB_KEY = os.path.join(os.environ["HOME"], ".netrc")


def ensure_torch_compatibility():
    if not os.path.exists("requirements.txt"):
        return

    with open("requirements.txt") as fin:
        reqs = fin.read()
        # torch, vision or audio.
        matches = re.search(r"torch==.*cu.*", reqs)
        if "torch" in reqs and not matches:
            # https://mila-umontreal.slack.com/archives/CFAS8455H/p1624292393273100?thread_ts=1624290747.269100&cid=CFAS8455H
            warnings.warn(
                """torch rocm4.2 version will be installed on the cluster which is not supported specify torch==1.7.1+cu110 in requirements.txt instead""")


def _insert_extra_header(extra_slurm_header, script_path):
    tmp_script_path = f"/tmp/{os.path.basename(script_path)}"
    with open(script_path) as f_in, open(tmp_script_path, "w") as f_out:
        rows = f_in.readlines()
        first_free_idx = 1 + next(i for i in reversed(range(len(rows))) if "#SBATCH" in rows[i])
        rows.insert(first_free_idx, f"\n{extra_slurm_header}\n")
        f_out.write("\n".join(rows))
    return tmp_script_path


class Executor(abc.ABC):
    @abc.abstractmethod
    def __init__(self, url):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, cmd):
        raise NotImplementedError

    @abc.abstractmethod
    def setup_remote(self, extra_slurm_header: str, working_dir: str) -> str:
        raise NotImplementedError

    def put(self, local_path, remote_path):
        return


class SSHExecutor(Executor):
    def __init__(self, url):
        ensure_torch_compatibility()
        try:
            self.ssh_session = fabric.Connection(host=url.netloc, connect_timeout=10, forward_agent=True)
            self.ssh_session.run("")
        except SSHException as e:
            raise SSHException(
                "SSH connection failed!,"
                f"Make sure you can successfully run `ssh {url.netloc}` with no parameters, "
                f"any parameters should be set in the ssh_config file"
            )
        # self.scripts_dir = None
        self.scripts_folder = None  # TODO: what is the difference between scripts_dir and scripts_folder?
        self.working_dir = None

    def run(self, cmd):
        return self.ssh_session.run(cmd)

    def put(self, local_path, remote_path):
        return self.ssh_session.put(local_path, remote_path)

    def launch(self, git_url, entrypoint, hash_commit, extra_modules):
        ssh_command = f"bash -l {self.scripts_folder}/run_experiment.sh {git_url} {entrypoint} {hash_commit} {extra_modules}"
        self.ssh_session.run(ssh_command)
        time.sleep(1)

    def sweep(self, git_url, hash_commit, extra_modules, sweep_id):
        ssh_command = f"sbatch {self.scripts_folder}/run_sweep.sh {git_url} {sweep_id} {hash_commit} {extra_modules}"
        raise NotImplementedError

    def check_or_copy_wandb_key(self):
        try:
            self.ssh_session.run("test -f $HOME/.netrc")
        except UnexpectedExit:
            print(f"Wandb api key not found in {self.ssh_session.host}. Copying it from {DEFAULT_WANDB_KEY}")
            self.ssh_session.put(DEFAULT_WANDB_KEY, ".netrc")

    def setup_remote(self, extra_slurm_header: Optional[str], working_dir: str):
        if extra_slurm_header:
            raise NotImplementedError

        self.working_dir = working_dir
        self._ensure_scripts_directory(self.working_dir)

    def _ensure_scripts_directory(self, working_dir: str):
        if self.scripts_folder:
            return

        self.check_or_copy_wandb_key()
        retr = self.ssh_session.run("mktemp -d -t experiment_buddy-XXXXXXXXXX")
        remote_tmp_folder = retr.stdout.strip() + "/"
        # self.ssh_session.put(f'{SCRIPTS_PATH}/common/common.sh', remote_tmp_folder)

        backend = experiment_buddy.utils.get_backend(self.ssh_session, working_dir)
        scripts_dir = os.path.join(SCRIPTS_PATH, backend)

        for file in os.listdir(scripts_dir):
            self.ssh_session.put(os.path.join(scripts_dir, file), remote_tmp_folder)

        self.scripts_folder = remote_tmp_folder


class SSHSLURMExecutor(Executor):
    def __init__(self, url):
        ensure_torch_compatibility()
        self.ssh_session = fabric.Connection(host=url.path, connect_timeout=10, forward_agent=True)
        self.scripts_dir = None
        self.scripts_folder = None  # TODO: what is the difference between scripts_dir and scripts_folder?
        self.extra_slurm_header = None
        self.working_dir = None

    def run(self, cmd):
        return self.ssh_session.run(cmd)

    def put(self, local_path, remote_path):
        return self.ssh_session.put(local_path, remote_path)

    def launch(self, git_url, entrypoint, hash_commit, extra_modules):
        ssh_command = f"bash -l {self.scripts_folder}/run_experiment.sh {git_url} {entrypoint} {hash_commit} {extra_modules}"
        if not self.scripts_dir:
            self.check_or_copy_wandb_key()
            self.scripts_dir = self._ensure_scripts_directory(self.extra_slurm_header, self.working_dir)
        self.ssh_session.run(ssh_command)
        time.sleep(1)

    def sweep(self, git_url, hash_commit, extra_modules, sweep_id):
        ssh_command = f"sbatch {self.scripts_folder}/run_sweep.sh {git_url} {sweep_id} {hash_commit} {extra_modules}"
        raise NotImplementedError

    def check_or_copy_wandb_key(self):
        try:
            self.ssh_session.run("test -f $HOME/.netrc")
        except UnexpectedExit:
            print(f"Wandb api key not found in {self.ssh_session.host}. Copying it from {DEFAULT_WANDB_KEY}")
            self.ssh_session.put(DEFAULT_WANDB_KEY, ".netrc")

    def setup_remote(self, extra_slurm_header: str, working_dir: str):
        self.extra_slurm_header = extra_slurm_header
        self.working_dir = working_dir

    def _ensure_scripts_directory(self, extra_slurm_header: str, working_dir: str):
        retr = self.ssh_session.run("mktemp -d -t experiment_buddy-XXXXXXXXXX")
        remote_tmp_folder = retr.stdout.strip() + "/"
        self.ssh_session.put(f'{SCRIPTS_PATH}/common/common.sh', remote_tmp_folder)

        backend = experiment_buddy.utils.get_backend(self.ssh_session, working_dir)
        scripts_dir = os.path.join(SCRIPTS_PATH, backend)

        for file in os.listdir(scripts_dir):
            if extra_slurm_header and file in ("run_sweep.sh", "srun_python.sh"):
                new_tmp_file = _insert_extra_header(extra_slurm_header, os.path.join(scripts_dir, file))
                self.ssh_session.put(new_tmp_file, remote_tmp_folder)
            else:
                self.ssh_session.put(os.path.join(scripts_dir, file), remote_tmp_folder)

        self.scripts_folder = remote_tmp_folder


class DockerExecutor(Executor):
    EXPERIMENT_PATH = "/experiment"

    def __init__(self, url):
        self.docker_client = docker.from_env()
        self.docker_tag = f"base-cpu:latest"
        self.context = url.hostname or "default"
        self.repo_archive = None

    def run(self, cmd, container_id=None, environment=None):
        image = container_id or self.docker_tag

        environment = environment or {}  # TODO: overwrites the environment?
        environment.update({"DOCKER_CONTEXT": self.context})
        self.docker_client.containers.run(image, cmd, environment=environment, remove=True)

    def launch(self, git_url, entrypoint, hash_commit, extra_modules):
        self.maybe_pack_archive(git_url, hash_commit)

        out = subprocess.run(f"docker --context {self.context} run --name {hash_commit} --rm -it -t {self.docker_tag} sleep infinity",
                             shell=True,
                             capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(f"Error launching docker container: {out.stderr}")

        # container = self.docker_client.containers.create(self.docker_tag, "sleep infinity", detach=True)
        # docker cp archive.tar container_name:/path/to/destination
        out = subprocess.run(f"docker --context {self.context} cp {self.repo_archive} {hash_commit}:{self.EXPERIMENT_PATH}", shell=True,
                             capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(f"Error copying archive to docker container: {out.stderr}")

        # container.put_archive(self.EXPERIMENT_PATH, open(self.repo_archive, "rb"))
        # container.exec_run(f"{self.EXPERIMENT_PATH}/run_experiment.sh {self.EXPERIMENT_PATH} {entrypoint} {extra_modules}",
        #                    environment={wandb.env.API_KEY: wandb.api.api_key}, detach=True)
        out = subprocess.run(
            f"docker --context {self.context} exec -it {hash_commit} {self.EXPERIMENT_PATH}/run_experiment.sh {self.EXPERIMENT_PATH} {entrypoint} {extra_modules}",
            shell=True,
            capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(f"Error running experiment in docker container: {out.stderr}")

        # container.logs(stream=True, follow=True)

    def maybe_pack_archive(self, git_url, hash_commit):
        if self.repo_archive:
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            os.system(
                f"git clone --depth 1 {git_url} {tmpdir} && cd {tmpdir} && git fetch {git_url} {hash_commit} && git checkout {hash_commit}")
            os.system(f"tar -czf {tmpdir}.tar.gz -C {tmpdir} .")
            self.repo_archive = f"{tmpdir}.tar.gz"

    def put(self, local_path, remote_path):
        pass

    def check_or_copy_wandb_key(self):
        # wandb_api_key = wandb.api.api_key
        # self.docker_client.containers.run(self.docker_tag, f"wandb login {wandb_api_key}")
        pass

    def setup_remote(self, extra_slurm_header: str, working_dir: str) -> Optional[str]:
        docker_scripts_dir = os.path.join(os.path.dirname(__file__), "../scripts/docker")
        docker_file = os.path.join(docker_scripts_dir, "Dockerfile")

        out = subprocess.run(f"docker --context {self.context} inspect {self.docker_tag}", shell=True, capture_output=True)
        # out = subprocess.run(f"ssh {self.context} 'docker inspect {self.docker_tag}'", shell=True, capture_output=True)

        if out.returncode == 0:
            print("Image already exists in remote context")
            return

        print("running", f"docker buildx build -t {self.docker_tag} -f {docker_file} {docker_scripts_dir}")
        subprocess.run(f"docker --context default buildx build -t {self.docker_tag} -f {docker_file} {docker_scripts_dir}", shell=True)
        print("running", f"docker --context default save {self.docker_tag} | docker --context {self.context} load")
        subprocess.run(f"docker --context default save {self.docker_tag} | docker --context {self.context} load", shell=True)
        # print("running",
        #       f"docker --context {self.context} save {self.docker_tag} | bzip2 | pv | ssh {self.context} 'bunzip2 | docker load'")
        # out = subprocess.run(f"docker save {self.docker_tag} | bzip2 | pv | ssh {self.context} 'bunzip2 | docker load'", shell=True,
        #                      capture_output=True)
        print("done")


def get_executor(url):
    url = urllib.parse.urlparse(url)
    if not url.scheme:
        try:
            executor = SSHSLURMExecutor(url)
        except SSHException as e:
            raise SSHException(
                "SSH connection failed!,"
                f"Make sure you can successfully run `ssh {url}` with no parameters, "
                f"any parameters should be set in the ssh_config file"
            )
        executor.run("")
    elif url.scheme == "ssh":
        executor = SSHExecutor(url)
    elif url.scheme == "local":
        raise NotImplementedError
    elif url.scheme == "docker":
        executor = DockerExecutor(url)
    else:
        raise NotImplementedError
    return executor
