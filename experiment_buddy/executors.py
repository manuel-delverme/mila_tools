import abc
import logging
import os
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import uuid
import warnings
from typing import Optional

import fabric
from invoke import UnexpectedExit, Result
from paramiko.ssh_exception import SSHException

import experiment_buddy.utils

wandb_escape = "^"
tb = tensorboard = None
if os.path.exists("buddy_scripts/"):
    SCRIPTS_FOLDER = "buddy_scripts/"
else:
    SCRIPTS_FOLDER = os.path.join(os.path.dirname(__file__), "../scripts/")
ARTIFACTS_PATH = "runs/"
DEFAULT_WANDB_KEY = os.path.join(os.environ["HOME"], ".netrc")
GIT_CLONE_PREFIX = "GIT_SSH_COMMAND=\"ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no\""


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

    @abc.abstractmethod
    def put(self, local_path, remote_path):
        raise NotImplementedError

    @abc.abstractmethod
    def launch_job(self, git_url, entrypoint, hash_commit, extra_modules, conda_env):
        raise NotImplementedError

    @abc.abstractmethod
    def sweep_agent(self, git_url, hash_commit, extra_modules, sweep_id, conda_env):
        raise NotImplementedError


class SSHExecutor(Executor):
    def __init__(self, url: urllib.parse.ParseResult):
        try:
            self.ssh_session = fabric.Connection(host=url.hostname, connect_timeout=10, forward_agent=True)
            for _ in range(10):
                try:
                    self.ssh_session.run("")
                    break
                except Exception as e:
                    print(e, "Retrying in 5 seconds...")
                    time.sleep(5)
        except SSHException as e:
            raise SSHException(
                "SSH connection failed!,"
                f"Make sure you can successfully run `ssh {url.hostname}` with no parameters, "
                f"any parameters should be set in the ssh_config file"
            )
        # self.scripts_dir = None
        self.scripts_folder = None
        self.working_dir = None

    def run(self, cmd) -> Result:
        print("Running:\n", cmd)
        out = self.ssh_session.run(cmd)
        print(out)
        return out

    def put(self, local_path, remote_path):
        return self.ssh_session.put(local_path, remote_path)

    def launch_job(self, git_url, entrypoint, hash_commit, extra_modules, conda_env):
        ssh_command = f"bash -l {self.scripts_folder}/run_experiment.sh {git_url} {entrypoint} {hash_commit} {conda_env} {extra_modules}"
        print(ssh_command)
        self.ssh_session.run(ssh_command)

    def _pull_experiment(self, git_url, hash_commit, folder):
        logging.info(f"downloading source code from {git_url} to {folder}")
        self.run(f'{GIT_CLONE_PREFIX} git clone {git_url} {folder}/')
        with self.ssh_session.cd(folder):
            self.run(f'git checkout {hash_commit}')

    def remote_checkout(self, git_url, hash_commit):
        experiments_folder = '$HOME/experiments/'
        self.run(f'mkdir -p {experiments_folder}')
        with self.ssh_session.cd(experiments_folder):
            result = self.run('mktemp -p . -d')
            experiment_folder = result.stdout.strip()
            self._pull_experiment(git_url, hash_commit, experiment_folder)
            return f"{experiments_folder}/{experiment_folder}"

    def sweep_agent(self, git_url, hash_commit, extra_modules, sweep_id, conda_env):
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
        self.check_or_copy_wandb_key()

        # Check if github-personal is in the ssh config
        try:
            self.ssh_session.run("cat ~/.ssh/config | grep github-personal")
        except UnexpectedExit:
            self.ssh_session.run("echo 'Host github-personal\n\tHostName github.com\n' >> ~/.ssh/config")

        if self.scripts_folder:
            return

        retr = self.ssh_session.run("mktemp -d -t experiment_buddy-XXXXXXXXXX")
        remote_tmp_folder = retr.stdout.strip() + "/"

        # Create temp local folder
        with tempfile.TemporaryDirectory() as scripts_folder:
            with tarfile.open(f'{scripts_folder}/scripts_folder.tar.gz', 'w:gz') as tar:
                tar.add(SCRIPTS_FOLDER, arcname=os.path.basename(SCRIPTS_FOLDER))

            self.ssh_session.put(f'{scripts_folder}/scripts_folder.tar.gz', remote_tmp_folder + "scripts_folder.tar.gz")
            self.ssh_session.run(f'tar -xzf {remote_tmp_folder}scripts_folder.tar.gz -C {remote_tmp_folder}')

        self.scripts_folder = remote_tmp_folder


class HetznerExecutor(SSHExecutor):
    def __init__(self, url):
        requested_machine = url.hostname
        from hcloud import Client
        from hcloud.server_types.domain import ServerType
        from hcloud.images.domain import Image
        self.hclient = Client(token=os.environ["HCLOUD_TOKEN"])
        # TODO: check if the machine is already available
        # servers = self.hclient.servers.get_all()
        ssh_keys = self.hclient.ssh_keys.get_all()
        for key in ssh_keys:
            key.data_model.id = None  # Yay bugs, this causes a fallback to the name, which works... unlike the id.

        # name = "auto-experiment-buddy-" + str(uuid.uuid4()),
        response = self.hclient.servers.create(
            name=str(uuid.uuid4()),
            server_type=ServerType(name=requested_machine),
            image=Image(name="ubuntu-20.04"),
            ssh_keys=self.hclient.ssh_keys.get_all(),
        )
        new_host = response.server.data_model.public_net.ipv4.ip
        for a in response.next_actions:
            print("Waiting for server to be ready...", a.command)
            a.wait_until_finished()

        url = url._replace(netloc=new_host)
        super().__init__(url)

    def setup_remote(self, extra_slurm_header: Optional[str], working_dir: str):
        if extra_slurm_header:
            raise NotImplementedError

        self.working_dir = working_dir
        self._ensure_scripts_directory(self.working_dir)


class AwsExecutor(SSHExecutor):
    def __init__(self, url):
        import boto3
        requested_machine = url.hostname

        # Create an EC2 client
        client = boto3.client('ec2', region_name='us-east-1')

        # find the right image for the requested_machine
        description = None
        args = dict(Filters=[{'Name': 'instance-type', 'Values': [requested_machine, ]}, ], )
        while True:
            response = client.describe_instance_types(**args)
            if response["InstanceTypes"]:
                description, = response["InstanceTypes"]
                break

            if "NextToken" not in response:
                break
            args["NextToken"] = response["NextToken"]

        supported_architectures = description["ProcessorInfo"]["SupportedArchitectures"]
        print("Selected architecture", supported_architectures)

        # we want Canonical, Ubuntu, 22.04 LTS
        response = client.describe_images(
            Filters=[
                {'Name': 'description', 'Values': [f"*Canonical, Ubuntu, 22.04 LTS*", ]},
                {'Name': 'architecture', 'Values': supported_architectures, },
                {'Name': 'owner-alias', 'Values': ['amazon', ]}
            ],
        )

        # Get the latest Ubuntu 22.04 image from Canonical
        images = response['Images']
        images.sort(key=lambda x: x['CreationDate'], reverse=True)
        image_id = images[0]['ImageId']
        print("Selected image", image_id)

        # Launch the instance, allow ssh access from anywhere
        # --instance-initiated-shutdown-behavior terminate
        response = client.run_instances(
            ImageId=image_id,
            KeyName="us1",
            InstanceType=requested_machine,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=['sg-080085f27a7461608', ],
            InstanceInitiatedShutdownBehavior='terminate',
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': 'experiment-buddy'
                        },
                    ]
                },
            ],
        )

        istance, = response['Instances']
        instance_id = istance['InstanceId']

        # Wait until the instance is running
        client.get_waiter('instance_running').wait(InstanceIds=[instance_id, ])

        response = client.describe_instances(InstanceIds=[instance_id, ])
        reservation, = response["Reservations"]
        istance, = reservation["Instances"]
        public_dns_name = istance['PublicDnsName']

        print(public_dns_name)

        url = url._replace(netloc=public_dns_name)
        super().__init__(url)

    def setup_remote(self, extra_slurm_header: Optional[str], working_dir: str):
        if extra_slurm_header:
            raise NotImplementedError

        self.working_dir = working_dir
        self._ensure_scripts_directory(self.working_dir)

    def sweep_agent(self, git_url, hash_commit, extra_modules, sweep_id, conda_env):
        ssh_command = f"bash -l {self.scripts_folder}/run_sweep.sh {git_url} {sweep_id} {hash_commit} {conda_env} {extra_modules}"
        print(ssh_command)
        self.ssh_session.run(ssh_command)


class SSHSLURMExecutor(Executor):
    def __init__(self, url):
        ensure_torch_compatibility()
        for trial_idx in range(3):
            try:
                self.ssh_session = fabric.Connection(host=url.path, connect_timeout=10, forward_agent=False)
                self.ssh_session.run("")
                break
            except Exception as e:
                print(f"Failed to connect to the remote machine, retrying in {trial_idx * 5} seconds", e)
                time.sleep(trial_idx * 5)
        self.scritps_folder = None
        self.extra_slurm_header = None
        self.working_dir = None

    def run(self, cmd):
        return self.ssh_session.run(cmd)

    def put(self, local_path, remote_path):
        return self.ssh_session.put(local_path, remote_path)

    def launch_job(self, git_url, entrypoint, hash_commit, extra_modules, conda_env):
        ssh_command = f"bash -l {self.scripts_folder}/run_experiment.sh {git_url} {entrypoint} {hash_commit} {conda_env} {extra_modules}"
        print("Running", ssh_command)
        self.ssh_session.run(ssh_command)
        time.sleep(1)

    def sweep_agent(self, git_url, hash_commit, extra_modules, sweep_id, conda_env):
        ssh_command = f"source /etc/profile; sbatch {self.scripts_folder}/run_sweep.sh {git_url} {sweep_id} {hash_commit} {conda_env} {extra_modules}"
        print(ssh_command)
        self.run(ssh_command)

    def _check_or_copy_wandb_key(self):
        try:
            self.ssh_session.run("test -f $HOME/.netrc")
        except UnexpectedExit:
            print(f"Wandb api key not found in {self.ssh_session.host}. Copying it from {DEFAULT_WANDB_KEY}")
            self.ssh_session.put(DEFAULT_WANDB_KEY, ".netrc")

    def setup_remote(self, extra_slurm_header: str, working_dir: str):
        self.extra_slurm_header = extra_slurm_header
        self.working_dir = working_dir
        self._check_or_copy_wandb_key()
        self._ensure_scripts_directory(self.extra_slurm_header, self.working_dir)

    def _ensure_scripts_directory(self, extra_slurm_header: str, working_dir: str):
        retr = self.ssh_session.run("mktemp -d -t -p /network/scratch/d/delvermm/ experiment_buddy-XXXXXXXXXX")
        remote_tmp_folder = retr.stdout.strip() + "/"
        backend = experiment_buddy.utils.get_backend(self.ssh_session, working_dir)
        scripts_dir = os.path.join(SCRIPTS_FOLDER, backend)

        for file in os.listdir(scripts_dir):
            if extra_slurm_header and file in ("run_sweep.sh", "srun_python.sh"):
                new_tmp_file = _insert_extra_header(extra_slurm_header, os.path.join(scripts_dir, file))
                self.ssh_session.put(new_tmp_file, remote_tmp_folder)
            else:
                self.ssh_session.put(os.path.join(scripts_dir, file), remote_tmp_folder)

        self.scripts_folder = remote_tmp_folder


class HydraExecutor(Executor):
    def __init__(self, url):
        ensure_torch_compatibility()
        for trial_idx in range(3):
            try:
                self.ssh_session = fabric.Connection(host=url.hostname, connect_timeout=10, forward_agent=False)
                self.ssh_session.run("")
                break
            except Exception as e:
                print(f"Failed to connect to the remote machine, retrying in {trial_idx * 5} seconds", e)
                time.sleep(trial_idx * 5)
        self.scritps_folder = None
        self.extra_slurm_header = None
        self.working_dir = None

    def run(self, cmd):
        return self.ssh_session.run(cmd)

    def put(self, local_path, remote_path):
        return self.ssh_session.put(local_path, remote_path)

    def launch_job(self, git_url, entrypoint, hash_commit, extra_modules, conda_env):
        ssh_command = f"bash -l {self.scripts_folder}/run_experiment.sh {git_url} {entrypoint} {hash_commit} {conda_env} {extra_modules}"
        print("Running", ssh_command)
        self.ssh_session.run(ssh_command)
        time.sleep(1)

    def sweep_agent(self, git_url, hash_commit, extra_modules, sweep_id, conda_env):
        ssh_command = f"source /etc/profile; sbatch {self.scripts_folder}/run_sweep.sh {git_url} {sweep_id} {hash_commit} {conda_env} {extra_modules}"
        print(ssh_command)
        self.run(ssh_command)

    def _check_or_copy_wandb_key(self):
        try:
            self.ssh_session.run("test -f $HOME/.netrc")
        except UnexpectedExit:
            print(f"Wandb api key not found in {self.ssh_session.host}. Copying it from {DEFAULT_WANDB_KEY}")
            self.ssh_session.put(DEFAULT_WANDB_KEY, ".netrc")

    def setup_remote(self, extra_slurm_header: str, working_dir: str):
        self.extra_slurm_header = extra_slurm_header
        self.working_dir = working_dir
        self._check_or_copy_wandb_key()
        self._ensure_scripts_directory(self.extra_slurm_header, self.working_dir)

    def _ensure_scripts_directory(self, extra_slurm_header: str, working_dir: str):
        retr = self.ssh_session.run("mktemp -d -t -p /network/scratch/d/delvermm/ experiment_buddy-XXXXXXXXXX")
        remote_tmp_folder = retr.stdout.strip() + "/"

        scripts_dir = os.path.join(SCRIPTS_FOLDER, "hydra")

        for file in os.listdir(scripts_dir):
            self.ssh_session.put(os.path.join(scripts_dir, file), remote_tmp_folder)

        self.scripts_folder = remote_tmp_folder


class DockerExecutor(Executor):
    EXPERIMENT_PATH = "/experiment"

    def __init__(self, url):
        import docker
        self.docker_client = docker.from_env()
        self.docker_tag = f"base-cpu:latest"
        self.context = url.hostname or "default"
        self.repo_archive = None

    def run(self, cmd, container_id=None, environment=None):
        image = container_id or self.docker_tag

        environment = environment or {}  # TODO: overwrites the environment?
        environment.update({"DOCKER_CONTEXT": self.context})
        self.docker_client.containers.run(image, cmd, environment=environment, remove=True)

    def launch_job(self, git_url, entrypoint, hash_commit, extra_modules, conda_env):
        raise NotImplementedError
        self.maybe_pack_archive(git_url, hash_commit)

        out = subprocess.run(
            f"docker --context {self.context} run --name {hash_commit} --rm -it -t {self.docker_tag} sleep infinity",
            shell=True,
            capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(f"Error launching docker container: {out.stderr}")

        # container = self.docker_client.containers.create(self.docker_tag, "sleep infinity", detach=True)
        # docker cp archive.tar container_name:/path/to/destination
        out = subprocess.run(
            f"docker --context {self.context} cp {self.repo_archive} {hash_commit}:{self.EXPERIMENT_PATH}", shell=True,
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

        out = subprocess.run(f"docker --context {self.context} inspect {self.docker_tag}", shell=True,
                             capture_output=True)
        # out = subprocess.run(f"ssh {self.context} 'docker inspect {self.docker_tag}'", shell=True, capture_output=True)

        if out.returncode == 0:
            print("Image already exists in remote context")
            return

        print("running", f"docker buildx build -t {self.docker_tag} -f {docker_file} {docker_scripts_dir}")
        subprocess.run(
            f"docker --context default buildx build -t {self.docker_tag} -f {docker_file} {docker_scripts_dir}",
            shell=True)
        print("running", f"docker --context default save {self.docker_tag} | docker --context {self.context} load")
        subprocess.run(f"docker --context default save {self.docker_tag} | docker --context {self.context} load",
                       shell=True)
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
    elif url.scheme == "ssh":
        executor = SSHExecutor(url)
    elif url.scheme == "hetzner":
        executor = HetznerExecutor(url)
    elif url.scheme == "aws":
        executor = AwsExecutor(url)
    elif url.scheme == "hydra":
        executor = HydraExecutor(url)
    elif url.scheme == "local":
        raise NotImplementedError
    elif url.scheme == "docker":
        executor = DockerExecutor(url)
    else:
        raise NotImplementedError
    return executor
