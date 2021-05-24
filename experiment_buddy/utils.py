import enum
import os
import time

import fabric
import git
import invoke
import requests


class Backend(enum.Enum):
    GENERAL: str = "general"
    SLURM: str = "slurm"
    DOCKER: str = "docker"


def get_backend(ssh_session: fabric.connection.Connection, project_dir: str) -> str:
    try:
        ssh_session.run("/opt/slurm/bin/scontrol ping")
        return Backend.SLURM
    except invoke.exceptions.UnexpectedExit:
        pass

    if not os.path.exists(os.path.join(project_dir, "Dockerfile")):
        return Backend.GENERAL
    try:
        ssh_session.run("docker -v")
        ssh_session.run("docker-compose -v")
        return Backend.DOCKER
    except invoke.exceptions.UnexpectedExit:
        pass
    return Backend.GENERAL


def get_project_name(git_repo: git.Repo) -> str:
    git_repo_remotes = git_repo.remotes
    assert isinstance(git_repo_remotes, list)
    remote_url = git_repo_remotes[0].config_reader.get("url")
    project_name, _ = os.path.splitext(os.path.basename(remote_url))
    return project_name


def telemetry(f):
    def wrapped_f(*args, **kwargs):
        tic = time.perf_counter()
        retr = f(*args, **kwargs)
        toc = time.perf_counter()
        method_name = f.__name__
        requests.get(f"http://65.21.155.92/{method_name}/{toc - tic}", timeout=2.)
        return retr

    return wrapped_f
