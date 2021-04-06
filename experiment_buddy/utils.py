import enum
import os

import invoke
from OpenSSL.SSL import Connection
import git


class Backend(enum.Enum):
    GENERAL: str = "general"
    SLURM: str = "slurm"
    DOCKER: str = "docker"


def get_backend(ssh_session: Connection, project_dir: str) -> str:
    try:
        ssh_session.run("/opt/slurm/bin/scontrol ping")
        return Backend.SLURM
    except invoke.exceptions.UnexpectedExit:
        pass

    if not os.path.exists(os.path.join(project_dir, "Dockerfile")):
        return Backend.GENERAL
    try:
        ssh_session.run("docker - v")
        return Backend.DOCKER
    except invoke.exceptions.UnexpectedExit:
        pass
    return Backend.GENERAL


def get_project_name(git_repo: git.Repo) -> str:
    git_repo_remotes = git_repo.remotes
    assert isinstance(git_repo_remotes, git.util.IterableList)
    remote_url = git_repo_remotes[0].config_reader.get("url")
    project_name, _ = os.path.splitext(os.path.basename(remote_url))
    return project_name
