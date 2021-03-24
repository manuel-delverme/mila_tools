import invoke
import os
from git.util import IterableList
import git


def check_if_has_slurm(ssh_session):
    try:
        has_slurm = ssh_session.run("/opt/slurm/bin/scontrol ping")
    except invoke.exceptions.UnexpectedExit:
        return False

    return has_slurm.ok


def get_project_name(git_repo):
    git_repo_remotes = git_repo.remotes
    assert isinstance(git_repo_remotes, IterableList)
    remote_url = git_repo_remotes[0].config_reader.get("url")
    project_name, _ = os.path.splitext(os.path.basename(remote_url))
    return project_name
