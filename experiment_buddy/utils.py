import invoke
import os
from git.util import IterableList

def check_if_has_slurm(ssh_session):
    try:
        has_slurm = ssh_session.run("/opt/slurm/bin/scontrol ping")
    except invoke.exceptions.UnexpectedExit:
        return False

    return has_slurm.ok

def project_name(git_repo):
    git_repo_remotes = git_repo.remotes
    if isinstance(git_repo_remotes, IterableList):
        remote_url = git_repo_remotes.config_reader.get("url")
        git_project_name = os.path.splitext(os.path.basename(remote_url))[0]
        return git_project_name

