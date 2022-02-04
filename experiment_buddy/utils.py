import os
import subprocess

import fabric
import git
import invoke


def get_backend(ssh_session: fabric.connection.Connection, project_dir: str) -> str:
    try:
        ssh_session.run("/opt/slurm/bin/scontrol ping")
        return "slurm"
    except invoke.exceptions.UnexpectedExit:
        pass

    if not os.path.exists(os.path.join(project_dir, "Dockerfile")):
        return "general"
    try:
        ssh_session.run("docker -v")
        ssh_session.run("docker-compose -v")
        return "docker"
    except invoke.exceptions.UnexpectedExit:
        pass
    return "general"


def get_project_name(git_repo: git.Repo) -> str:
    git_repo_remotes = git_repo.remotes
    assert isinstance(git_repo_remotes, list)
    if not len(git_repo_remotes):
        raise Exception("No remote url found, have you pushed the new repo?")
    remote_url = git_repo_remotes[0].config_reader.get("url")
    project_name, _ = os.path.splitext(os.path.basename(remote_url))
    return project_name


def _get_job_info(jid):
    result = subprocess.check_output(f"/opt/slurm/bin/sacct --brief -j {jid}".split()).decode()
    jid, state, exit_code = result.split("\n")[2].split()
    return state
