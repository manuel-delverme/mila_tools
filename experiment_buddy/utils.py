import invoke


def check_if_has_slurm(ssh_session):
    try:
        has_slurm = ssh_session.run("/opt/slurm/bin/scontrol ping")
    except invoke.exceptions.UnexpectedExit:
        return False

    return has_slurm.ok
