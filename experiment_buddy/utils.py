import time
import invoke

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


def check_if_has_slurm(ssh_session):
    try:
        has_slurm = ssh_session.run("/opt/slurm/bin/scontrol ping")
        return has_slurm
    except invoke.exceptions.UnexpectedExit:
        return False
