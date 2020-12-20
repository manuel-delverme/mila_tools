import os
import git
import time
from .utils import timeit


@timeit
def git_sync(experiment_id, git_repo):

    # experiment_buddy_branch_name = f"{git_repo.active_branch.name}_experiment_buddy_log"
    # experiment_buddy_branch = git_repo.create_head(experiment_buddy_branch_name)
    # experiment_buddy_branch.checkout()

    # 2) commits everything to git with the name as message (so i r later reproduce the same experiment)
    os.system(f"git add .")
    os.system(f"git commit -m '{experiment_id}'")
    # TODO: ideally the commits should go to a parallel branch so the one in use is not filled with versioning checkpoints
    # 3) pushes the changes to git
    os.system("git push")  # TODO: only if commit
    code_version = git_repo.commit().hexsha
    return code_version
