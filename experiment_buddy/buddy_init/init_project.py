import logging
import shutil
import sys

import experiment_buddy.buddy_init.init_actions

logger = logging.getLogger(__name__)


def preflight_check():
    if shutil.which('ssh') is None:
        raise Exception('missing ssh, you can install it with "sudo apt install openssl-client"')
    if shutil.which('git') is None:
        raise Exception('missing git, you can install it with "sudo apt install git"')
    if shutil.which('pip') is None:
        raise Exception('missing pip, you can install it with "sudo apt install python-pip"')


def init():
    preflight_check()
    experiment_buddy.buddy_init.init_actions.setup_ssh()
    experiment_buddy.buddy_init.init_actions.setup_wandb()
    experiment_buddy.buddy_init.init_actions.setup_github()
    logger.info("experiment buddy is all set and ready to help.")


def sys_main():
    return_code = 1
    try:
        init()
        return_code = 0
    except Exception as e:
        import traceback
        print(f'Error: {e}', file=sys.stderr)

    sys.exit(return_code)
