import logging
import os
import shutil
import subprocess

import fabric
import paramiko
from invoke import UnexpectedExit

SSH_TIMEOUT = 10
logger = logging.getLogger(__name__)


def setup_ssh():
    config_file = os.path.expanduser('~/.ssh/config')
    if os.path.exists(config_file):
        shutil.copy(config_file, config_file + ".bak")
        logger.info(f"Found config file. Saved a copy as {config_file}.bak.")
    else:
        logger.info("SSH config file not found. Creating a new one.")
        with open(config_file, "w"):
            pass

    with open(config_file) as fin:
        current_config = fin.read()

    if 'Host mila' not in current_config:
        logger.info("One time setup to connect with the Mila servers")
        wrong_username = True
        while wrong_username:
            mila_username = input("What is your mila username?")
            wrong_username = enforce_answer_list(f"Your cluster username is: {mila_username}, correct?",
                                                 ["y", "n"]) == "n"

        with open(config_file, 'a') as fout:
            fout.write(
                "\n\nHost mila\n"
                "    Hostname         login.server.mila.quebec\n"
                "    Port 2222\n"
                f"    User {mila_username}\n"
                "    PreferredAuthentications publickey,keyboard-interactive\n"
                "    ServerAliveInterval 120\n"
                "    ServerAliveCountMax 5\n\n")

    if not os.path.exists(os.path.expanduser("~/.ssh/id_rsa.pub")):
        logger.info("Keys not found. Generate keys:")
        retr = subprocess.run("ssh-keygen", shell=True)
        logger.info(retr.stdout)
        logger.info("Copy key pair to your account.")
        retr = subprocess.run("ssh-copy-id mila", shell=True)
        logger.info(retr.stdout)

    try:
        logger.info("Checking cluster connection...")
        with fabric.Connection(host='mila', connect_timeout=SSH_TIMEOUT) as connection:
            mila_username = connection.ssh_config.get('user')
            connection.run("")
        logger.info(f"SSH for user: {mila_username} \u2713.")
    except paramiko.ssh_exception.SSHException as e:
        raise Exception(f"""
Error while checking SSH connection, stopping
 - Double check that your username is '{mila_username}'?
 - Can you successfully connect as "ssh mila"?
 - Setup the public and private key for you and the mila cluster?
Raised Exception:
""" + str(e))


def setup_wandb():
    logger.info("Checking wandb connection.")
    retr = subprocess.run("python -m wandb login", shell=True)
    if retr.returncode != 0 or not os.path.exists(os.path.expanduser("~/.netrc")):
        logging.error("Something went wrong with wandb setup, please ask for help \u274c.")
    else:
        logger.info("wandb \u2713.")


def enforce_answer_list(question, possible_answers):
    possible_answers = [pa.lower() for pa in possible_answers]
    text = f"{question} [{'|'.join(possible_answers)}]"
    answer = input(text).lower()
    if possible_answers and answer not in possible_answers:
        while answer := input(text).lower() not in possible_answers:
            pass
    return answer


def setup_github():
    logger.info("Checking remote github account")
    retr = fabric.Connection(host='mila').run("ssh -T git@github.com 2>&1", shell=True, warn=True)
    if "successfully authenticated" in retr.stdout:
        logger.info("github \u2713.")
        return

    logger.info("Github auth failed. We will verify your ssh keys.")

    with fabric.Connection(host='mila', connect_timeout=SSH_TIMEOUT) as remote_connection:
        try:
            retr = remote_connection.run("cat ~/.ssh/id_rsa.pub", hide=True)
        except UnexpectedExit:
            logger.info("Public key not found. Generate new key pair ...")
            remote_connection.run("ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa", hide=True)
            logger.info("Sending Public key to the cluster...")
            retr = remote_connection.run("cat .ssh/id_rsa.pub", hide=True)
            logger.info("You key successfully added.")
        else:
            logger.info("id_rsa.pub found on the cluster")

    logger.info("\nFollow these instructions to add your cluster public to your github account.")
    if retr.exited != 0:
        raise Exception("Failed to generate ssh keys. Check your ~/.ssh/id_rsa.")
    if retr.stdout.startswith("ssh-rsa"):
        logger.info(
            "Navigate to https://github.com/settings/ssh/new and add your cluster public key, to allow the cluster access to private repositories")
        logger.info("Give it a title such as \"Mila cluster\". The key is:")
        print(retr.stdout.strip())
