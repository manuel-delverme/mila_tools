#!/bin/bash

set -e
source /etc/profile
echo "Using $SHELL as shell"

function log() {
  echo -e "\e[32m\"[DEPLOY LOG] $*\"\e[0m"
  echo -e "[DEPLOY LOG] $*" >>$HOME/last_buddy_run_experiment_log.txt
}

function pull_experiment() {
  GIT_URL=$1
  HASH_COMMIT=$2
  FOLDER=$3
  log "downloading source code from $GIT_URL to $FOLDER"
  # https://stackoverflow.com/questions/7772190/passing-ssh-options-to-git-clone/28527476#28527476
  GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone "$GIT_URL" "$FOLDER"/
  cd "$FOLDER" || exit
  git checkout "$HASH_COMMIT"
  log "pwd is now $(pwd)"
}

function load_git_folder() {
  SCRIPT=$(realpath $0)
  GIT_URL=$1
  HASH_COMMIT=$2

  log "script realpath: $SCRIPT"
  SCRIPTS_FOLDER=$(dirname "$SCRIPT")
  log "scripts home: $SCRIPTS_FOLDER"

  log "cd $HOME/experiments/"
  mkdir -p "$HOME"/experiments/
  cd "$HOME"/experiments/

  EXPERIMENT_FOLDER="$(mktemp -p . -d)"

  pull_experiment $GIT_URL $HASH_COMMIT $EXPERIMENT_FOLDER
  EXPERIMENT_NAME=$(git log --format=%B -n 1)
  cd ".."
  mv "$EXPERIMENT_FOLDER" "$EXPERIMENT_FOLDER-$EXPERIMENT_NAME"
  cd "$EXPERIMENT_FOLDER-$EXPERIMENT_NAME"
}

log "running on $(hostname)"
GIT_URL=$1
ENTRYPOINT=$2
HASH_COMMIT=$3

load_git_folder $GIT_URL $HASH_COMMIT

if ! source $HOME/venv/bin/activate; then
  log "venv not found, setting up venv @ $HOME/venv..."
  python3 -m venv $HOME/"venv"
  source $HOME/venv/bin/activate
fi

log "Using shared venv @ $HOME/venv"

log "Upgrading pip"
python3 -m pip install --upgrade pip
log "Upgrading requirements"

log "Upgrading requirements"
python3 -m pip cache purge
python3 -m pip install --upgrade -r "requirements.txt" --exists-action s -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html | grep -v "Requirement already satisfied"
log "Requirements upgraded"

SESSION_NAME=my_session_$(date +%Y%m%d%H%M%S)

log "Starting screen session: $SESSION_NAME"
screen -dmS $SESSION_NAME bash -c "python3 $ENTRYPOINT --multirun hydra/launcher=submitit_slurm; bash -i"
log "Detached screen session: $SESSION_NAME"
