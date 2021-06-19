#! /bin/bash

set -e
source /etc/profile

# Module system
function log() {
  echo -e "\e[32m\"[DEPLOY LOG] $*\"\e[0m"
}

log "Refreshing modules..."
module purge
module load python/3.7

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

  EXPERIMENT_FOLDER=$(mktemp -p . -d)
  pull_experiment $GIT_URL $HASH_COMMIT $EXPERIMENT_FOLDER
}

function set_up_venv() {
  VENV_NAME=$1

  if ! source $HOME/venv/bin/activate; then
    log "venv not found, setting up venv @ $HOME/venv..."
    python3 -m venv $HOME/"venv"
    source $HOME/venv/bin/activate
  fi

  log "Using shared venv @ $HOME/venv"

  python3 -m pip install --upgrade pip
  python3 -m pip install -r "requirements.txt" --exists-action w | grep -v "Requirement already satisfied"
}

log "running on $(hostname)"

GIT_URL=$1
ENTRYPOINT=$2
HASH_COMMIT=$3

load_git_folder $GIT_URL $HASH_COMMIT
set_up_venv "venv"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/

# TODO: the client should send the mila_tools version to avoid issues
log "/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $ENTRYPOINT"
/opt/slurm/bin/sbatch --comment "$(git log --format=%B -n 1)" $SCRIPTS_FOLDER/srun_python.sh $ENTRYPOINT
