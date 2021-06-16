#! /bin/bash
#SBATCH --job-name=spython
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=2-00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --get-user-env=L

set -e

GIT_URL=$1
SWEEP_ID=$2
HASH_COMMIT=$3

source /etc/profile
module purge
module load python/3.7
module load pytorch/1.7

# Module system
function log() {
  echo -e "\e[32m\"[DEPLOY LOG] $*\"\e[0m"
}

function pull_experiment() {
  GIT_URL=$1
  HASH_COMMIT=$2
  FOLDER=$3
  log "downloading source code from $GIT_URL to $FOLDER"
  git clone -q "$GIT_URL" "$FOLDER"/
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
    log "Setting up venv @ $HOME/venv..."
    python3 -m virtualenv $HOME/"venv"
    source $HOME/venv/bin/activate
  fi

  log "Using shared venv @ $HOME/venv"

  python3 -m pip -q install --upgrade pip

  python3 -m pip -q install -r "requirements.txt" --exists-action w
}
FOLDER=$SLURM_TMPDIR/src/

pull_experiment $GIT_URL $HASH_COMMIT $FOLDER

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $SLURM_TMPDIR/venv..."

python3 -m virtualenv --system-site-packages "$SLURM_TMPDIR/venv"
# shellcheck disable=SC1090
source "$SLURM_TMPDIR/venv/bin/activate"
python3 -m pip -q install --upgrade pip

log "Downloading modules"
python3 -m pip -q install -r "requirements.txt" --exists-action w -f https://download.pytorch.org/whl/torch_stable.html

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the experiment_buddy version to avoid issues
wandb agent "$SWEEP_ID"
