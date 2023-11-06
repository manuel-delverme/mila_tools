#! /bin/bash
#SBATCH --job-name=spython
#SBATCH --time=12-00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
set -e

function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}

source /etc/profile
log "Refreshing modules..."
module purge

GIT_URL=$1
ENTRYPOINT=$2
HASH_COMMIT=$3
CONDA_ENV=$4
EXTRA_MODULES=$(echo $5 | tr "@" " ")

# /tmp/experiment_buddy-CplQ5vtnzn//run_sweep.sh  delvermm/option-base/ukjy4sv6 094e0a0ecf6b7dc444cceb361c8cca7f9c843a75 python/3.7@pytorch/1.7
for MODULE in $EXTRA_MODULES; do
  module load $MODULE
done

FOLDER=$SLURM_TMPDIR/src/

log "downloading source code from $GIT_URL to $FOLDER"
# https://stackoverflow.com/questions/7772190/passing-ssh-options-to-git-clone/28527476#28527476
GIT_SSH_COMMAND="ssh -v -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone $GIT_URL $FOLDER/
cd $FOLDER || exit
git checkout $HASH_COMMIT
log "pwd is now $(pwd)"

log "trying to use conda env $CONDA_ENV"
if [ ! -z "$CONDA_ENV" ]; then
  module load anaconda/3 || true
  if conda activate "$CONDA_ENV"; then
    echo "Successfully activated conda environment '$CONDA_ENV'"
  else
    echo "Failed to activate conda environment '$CONDA_ENV'"
  fi
fi

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
# log "Setting up venv @ $SLURM_TMPDIR/venv..."

python3 -m venv --system-site-packages "$SLURM_TMPDIR/venv"
# shellcheck disable=SC1090

source "$SLURM_TMPDIR/venv/bin/activate"
# python3 -m pip install --upgrade pip
python3 -m pip install pip==22.2

log "Downloading modules"
python3 -m pip install --upgrade -r "requirements.txt" --exists-action w -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html --use-deprecated=legacy-resolver

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the experiment_buddy version to avoid issues

wandb agent "$ENTRYPOINT"
