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

# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}

source /etc/profile
log "Refreshing modules..."
module purge
module load python/3.8
module load cuda/10.1/cudnn/7.6

# log "cd $HOME/experiments/"
# cd $HOME/experiments/

# FOLDER=$(mktemp -p . -d)
FOLDER=$SLURM_TMPDIR/src/

log "downloading source code from $1 to $FOLDER"
git clone $1 $FOLDER/
cd $FOLDER || exit
git checkout $3
log "pwd is now $(pwd)"

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $SLURM_TMPDIR/venv..."
python -m virtualenv "$SLURM_TMPDIR/venv"
# shellcheck disable=SC1090
source "$SLURM_TMPDIR/venv/bin/activate"
python -m pip install --upgrade pip

# log "Using shared venv @ $HOME/venv"
# source $HOME/venv/bin/activate

log "Downloading modules"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
sh $HOME/install_jax.sh # TODO: move this to mila_tools
python -m pip install -r "$HOME/requirements.txt" --exists-action w

# TODO: the client should send the mila_tools version to avoid issues
python -m pip install --upgrade git+https://github.com/manuel-delverme/mila_tools/

wandb agent "$2"
