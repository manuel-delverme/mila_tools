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

FOLDER=$SLURM_TMPDIR/src/

log "downloading source code from $1 to $FOLDER"
git clone $1 $FOLDER/
cd $FOLDER || exit
git checkout $3
log "pwd is now $(pwd)"

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $SLURM_TMPDIR/venv..."
python3 -m virtualenv "$SLURM_TMPDIR/venv"
# shellcheck disable=SC1090
source "$SLURM_TMPDIR/venv/bin/activate"
python3 -m pip install --upgrade pip

log "Downloading modules"
sh $HOME/install_jax.sh # TODO: move this to experiment_buddy
python3 -m pip install -r "requirements.txt" --exists-action w

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the experiment_buddy version to avoid issues
wandb agent "$2"
