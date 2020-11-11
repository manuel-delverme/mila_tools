#! /bin/bash
#SBATCH --job-name=spython
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --time=2-00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=main
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

log "cd $HOME/experiments/"
cd $HOME/experiments/ || exit

FOLDER=$(mktemp -p . -d)
log "FOLDER=$FOLDER"

log "downloading source code from $1 to $FOLDER"
git clone $1 $FOLDER/
cd $FOLDER || exit
git checkout $3
log "pwd is now $(pwd)"

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $FOLDER/venv..."
python -m virtualenv "$FOLDER/venv"
# shellcheck disable=SC1090
source "$FOLDER/venv/bin/activate"
python -m pip install --upgrade pip

log "Downloading modules"
sh $HOME/install_jax.sh # TODO: move this to mila_tools
python -m pip install -r "requirements.txt" --exists-action w

log "/opt/slurm/bin/sbatch $HOME/srun_python.sh $2"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the mila_tools version to avoid issues
python -O -u $2
