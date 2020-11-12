#! /bin/bash
# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}
SCRIPTS_FOLDER=$(dirname realpath $0)
log "scripts home: $SCRIPTS_FOLDER"

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
python -m virtualenv "venv"
# shellcheck disable=SC1090
source "venv/bin/activate"

# export WHEELHOUSE="${HOME}/mila_tools_wheelhouse/"
# mkdir -p $WHEELHOUSE
# export PIP_FIND_LINKS="file://${WHEELHOUSE}"
# export PIP_WHEEL_DIR="${WHEELHOUSE}"

python -m pip install --upgrade pip

log "Downloading modules"
sh $HOME/install_jax.sh # TODO: move this to mila_tools
python -m pip install -r "requirements.txt" --exists-action w

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the mila_tools version to avoid issues
log "/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $2"
/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $2
