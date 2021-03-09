#! /bin/bash
set -e
# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}
SCRIPT=$(realpath $0)
log "script realpath: $SCRIPT"
SCRIPTS_FOLDER=$(dirname $SCRIPT)
log "scripts home: $SCRIPTS_FOLDER"

source /etc/profile
log "Refreshing modules..."
module purge
module load python/3.8
module load cuda/10.1/cudnn/7.6

log "cd $HOME/experiments/"
mkdir -p $HOME/experiments/
cd $HOME/experiments/

EXPERIMENT_FOLDER=$(mktemp -p . -d)
log "EXPERIMENT_FOLDER=$EXPERIMENT_FOLDER"

log "downloading source code from $1 to $EXPERIMENT_FOLDER"
git clone $1 $EXPERIMENT_FOLDER/
cd $EXPERIMENT_FOLDER
git checkout $3
log "pwd is now $(pwd)"


if ![[ source $HOME/venv/bin/activate ]]
then
  log "Setting up venv @ $HOME/venv..."
  python3 -m virtualenv $HOME/"venv"
fi

log "Using shared venv @ $HOME/venv"

python3 -m pip install --upgrade pip

log "installing experiment_buddy"
pip3 install -e git+https://github.com/ministry-of-silly-code/experiment_buddy#egg=experiment_buddy

python3 -m pip install -r "requirements.txt" --exists-action w

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/

# TODO: the client should send the mila_tools version to avoid issues
log "/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $2"
/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $2
