#! /bin/bash

# Module system

function log() {
  echo -e "\e[31m" $1 "\e[0m"
}

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

# # Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
# log "Setting up venv @ $SLURM_TMPDIR/venv..."
# python -m virtualenv "$SLURM_TMPDIR/venv"
log "Using shared venv @ $HOME/venv"
source $HOME/venv/bin/activate
$HOME/venv/bin/python -m pip install --upgrade pip

# TODO: the client should send the mila_tools version to avoid issues
$HOME/venv/bin/python -m pip install --upgrade git+https://github.com/manuel-delverme/mila_tools/

log "Updating latest requirements"

# install cuda-jaxlib
PYTHON_VERSION=cp38           # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda101          # alternatives: cuda100, cuda101, cuda102, cuda110
PLATFORM=manylinux2010_x86_64 # alternatives: manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install $BASE_URL/$CUDA_VERSION/jaxlib-0.1.52-$PYTHON_VERSION-none-$PLATFORM.whl
pip install jax # install jax

pip install -r $HOME/requirements.txt --exists-action w

log "/opt/slurm/bin/sbatch $HOME/srun_python.sh $2"
/opt/slurm/bin/sbatch $HOME/srun_python.sh $2
