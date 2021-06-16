#! /bin/bash

set -e
source /etc/profile

log "running on $(hostname)"

log "Refreshing modules..."
module purge
module load python/3.7

log "Sourcing common"
. $(dirname "$0")/common.sh

GIT_URL=$1
ENTRYPOINT=$2
HASH_COMMIT=$3

load_git_folder $GIT_URL $HASH_COMMIT

set_up_venv "venv"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/

# TODO: the client should send the mila_tools version to avoid issues
log "/opt/slurm/bin/sbatch $SCRIPTS_FOLDER/srun_python.sh $ENTRYPOINT"

/opt/slurm/bin/sbatch --comment "$(git log --format=%B -n 1)" $SCRIPTS_FOLDER/srun_python.sh $ENTRYPOINT
