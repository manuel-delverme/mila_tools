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

# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}

source /etc/profile
log "Refreshing modules..."
module purge
GIT_URL=$1
ENTRYPOINT=$2
HASH_COMMIT=$3
EXTRA_MODULES=$(echo $4 | tr "@" " ")
COUNT=$5

for MODULE in $EXTRA_MODULES
do
	module load $MODULE
done

FOLDER=$SLURM_TMPDIR/src/

log "downloading source code from $GIT_URL to $FOLDER"
# https://stackoverflow.com/questions/7772190/passing-ssh-options-to-git-clone/28527476#28527476
GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone $GIT_URL $FOLDER/
cd $FOLDER || exit
git checkout $HASH_COMMIT
log "pwd is now $(pwd)"

# Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $SLURM_TMPDIR/venv..."

python3 -m venv --system-site-packages "$SLURM_TMPDIR/venv"
# shellcheck disable=SC1090
source "$SLURM_TMPDIR/venv/bin/activate"
python3 -m pip install --upgrade pip

log "Downloading modules"
python3 -m pip install --upgrade -r "requirements.txt" --exists-action w -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/
# TODO: the client should send the experiment_buddy version to avoid issues

if [ "$COUNT" = "None" ]; then
  wandb agent "$ENTRYPOINT"
else
  wandb agent --count $COUNT "$ENTRYPOINT"
fi
