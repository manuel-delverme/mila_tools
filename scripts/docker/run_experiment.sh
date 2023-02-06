#! /bin/bash

set -e

function log() {
  echo -e "\e[32m\"[DEPLOY LOG] $*\"\e[0m"
}

log "running on $(hostname)"
EXPERIMENT_PATH=$1
ENTRYPOINT=$2
# HASH_COMMIT=$3
# EXTRA_MODULES=$(echo $4 | tr "@" " ")

# log "Refreshing modules..."
# log "EXTRA_MODULES=$EXTRA_MODULES"
# for MODULE in $EXTRA_MODULES; do
#   log "module load $MODULE"
#   apt install -y $MODULE
# done

cd "$EXPERIMENT_PATH"
source "$EXPERIMENT_PATH/venv/bin/activate"

log "Upgrading pip"
python3 -m pip install --upgrade pip
log "Upgrading requirements"

python3 -m pip install --upgrade -r "requirements.txt" --exists-action s -f https://download.pytorch.org/whl/torch_stable.html -f https://storage.googleapis.com/jax-releases/jax_releases.html --use-deprecated=legacy-resolver
log "Requirements upgraded"

git config --global --add safe.directory "$EXPERIMENT_PATH"

log "$EXPERIMENT_PATH"/"$ENTRYPOINT"
python3 "$EXPERIMENT_PATH"/"$ENTRYPOINT"
