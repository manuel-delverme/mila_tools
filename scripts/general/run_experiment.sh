#! /bin/bash

# Script to run a single experiment

set -e
# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}
SCRIPT=$(realpath $0)
log "script realpath: $SCRIPT"
SCRIPTS_FOLDER=$(dirname $SCRIPT)
log "scripts home: $SCRIPTS_FOLDER"

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

if ! source $HOME/buddy-venv/bin/activate; then
  log "Setting up venv @ $HOME/buddy-venv..."
  python3 -m venv $HOME/"buddy-venv"
  source $HOME/buddy-venv/bin/activate
fi

log "Using shared venv @ $HOME/buddy-venv"

python3 -m pip install --upgrade pip

log "installing experiment_buddy"
pip3 install -e git+https://github.com/ministry-of-silly-code/experiment_buddy#egg=experiment_buddy

python3 -m pip install --no-cache-dir -r "requirements.txt" --exists-action w

log "python3 $2"
export BUDDY_IS_DEPLOYED=1
screen -m -d bash -c "python3 -O -u $2"
