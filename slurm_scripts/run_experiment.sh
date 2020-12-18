#! /bin/bash
# Module system
function log() {
  echo -e "\e[32m"[DEPLOY LOG] $1"\e[0m"
}
SCRIPT=$(realpath $0)
log "script realpath: $SCRIPT"
SCRIPTS_FOLDER=$(dirname $SCRIPT)
log "scripts home: $SCRIPTS_FOLDER"

source /etc/profile

log "cd $HOME/experiments/"
mkdir -p $HOME/experiments/
cd $HOME/experiments/ || exit

FOLDER=$(mktemp -p . -d)
log "FOLDER=$FOLDER"

log "downloading source code from $1 to $FOLDER"
git clone $1 $FOLDER/
cd $FOLDER || exit
git checkout $3
log "pwd is now $(pwd)"

# # Set up virtualenv in $SLURM_TMPDIR. Will get blown up at job end.
log "Setting up venv @ $FOLDER/venv..."
python3 -m pip install virtualenv
python3 -m virtualenv "venv"
# python -m clonevirtualenv "$HOME/venv" "venv"

# # shellcheck disable=SC1090
# source "venv/bin/activate"

# log "Using shared venv @ $HOME/venv"
# shellcheck disable=SC1090
# source $HOME/venv/bin/activate

python3 -m pip install --upgrade pip
#python3 -m pip install -r "requirements.txt" --exists-action w

log "HELLO python3 $2"
byobu python3 -O -u $2
