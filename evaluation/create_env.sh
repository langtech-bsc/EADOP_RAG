#!/bin/bash

# LOAD CONFIG
DELIM="##############################"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG="$SCRIPT_DIR/eadop_rag_evaluation.config"
. $CONFIG
echo -e "$DELIM\nLoading config in file $CONFIG\n" && cat $CONFIG && echo -e "\n$DELIM\n"

# TO BE ABLE CONDA INSIDE THE SCRIPT
eval "$(conda shell.bash hook)"

# STEPS
conda activate base
conda remove -n $CONDA_ENV --all -y
cmd="conda create --name $CONDA_ENV python=$PY_VERSION -y"
echo -e "\n$cmd\n"
eval $cmd

# HELPER COMMANDS TO CONNECT TO THE ENV
conda env list
cmd="conda activate $CONDA_ENV"
echo $cmd && eval $cmd
REQ="$SCRIPT_DIR/requirements.txt"
pip install -r $REQ

