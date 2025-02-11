#!/bin/bash

# PARAMETERS
SCRIPT_DIR=$(dirname $(realpath $0))
CONFIG_FILE="config/test.yaml"
ENV_NAME="eadop_rag"

# ACTIVATE CONDA ENVIRONMENT
eval "$(conda shell.bash hook)"
cmd="conda activate $ENV_NAME"
echo $cmd && eval $cmd

# RUN SCRIPT
cd $SCRIPT_DIR
cmd="python evaluate_retrieval.py -c $CONFIG_FILE"
echo $cmd && eval $cmd