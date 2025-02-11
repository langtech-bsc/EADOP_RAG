#!/bin/bash

# PARAMETERS
DELIM="##############################"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_BASH="$SCRIPT_DIR/eadop_rag_evaluation.config"
CONFIG_YAML="config/test.yaml"

# LOAD CONFIG
. $CONFIG_BASH
echo -e "$DELIM\nLoading config in file $CONFIG_BASH\n" && cat $CONFIG_BASH && echo -e "\n$DELIM\n"

# ACTIVATE CONDA ENVIRONMENT
eval "$(conda shell.bash hook)"
cmd="conda activate $CONDA_ENV"
echo -e "\n$cmd\n" && eval $cmd

# RUN SCRIPT
cd $SCRIPT_DIR
cmd="python -W ignore evaluate_retrieval.py -c $CONFIG_YAML"
echo -e "\n$cmd\n" && eval $cmd