#!/usr/bin/env bash
export MHCNN_HOME=$(pwd)
export LOG_DIR="$MHCNN_HOME/logs"
export PYTHONPATH="$MHCNN_HOME:$PYTHONPATH"
export DATAPATH="$MHCNN_HOME/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
# source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv
