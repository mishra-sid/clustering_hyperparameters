#!/usr/bin/env bash

set -exu

threads=$1
num_datasets=$2

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

run_index=$(( SLURM_ARRAY_TASK_ID / num_datasets))
data_index=$(( SLURM_ARRAY_TASK_ID % num_datasets))

clustering_hyperparameters dataset_index=$data_index optim.run_index=$run_index "${@:3}"
