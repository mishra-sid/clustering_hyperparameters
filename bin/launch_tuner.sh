#!/usr/bin/env bash

set -exu
threads=${1:-12}
num_cpus=${2:-60}
mem=${3:-60000}
num_datasets=$4
num_rounds=$5

total_exps=$((num_datasets * num_rounds - 1))

TIME=`(date +%Y-%m-%d-%H-%M-%S-%N)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

job_name="clustering_hyperparams-$TIME"
log_dir=logs/clustering_hyperparams/$TIME
log_base=$log_dir/log

mkdir -p $log_dir

sbatch -J $job_name \
            -e $log_base-%A_%a.err \
            -o $log_base-%A_%a.log \
            --partition longq \
            --time=3-0 \
            --cpus-per-task $num_cpus \
            --nodes=1 \
            --ntasks=1 \
            --mem=$mem \
            --array=0-$total_exps \
            bin/run_tuner.sh $threads $num_datasets "${@:6}"
