#!/usr/bin/env bash

set -exu

threads=12
cpus=60
mem=60000

num_generic_datasets=56
num_nlp_datasets=9

num_generic_rounds=10
num_nlp_rounds=5

declare -a models=("kmeans-minibatch" "dbscan" "hac" "birch" "hdbscan")

# Spawn generic sbatch jobs
for model_name in ${models[@]}; do
    ./launch_tuner.sh $threads $cpus $mem $num_generic_datasets $num_generic_rounds model=$model_name 
done

# Spawn nlp sbatch jobs
for model_name in ${models[@]}; do
    ./launch_tuner.sh $threads $cpus $mem $num_nlp_datasets $num_nlp_rounds model=$model_name 
done
