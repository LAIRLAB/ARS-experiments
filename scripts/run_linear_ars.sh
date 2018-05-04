#!/bin/bash

seeds=(5488 7151 6027 5448 4236 6458 4375 8917 9636 3834)

for s in ${seeds[@]}
do
    if [ "$1" == "10" ];
    then
        python -m ars.lin_ars --seed=$s --input_dim=$1 --n_accesses=300000 --stepsize=0.03 --num_directions=10 --num_top_directions=10 --perturbation_length=0.03
    elif [ "$1" == "100" ];
    then
        python -m ars.lin_ars --seed=$s --input_dim=$1 --n_accesses=300000 --stepsize=0.03 --num_directions=10 --num_top_directions=10 --perturbation_length=0.02
    elif [ "$1" == "1000" ];
    then
        python -m ars.lin_ars --seed=$s --input_dim=$1 --n_accesses=300000 --stepsize=0.03 --num_directions=200 --num_top_directions=200 --perturbation_length=0.03
    else
        echo "Hyperparameters tuned only for d=10, 100, 1000"
        break
    fi
done
