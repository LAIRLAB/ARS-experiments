#!/bin/bash

seeds=(5488 7151 6027 5448 4236 6458 4375 8917 9636 3834)

for s in ${seeds[@]}
do
    if [ "$1" == "10" ];
    then        
        python -m reinforce.lin_reinforce --seed=$s --input_dim=$1 --n_accesses=300000 --lr=0.08
    elif [ "$1" == "100" ];
    then
        python -m reinforce.lin_reinforce --seed=$s --input_dim=$1 --n_accesses=300000 --lr=0.03
    elif [ "$1" == "1000" ];
    then
        python -m reinforce.lin_reinforce --seed=$s --input_dim=$1 --n_accesses=300000 --lr=0.01
    else
        echo "Tested only for d=10,100,1000"
        break
    fi
done
