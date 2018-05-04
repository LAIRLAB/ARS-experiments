#!/bin/bash

seeds=(5488 7151 6027 5448 4236 6458 4375 8917 9636 3834)

for s in ${seeds[@]}
do
    python -m ars.lin_ars --seed=$s --input_dim=$1 --n_accesses=300000 --stepsize=0.03 --num_directions=200 --num_top_directions=200 --perturbation_length=0.03
done
