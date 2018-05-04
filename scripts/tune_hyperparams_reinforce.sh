#!/bin/bash
lrs=(0.1 0.08 0.05 0.03 0.01 0.007 0.005 0.003 0.001)
seeds=(283 457 623)

for l in ${lrs[@]}
do
    for sd in ${seeds[@]}
    do
        python -m reinforce.lin_natural_reinforce --exp --lr=$l --seed=$sd --input_dim=1000 --n_accesses=100000
    done
done
