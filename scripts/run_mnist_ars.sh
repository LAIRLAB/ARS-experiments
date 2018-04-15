#!/bin/bash

seeds=(5488 7151 6027 5448 4236 6458 4375 8917 9636 3834)

for s in ${seeds[@]}
do
    python -m ars.mnist_ars --seed=$s
done
