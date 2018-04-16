#!/bin/bash
stepsizes=(0.01 0.02 0.03)
numdirections=(50 100 200)
pertubationlengths=(0.01 0.02 0.03)
numtopdirections=(20 50 100)
seeds=(283 457 623)

for s in ${stepsizes[@]}
do
    for nd in ${numdirections[@]}
    do
        for pl in ${pertubationlengths[@]}
        do
            for ntd in ${numtopdirections[@]}
            do
                if [ $ntd -gt $nd ]
                then
                    continue
                else
		    for sd in ${seeds[@]}
		    do
			python -m ars.mnist_ars --exp --stepsize=$s --num_directions=$nd --perturbation_length=$pl --num_top_directions=$ntd --seed=$sd &
		    done      
		    wait
                fi
            done
        done
    done
done
