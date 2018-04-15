#!/bin/bash
stepsizes=(0.1 0.05 0.01 0.005 0.001)
numdirections=(10 50 100 200 500)
pertubationlengths=(0.1 0.05 0.01 0.005 0.001)
numtopdirections=(5 10 20 50 100 200 500)
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
			python -m ars.mnist_ars --tsteps=100000 --exp --stepsize=$s --num_directions=$nd --perturbation_length=$pl --num_top_directions=$ntd --seed=$sd &
		    done      
		    wait
                fi
            done
        done
    done
done
