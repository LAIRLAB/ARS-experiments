#!/bin/bash
stepsizes=(0.001 0.005 0.01 0.02 0.03)
numdirections=(10 50 100 200 500)
# numdirections=(50)
pertubationlengths=(0.001 0.005 0.01 0.02 0.03)
numtopdirections=(5 10 50 100 200)
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
		python -m ars.lin_ars --exp --stepsize=$s --num_directions=$nd --perturbation_length=$pl --num_top_directions=$ntd --seed=$sd --input_dim=100 --threshold=100000 &
	    done
	    wait
        fi
      done
    done
  done
done
