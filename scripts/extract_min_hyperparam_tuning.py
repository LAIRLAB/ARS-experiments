import numpy as np
import csv

seeds = [283, 457, 623]
losses = np.zeros(525)
stepsizes_arr = np.zeros(525)
numdirections_arr = np.zeros(525)
numtopdirections_arr = np.zeros(525)
pertlength_arr = np.zeros(525)
for s in seeds:
    filename='data/hyperparam_tuning_results_'+str(s)
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        e = 0
        for row in reader:
            losses[e] += float(row[4])
            stepsizes_arr[e] = float(row[0])
            numdirections_arr[e] = float(row[1])
            numtopdirections_arr[e] = float(row[2])
            pertlength_arr[e] = float(row[3])
            e += 1
        
losses = losses / 3.

ind = np.argmin(losses)
stepsize = stepsizes_arr[ind]
numdirections = numdirections_arr[ind]
numtopdirections = numtopdirections_arr[ind]
perturbationlength = pertlength_arr[ind]
min_loss = losses[ind]

print('Stepsize:', stepsize, 'Num Directions:', numdirections, 'Num Top Directions:', numtopdirections, 'Perturbation Length:', perturbationlength, 'Loss:', min_loss)
