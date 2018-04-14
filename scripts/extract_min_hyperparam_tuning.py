import numpy as np
import csv

filename='data/hyperparam_tuning_results'
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    min_loss = np.inf
    stepsize = None
    numdirections = None
    numtopdirections = None
    perturbationlength = None
    for row in reader:
        if float(row[4]) < min_loss:
            min_loss = float(row[4])
            stepsize = float(row[0])
            numdirections = int(row[1])
            numtopdirections = int(row[2])
            perturbationlength = float(row[3])

print('Stepsize:', stepsize, 'Num Directions:', numdirections, 'Num Top Directions:', numtopdirections, 'Perturbation Length:', perturbationlength, 'Loss:', min_loss)
