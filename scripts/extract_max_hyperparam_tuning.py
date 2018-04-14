import numpy as np
import csv

filename='data/hyperparam_tuning_results_mnist'
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    max_accuracy = 0
    stepsize = None
    numdirections = None
    numtopdirections = None
    perturbationlength = None
    for row in reader:
        if float(row[4]) > max_accuracy:
            max_accuracy = float(row[4])
            stepsize = float(row[0])
            numdirections = int(row[1])
            numtopdirections = int(row[2])
            perturbationlength = float(row[3])

print('Stepsize:', stepsize, 'Num Directions:', numdirections, 'Num Top Directions:', numtopdirections, 'Perturbation Length:', perturbationlength, 'Accuracy:', max_accuracy)
