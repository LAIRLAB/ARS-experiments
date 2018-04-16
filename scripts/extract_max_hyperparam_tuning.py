import numpy as np
import csv

seeds = [283, 457, 623]
accuracies = [[] for _ in range(len(seeds))]
stepsizes_arr = []
numdirections_arr = []
numtopdirections_arr=[]
perturbationlength_arr=[]
for s in range(len(seeds)):
    filename='data/hyperparam_tuning_results_mnist_'+str(seeds[s])
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            accuracies[s].append(float(row[4]))
            if s == 0:
                stepsizes_arr.append(float(row[0]))
                numdirections_arr.append(int(row[1]))
                numtopdirections_arr.append(int(row[2]))
                perturbationlength_arr.append(float(row[3]))

accuracies = np.array(accuracies)
accuracies = np.mean(accuracies, axis=0)
ind = np.argmax(accuracies)
stepsize = stepsizes_arr[ind]
numdirections = numdirections_arr[ind]
numtopdirections = numtopdirections_arr[ind]
perturbationlength = perturbationlength_arr[ind]
max_accuracy = accuracies[ind]

print('Stepsize:', stepsize, 'Num Directions:', numdirections, 'Num Top Directions:', numtopdirections, 'Perturbation Length:', perturbationlength, 'Accuracy:', max_accuracy)
