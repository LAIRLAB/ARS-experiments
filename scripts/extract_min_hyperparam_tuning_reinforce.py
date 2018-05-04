import numpy as np
import csv

seeds = [283, 457, 623]
losses = [[] for _ in range(len(seeds))]
lr_arr = []
for s in range(len(seeds)):
    filename='data/hyperparam_tuning_results_reinforce_'+str(seeds[s])
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            losses[s].append(float(row[1]))
            if s == 0:
                lr_arr.append(float(row[0]))

losses = np.array(losses)
losses = np.mean(losses, axis=0)
ind = np.argmin(losses)
lr = lr_arr[ind]
min_loss = losses[ind]

print('Learning Rate:', lr, 'Loss:', min_loss)
