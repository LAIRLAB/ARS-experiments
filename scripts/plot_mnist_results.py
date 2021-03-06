import matplotlib.pyplot as plt
import csv
import numpy as np

# EDIT: Increase font size according to reviwer's comments
plt.rcParams.update({'font.size': 25})

seeds = [5488, 7151, 6027, 5448, 4236, 6458, 4375, 8917, 9636, 3834]
exps = ['ars', 'reinforce', 'sgd']
num_accesses = 1000000

n_accesses = {}
losses = {}
for e in exps:
    n_accesses[e] = {}
    losses[e] = {}
    
for e in exps:
    for s in seeds:
        filename = 'data/mnist-'+e+'-'+str(s)+'.csv'
        losses[e][s] = []
        n_accesses[e][s] = []
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                n_accesses[e][s].append(float(row[0]))
                losses[e][s].append(float(row[1]))

results = {}
for e in exps:
    num = len(n_accesses[e][5488])
    results[e] = np.zeros((len(seeds), num))

    for ind_seed in range(len(seeds)):
        results[e][ind_seed] = np.array(losses[e][seeds[ind_seed]])

meanresults = {}
stdresults = {}
maxresults = {}
minresults = {}
for e in exps:
    meanresults[e] = np.mean(results[e], axis=0)
    stdresults[e] = np.std(results[e], axis=0)
    minresults[e] = np.amin(results[e], axis=0)
    maxresults[e] = np.amax(results[e], axis=0)

# Plotting
colors = ['red', 'blue', 'green', 'cyan']
for i, e in enumerate(exps):
    plt.plot(np.array(n_accesses[e][5488])/100000, meanresults[e], color=colors[i], label=e)
    plt.fill_between(np.array(n_accesses[e][5488])/100000, np.maximum(minresults[e], meanresults[e]-stdresults[e]), np.minimum(maxresults[e], meanresults[e]+stdresults[e]), facecolor=colors[i], alpha=0.2)


plt.xlabel('Number of samples (multiples of $10^5$)')
plt.ylabel('Test accuracy')
plt.title('MNIST Experiment')
plt.legend(loc='lower right')
plt.gcf().set_size_inches([11.16, 8.26])
plt.show()
