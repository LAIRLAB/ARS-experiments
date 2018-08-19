import numpy as np
from ars.lqr_ars import *
from envs.LQR.LQR import LQREnv
import pickle
import argparse
from progress.bar import Bar

parser = argparse.ArgumentParser()
parser.add_argument('--H_start', type=int, default=10, help="Horizon length to start with")
parser.add_argument('--H_end', type=int, default=200, help="Horizon length to end with")
parser.add_argument('--H_bin', type=int, default=20, help="Horizon length spacing at which experiments are done (or bin size)")
args = parser.parse_args()

print("start tuning parameters for ars")
stepsize = [0.0005, 0.001, 0.005, 0.01, 0.05]
num_directions = [1, 10, 20, 50]
num_top_directions = [1, 5, 10, 20, 50]
pertubation = [0.0005, 0.001, 0.005, 0.01]
horizons = list(range(args.H_start, args.H_end + args.H_bin, args.H_bin))
result_table = [np.zeros((len(stepsize), len(num_directions), len(num_top_directions), len(pertubation))) for _ in range(len(horizons))]


initial_seed = 100
np.random.seed(initial_seed)
tune_param_seed = np.random.randint(low = 1, high = 1e8,size = 3)
x_dim = 100
a_dim = 1

K0 = np.ones((a_dim, x_dim))*0.01
bar = Bar('Processing', max=len(horizons)*len(stepsize)*len(num_directions)*len(num_top_directions)*len(pertubation))
for h_id, h in enumerate(horizons):    
    for ss_id, ss in enumerate(stepsize):
        for num_dir_id, num_dir in enumerate(num_directions):
            for top_dir_id, top_dir in enumerate(num_top_directions):
                for per_id, per in enumerate(pertubation):
                    #print("Horizon length {} at {} {} {} {}".format(h, ss, num_dir, top_dir, per))
                    bar.next()
                    if num_dir < top_dir:
                        result_table[h_id][ss_id, num_dir_id, top_dir_id, per_id] = np.inf
                    elif num_dir >= top_dir:
                        steps = []
                        for seed in tune_param_seed:
                            #print("at seed {}".format(seed))
                            np.random.seed(seed)
                            random.seed(seed)
                            env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T=h)
                            test_steps = lqr_ars(env, None, ss, per, top_dir, num_dir, 1e6, K0 = K0, verbose=False)
                            steps.append(test_steps)
                            
                        avg_steps = np.mean(steps)
                        result_table[h_id][ss_id, num_dir_id, top_dir_id, per_id] = avg_steps
bar.finish()
min_indices = np.array([np.unravel_index(np.argmin(result_table[i]), result_table[i].shape) for i in range(len(horizons))])
ss = np.array(stepsize)[min_indices[:, 0]]
nd = np.array(num_directions)[min_indices[:, 1]]
ntd = np.array(num_top_directions)[min_indices[:, 2]]
pt = np.array(pertubation)[min_indices[:, 3]]

filename = "tune_lqr_ars_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
data = (result_table, (ss, nd, ntd, pt))
pickle.dump(data, open(filename, 'wb'))
