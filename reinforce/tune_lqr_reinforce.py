import numpy as np
from reinforce.lqr_reinforce import *
from envs.LQR.LQR import LQREnv
import pickle
import random
import argparse
from progress.bar import Bar

parser = argparse.ArgumentParser()
parser.add_argument('--H_start', type=int, default=10, help="Horizon length to start with")
parser.add_argument('--H_end', type=int, default=200, help="Horizon length to end with")
parser.add_argument('--H_bin', type=int, default=20, help="Horizon length spacing at which experiments are done (or bin size)")
args = parser.parse_args()

print("start tuning parameters for reinforce")
lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
explore_mags = [0.005, 0.01, 0.05, 0.1]
horizons = list(range(args.H_start, args.H_end + args.H_bin, args.H_bin))
result_table = [np.zeros((len(lrs), len(explore_mags))) for _ in range(len(horizons))]


initial_seed = 100
np.random.seed(initial_seed)
tune_param_seed = np.random.randint(low = 1, high = 1e8,size = 3)
x_dim = 100
a_dim = 1

K0 = np.ones((a_dim, x_dim))*0.01
bar = Bar('Processing', max=len(horizons)*len(lrs)*len(explore_mags))
for h_id, h in enumerate(horizons):
    for lr_id, lr in enumerate(lrs):
        for exp_id, exp in enumerate(explore_mags):
            #print("Horizon length {} at {} {} {} {}".format(h, ss, num_dir, top_dir, per))
            bar.next()
            steps = []
            for seed in tune_param_seed:
                #print("at seed {}".format(seed))
                np.random.seed(seed)
                random.seed(seed)
                env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T=h) 
                optimizer = Adam(x_dim*a_dim+1, lr)
                test_steps = policy_gradient_adam_linear_policy(env, explore_mag=exp, optimizer=optimizer, batch_size=100, max_iter=1e6, K0=K0, Natural=False, kl=0.005, stats=None, verbose=False)
                steps.append(test_steps)

            avg_steps = np.mean(steps)
            result_table[h_id][lr_id, exp_id] = avg_steps
bar.finish()
min_indices = np.array([np.unravel_index(np.argmin(result_table[i]), result_table[i].shape) for i in range(len(horizons))])
lr = np.array(lrs)[min_indices[:, 0]]
explore_mag = np.array(explore_mags)[min_indices[:, 1]]

filename = "tune_lqr_reinforce_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
data = (result_table, (lr, explore_mag))
pickle.dump(data, open(filename, 'wb'))

