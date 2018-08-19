import numpy as np
from envs.LQR.LQR import LQREnv
from envs.linreg.linreg import LinReg
from reinforce.lqr_reinforce import *
import random
import pickle
import argparse
from progress.bar import Bar

parser = argparse.ArgumentParser()
parser.add_argument('--H_start', type=int, default=10, help="Horizon length to start with")
parser.add_argument('--H_end', type=int, default=200, help="Horizon length to end with")
parser.add_argument('--H_bin', type=int, default=20, help="Horizon length spacing at which experiments are done (or bin size)")
args = parser.parse_args()

filename = "tune_lqr_reinforce_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
_, data = pickle.load(open(filename, 'rb'))
lr, explore_mag = data

initial_seed=1000
np.random.seed(initial_seed)
num_random_seeds = 10
test_param_seed = np.random.randint(low = 1, high = 1e8, size = num_random_seeds)
x_dim = 100
a_dim = 1

Hs = list(range(args.H_start, args.H_end + args.H_bin, args.H_bin))
# FIXME: Reassigning lr and explore mag (not using tuning results)
lr, explore_mag = [0.001 for _ in range(len(Hs))], [0.1 for _ in range(len(Hs))]

K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
bar = Bar('Processing', max = len(Hs) * num_random_seeds)
for H_id, H in enumerate(Hs):
    #print("at H = {0}".format(H))    
    test_perf_seeds = []
    for seed in test_param_seed:
        #print ("at seed {0}".format(seed))
        bar.next()
        np.random.seed(seed)
        random.seed(seed)
        optimizer = Adam(x_dim*a_dim+1, lr[H_id])
        env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        batch_size = 100 # 10*H
        max_iter = 1e5 * H
        steps = policy_gradient_adam_linear_policy(env,explore_mag=explore_mag[H_id],
                                            optimizer = optimizer, batch_size=batch_size,
                                            max_iter = max_iter,
                                                   K0=K0, Natural=False, kl=0.005, stats=None, verbose=False)
        test_perf_seeds.append(steps)

    test_perf_cross_H.append(test_perf_seeds)
bar.finish()
filename = "lqr_reinforce_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
pickle.dump(test_perf_cross_H, open(filename, "wb"))


#H = 10: lr = 0.01, mag = 0.1, batch_size = 128
#H = 20: lr = 0.01, mag = 1.
