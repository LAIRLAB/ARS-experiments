import numpy as np
from exact.lqr_exact import *
from envs.LQR.LQR import LQREnv
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--H_start', type=int, default=10, help="Horizon length to start with")
parser.add_argument('--H_end', type=int, default=200, help="Horizon length to end with")
parser.add_argument('--H_bin', type=int, default=20, help="Horizon length spacing at which experiments are done (or bin size)")
args = parser.parse_args()

print ("start running exact experiment")

ss = 0.001
num_dir = 10
top_dir = 10
per = 0.005

initial_seed=1000
np.random.seed(initial_seed)
test_param_seed = np.random.randint(low = 1, high = 1e8, size = 10)
x_dim = 100 #500
a_dim = 1

Hs = list(range(args.H_start, args.H_end + args.H_bin, args.H_bin))

K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
for H in Hs:
    test_perf_seeds = []
    for seed in test_param_seed:
        print("at seed {}".format(seed))
        np.random.seed(seed)
        env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        test_perf = lqr_exact(env, None, ss, per, top_dir, num_dir, 1e5 * H, K0 = K0)
        test_perf_seeds.append(test_perf)

    test_perf_cross_H.append(test_perf_seeds)


filename = "exact_result_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
pickle.dump(test_perf_cross_H, open(filename, "wb"))
