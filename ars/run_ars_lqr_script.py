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
parser.add_argument('--use_one_direction', action="store_true")
args = parser.parse_args()

print ("start running ars experiment")

filename = filename = "tune_lqr_ars_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
_, data = pickle.load(open(filename, 'rb'))
ss, num_dir, top_dir, per = data

initial_seed=1000
np.random.seed(initial_seed)
num_random_seeds = 10
test_param_seed = np.random.randint(low = 1, high = 1e8, size = num_random_seeds)
x_dim = 100 #500
a_dim = 1

Hs = list(range(args.H_start, args.H_end + args.H_bin, args.H_bin))

K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
bar = Bar('Processing', max = len(Hs)*num_random_seeds)
for H_id, H in enumerate(Hs):
    test_perf_seeds = []
    for seed in test_param_seed:
        bar.next()
        #print("at seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)
        env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        test_perf = lqr_ars(env, None, ss[H_id], per[H_id], top_dir[H_id], num_dir[H_id], 1e5*H, K0 = K0, verbose=False, use_one_direction=args.use_one_direction)
        test_perf_seeds.append(test_perf)

    test_perf_cross_H.append(test_perf_seeds)
bar.finish()
filename = "ars_result_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
pickle.dump(test_perf_cross_H, open(filename, "wb"))
