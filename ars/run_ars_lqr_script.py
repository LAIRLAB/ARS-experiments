import numpy as np
from lqr_ars import *
from envs.LQR.LQR import LQREnv
import pickle

print ("start running ars experiment")
ss = 0.03 #0.02
num_dir = 20 #100
top_dir = 10 #50
per = 0.02

initial_seed=1000
np.random.seed(initial_seed)
test_param_seed = np.random.randint(low = 1, high = 1e8, size = 10)
x_dim = 100 #500
a_dim = 1

Hs = [10, 20, 40, 60, 80, 100, 120, 140,160]


K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
for H in Hs:
    test_perf_seeds = []
    for seed in test_param_seed:
        print("at seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)
        env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        test_perf = lqr_ars(env, None, ss, per, top_dir, num_dir, 1e5, K0 = K0)
        test_perf_seeds.append(test_perf)

    test_perf_cross_H.append(test_perf_seeds)

pickle.dump(test_perf_cross_H, open("ars_result_cross_H_10_160.p", "wb"))


