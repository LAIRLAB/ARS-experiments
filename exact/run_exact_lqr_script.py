import numpy as np
from exact.lqr_exact import *
from envs.LQR.LQR import LQREnv
import pickle

print ("start running ars experiment")
# ss = {10: 0.001, 20: 0.001, 40: 0.005, 60: 0.001, 80: 0.001, 100: 0.001, 120: 0.001, 140: 0.001, 160: 0.001}
# num_dir = {10: 10, 20: 10, 40: 10, 60: 10, 80: 20, 100: 10, 120: 10, 140: 10, 160: 10}
# top_dir = {10: 5, 20: 5, 40: 10, 60: 5, 80: 10, 100: 10, 120: 5, 140: 5, 160: 5}
# per = {10: 0.01, 20: 0.01, 40: 0.005, 60: 0.001, 80: 0.005, 100: 0.005, 120: 0.001, 140: 0.001, 160: 0.001}
ss = 0.001
num_dir = 10
top_dir = 10
per = 0.005

initial_seed=1000
np.random.seed(initial_seed)
test_param_seed = np.random.randint(low = 1, high = 1e8, size = 10)
x_dim = 100 #500
a_dim = 1

Hs = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]


K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
for H in Hs:
    test_perf_seeds = []
    for seed in test_param_seed:
        print("at seed {}".format(seed))
        np.random.seed(seed)
        env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        test_perf = lqr_exact(env, None, ss, per, top_dir, num_dir, 1e6, K0 = K0)
        test_perf_seeds.append(test_perf)

    test_perf_cross_H.append(test_perf_seeds)

pickle.dump(test_perf_cross_H, open("exact_result_cross_H_10_160.p", "wb"))
