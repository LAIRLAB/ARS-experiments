import numpy as np
from envs.LQG.LQG import LQGEnv
from envs.linreg.linreg import LinReg
from lqr_reinforce import *
import random
import pickle


initial_seed=1000
np.random.seed(initial_seed)
test_param_seed = np.random.randint(low = 1, high = 1e8, size = 10)
x_dim = 100
a_dim = 1


Hs = [10, 20, 40, 60, 80, 100, 120, 140, 160]


lr = 0.005 #0.01
K0 = np.ones((a_dim, x_dim))*0.01
test_perf_cross_H = []
for H in Hs:
    print("at H = {0}".format(H))
    test_perf_seeds = []
    for seed in test_param_seed:
        print ("at seed {0}".format(seed))
        np.random.seed(seed)
        random.seed(seed)
        optimizer = Adam(x_dim*a_dim+1, lr)
        env = LQGEnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T = H)
        batch_size = 10*H
        steps = policy_gradient_adam_linear_policy(env,explore_mag=0.1,
                                            optimizer = optimizer, batch_size=batch_size,
                                            max_iter = 100,
                                            K0=K0, Natural=False, kl=0.005, stats=None)
        test_perf_seeds.append(steps)

    test_perf_cross_H.append(test_perf_seeds)

pickle.dump(test_perf_cross_H, open("lqr_reinforce_cross_H_10_160.p".format(H), "wb"))


#H = 10: lr = 0.01, mag = 0.1, batch_size = 128
#H = 20: lr = 0.01, mag = 1.
