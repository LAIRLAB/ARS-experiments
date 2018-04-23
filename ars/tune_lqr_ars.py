import numpy as np
from lqr_ars import *
from envs.LQG.LQG import LQGEnv
import pickle


print "start tuning parameters for ars"
#stepsize = [0.005, 0.01, 0.02, 0.03]
stepsize = [0.01]
num_directions = [50,100,200]
num_top_directions = [20, 50, 100]
pertubation = [0.01, 0.02, 0.03, 0.04]
result_table = np.zeros((len(stepsize), len(num_directions), len(num_top_directions), len(pertubation)))


initial_seed = 100
np.random.seed(initial_seed)
tune_param_seed = np.random.randint(low = 1, high = 1e8,size = 3)
x_dim = 500
a_dim = 1

K0 = np.ones((a_dim, x_dim))*0.01
for ss_id, ss in enumerate(stepsize):
    for num_dir_id, num_dir in enumerate(num_directions):
        for top_dir_id, top_dir in enumerate(num_top_directions):
            for per_id, per in enumerate(pertubation):
                print "at {} {} {} {}".format(ss, num_dir, top_dir, per)

                if num_dir < top_dir:
                    result_table[ss_id, num_dir_id, top_dir_id, per_id] = np.inf
                elif num_dir >= top_dir:
                    steps = []
                    for seed in tune_param_seed:
                        print "at seed {}".format(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        env = LQGEnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed) 
                        test_steps = lqr_ars(env, None, ss, per, top_dir, num_dir, 1e6, K0 = K0)
                        steps.append(test_steps)

                    avg_steps = np.mean(steps)
                    result_table[ss_id, num_dir_id, top_dir_id, per_id] = avg_steps

print result_table
min_index = np.where(result_table == np.min(result_table))
print min_index
pickle.dump(result_table, open("{}_tune_ars_results.p".format(stepsize[0]), 'wb'))



