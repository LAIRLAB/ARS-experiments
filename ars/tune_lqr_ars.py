import numpy as np
from ars.lqr_ars import *
from envs.LQR.LQR import LQREnv
import pickle


print("start tuning parameters for ars")
stepsize = [0.001, 0.005, 0.01]
num_directions = [10, 20, 50]
num_top_directions = [5, 10, 20, 50]
pertubation = [0.001, 0.005, 0.01]
horizons = [10, 20, 40, 60, 80, 100, 120, 140, 160]
result_table = [np.zeros((len(stepsize), len(num_directions), len(num_top_directions), len(pertubation))) for _ in range(len(horizons))]


initial_seed = 100
np.random.seed(initial_seed)
tune_param_seed = np.random.randint(low = 1, high = 1e8,size = 5)
x_dim = 100
a_dim = 1

K0 = np.ones((a_dim, x_dim))*0.01
for h_id, h in enumerate(horizons):    
    for ss_id, ss in enumerate(stepsize):
        for num_dir_id, num_dir in enumerate(num_directions):
            for top_dir_id, top_dir in enumerate(num_top_directions):
                for per_id, per in enumerate(pertubation):
                    print("Horizon length {} at {} {} {} {}".format(h, ss, num_dir, top_dir, per))
                    
                    if num_dir < top_dir:
                        result_table[h_id][ss_id, num_dir_id, top_dir_id, per_id] = np.inf
                    elif num_dir >= top_dir:
                        steps = []
                        for seed in tune_param_seed:
                            print("at seed {}".format(seed))
                            np.random.seed(seed)
                            random.seed(seed)
                            env = LQREnv(x_dim = x_dim, u_dim = a_dim, rank = 5, seed=seed, T=h) 
                            test_steps = lqr_ars(env, None, ss, per, top_dir, num_dir, 1e6, K0 = K0)
                            steps.append(test_steps)
                            
                        avg_steps = np.mean(steps)
                        result_table[h_id][ss_id, num_dir_id, top_dir_id, per_id] = avg_steps

min_indices = [np.where(result_table[i] == np.min(result_table[i])) for i in range(len(horizons))]
print(min_indices)
pickle.dump(result_table, open("tune_ars_results.p", 'wb'))



