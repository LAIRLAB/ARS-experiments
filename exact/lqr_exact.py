import numpy as np
from utils.ars import *

class Trajectory(object):
    def __init__(self):
        self.xs = []
        self.acts = []
        self.rews = []

        self.c_rew = 0.0

def rollout_one_traj(env, K, stats = None, acts=None):
    '''
    input
        K: given linear policy a_dim x x_dim
        env: gym like environment
        stats: a running states estimating the mean and std of states.
    return:
        a trajectory containing (s,a,r)
    '''
    traj = Trajectory()
    x = env.reset()
    done = False
    t = 0
    while done is False:
        if acts is None:            
            if stats is None:
                a = K.dot(x)
            else:
                a = K.dot((x - stats.mean)/stats.std)  #normalize state
        else:
            a = acts[t, :]

        traj.xs.append(x)
        traj.acts.append(a)
        x, r, done, _ = env.step(a)
        traj.rews.append(r)
        t += 1
    traj.c_rew = np.sum(traj.rews) #total reward for this trajectory

    return traj

def evaluation(env, K, stats = None, num_trajs = 10):
    trajs_c_rew = []
    for i in xrange(num_trajs):
        traj = rollout_one_traj(env, K, stats)
        trajs_c_rew.append(traj.c_rew)
    
    return np.mean(trajs_c_rew)


def lqr_exact(env, stats, lr, explore_mag=0.1, num_top_directions=5,
              num_directions=10, num_total_steps=100,K0=None, verbose=True, use_one_direction=False):
    a_dim = env.a_dim
    x_dim = env.x_dim
    T = env.T
    if not use_one_direction:        
        batch_size = int(num_directions*2*T) + T  # 1 extra rollout because of initial predicted actions
    else:
        batch_size = int(num_directions*T) + T

    exact_dim = a_dim * T

    if K0 is None:
        K0 = np.zeros((a_dim, x_dim))
    K = K0

    if verbose:        
        print("optimal K's performance is {}".format(env.optimal_cost))
    test_perfs = []

    e = 0
    while True:
        cum_c = env.evaluate_policy(K)
        info = (e, e*batch_size, cum_c)
        if verbose:            
            print(info)
        test_perfs.append(info)

        # If the policy cost is within 5%
        if abs(cum_c - env.optimal_cost) / env.optimal_cost < 0.05:
            return e * batch_size

        directions = sample_directions(num_directions, exact_dim)
        if not use_one_direction:            
            returns = np.zeros((2, num_directions))
        else:
            returns = np.zeros((1, num_directions))

        traj = rollout_one_traj(env = env, K=K, stats=stats)
        acts = traj.acts
        orig_return = traj.c_rew
        xs = np.array(traj.xs)

        for d in range(num_directions):
            for posneg in range(2):
                if use_one_direction and posneg==1:
                    continue
                perturbations = (-2*posneg + 1) * explore_mag * directions[d]
                perturbations = perturbations.reshape(T, a_dim)
                perturbed_acts = np.array(acts) + perturbations
                traj = rollout_one_traj(env = env, K=K, stats=stats, acts=perturbed_acts)
                returns[posneg, d] = traj.c_rew

                if stats is not None:
                    stats.push_batch(np.array(traj.xs))

        top_directions, top_returns = get_top_directions_returns(returns.T, directions, num_top_directions)
        std_top_returns = np.std(top_returns)
        if std_top_returns == 0:
            std_top_returns = 1.
        if not use_one_direction:
            diff_returns = top_returns[:, 0] - top_returns[:, 1]
        else:
            diff_returns = top_returns[:, 0] - orig_return
        action_gradient = np.dot(top_directions.T, (diff_returns)) / (top_directions.shape[0] * std_top_returns)
        action_gradient = action_gradient.reshape(T, a_dim)        
        param_update = lr * action_gradient.T.dot(xs)
        K = K + param_update

        if (e+1)*batch_size >= num_total_steps:
            return num_total_steps

        e += 1
