import argparse
import numpy as np
import random
import ipdb
from utils.adam import Adam
from utils.ars import *

class Trajectory(object):

    def __init__(self):
        self.xs = []
        self.acts = []
        self.rews = []

        self.c_rew = 0.0

def rollout_one_traj(env, K, stats = None):
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
    while done is False:
        if stats is None:
            a = K.dot(x)
        else:
            a = K.dot((x - stats.mean)/stats.std)  #normalize state 

        traj.xs.append(x)
        traj.acts.append(a)
        x, r, done, _ = env.step(a)
        traj.rews.append(r)
    traj.c_rew = np.sum(traj.rews) #total reward for this trajectory

    return traj

def evaluation(env, K, stats = None, num_trajs = 10):
    trajs_c_rew = []
    for i in xrange(num_trajs):
        traj = rollout_one_traj(env, K, stats)
        trajs_c_rew.append(traj.c_rew)
    
    return np.mean(trajs_c_rew)


def lqr_ars(env, stats, lr, explore_mag = 0.1, num_top_directions = 5, 
            num_directions = 10, num_total_steps = 100, K0 = None, verbose=True, use_one_direction=False):
    '''
    input:
        env: gym-like environment 
        stats: a running stats that computes state's mean and diagnoal std
        lr: learning rate
        explore_mag: exploration noise to scale the random direction
        num_top_directions: the # of top directions for updating policy 
        num_directions: total # of randomly sampled direction
        num_total_steps: max number of steps to interact with env
        K0: initail policy, if None, then K is initizlied to be zero.
    output:
        test_perfs: a list containing triples of 
        (iter id, total steps so far,current test cummulative cost)
    '''

    a_dim = env.a_dim
    x_dim = env.x_dim
    T = env.T #traj length
    if use_one_direction:
        batch_size = int(num_directions*T) + T
    else:        
        batch_size = int(num_directions*2*T)

    if K0 is None:
        K0 = 0.0 * np.random.randn(a_dim, x_dim)
    K = K0

    if verbose:        
        print ("[optimal K's performance is {}]".format(env.optimal_cost))
    test_perfs = []

    e = 0
    while True:
        cum_c = env.evaluate_policy(K)
        info = (e, e*batch_size, cum_c)
        if verbose:            
            print (info)
        test_perfs.append(info)

        # If the policy cost is within 5%
        if abs(cum_c - env.optimal_cost)/env.optimal_cost < 0.05:
            return e*batch_size

        # note in each epoch, we use 2*num_directions*T steps
        #hence batch_size is 2*num_directions*T
        #sample directions
        directions = sample_directions(num_directions, a_dim*x_dim)
        w = K.flatten()
        #given  w reshaped from the current K, generated perturbed policies. 
        perturbed_ws = perturb_parameters(w, directions, explore_mag)
        if use_one_direction:
            returns = np.zeros((1, num_directions))
        else:            
            returns = np.zeros((2, num_directions))

        orig_return=None
        if use_one_direction:
            orig_return = rollout_one_traj(env=env, K=K, stats=stats).c_rew

        for d in range(num_directions): #for each direction
            for posneg in range(2): # do twice: + and -
                if use_one_direction and posneg == 1:
                    continue
                Kp = perturbed_ws[posneg, d].reshape(a_dim,x_dim)
                #get return by genearting a traj using the pertubed policy Kp
                traj = rollout_one_traj(env = env, K = Kp, stats = stats)
                returns[posneg, d] = traj.c_rew
                #update running mean and std using the latest trajectory
                if stats is not None:
                    stats.push_batch(np.array(traj.xs))

        top_directions, top_returns = get_top_directions_returns(
                    returns.T, directions, num_top_directions)

        w = update_parameters(w, lr, top_returns, top_directions, use_one_direction=use_one_direction, orig_return=orig_return)
        K = w.reshape(a_dim, x_dim)

        #perform test: report cummualative cost:

        if (e+1)*batch_size >= num_total_steps: #break if hits the max number of steps.
            return num_total_steps
            break
        e += 1
