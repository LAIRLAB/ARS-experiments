import argparse
import numpy as np
import random
from envs.LQG.LQG import LQGEnv
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


def lqr_ars(env, stats, lr, explore_mag = 0.1, num_top_directions = 5, 
            num_directions = 10, num_total_steps = 100, K0 = None):
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

    print "running ars.."
    a_dim = env.a_dim
    x_dim = env.x_dim
    T = env.T #traj length
    batch_size = int(num_directions*2*T)

    if K0 is None:
        K0 = 0.0 * np.random.randn(a_dim, x_dim)
    K = K0

    print "[optimal K's performance is {}]".format(env.optimal_cost)
    test_perfs = []

    e = 1
    while True:
        cum_c = env.evaluate_policy(K) #analytically compute cost
        info = (e, e*batch_size, cum_c)
        print info
        test_perfs.append(info)

        # note in each epoch, we use 2*num_directions*T steps
        #hence batch_size is 2*num_directions*T
        #sample directions
        directions = sample_directions(num_directions, a_dim*x_dim)
        w = K.flatten()
        #given  w reshaped from the current K, generated perturbed policies. 
        perturbed_ws = perturb_parameters(w, directions, explore_mag) 
        returns = np.zeros((2, num_directions))

        for d in range(num_directions): #for each direction
            for posneg in range(2): # do twice: + and -
                Kp = perturbed_ws[posneg, d].reshape(a_dim,x_dim)
                #get return by genearting a traj using the pertubed policy Kp
                traj = rollout_one_traj(env = env, K = Kp, stats = stats)
                returns[posneg, d] = traj.c_rew
                #update running mean and std using the latest trajectory
                if stats is not None:
                    stats.push_batch(np.array(traj.xs)) 
        
        top_directions, top_returns = get_top_directions_returns(
                    returns.T, directions, num_top_directions)
        
        w = update_parameters(w, lr, top_returns, top_directions)
        K = w.reshape(a_dim, x_dim)

        #perform test: report cummualative cost:

        if e*batch_size >= num_total_steps: #break if hits the max number of steps.
            break
        e += 1

    return test_perfs
        
        
parser = argparse.ArgumentParser()
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=1)
# ARS parameters
parser.add_argument('--stepsize', type=float, default=0.02)
parser.add_argument('--num_directions', type=int, default=10)
parser.add_argument('--num_top_directions', type=int, default=5)
parser.add_argument('--perturbation_length', type=float, default=0.01)
parser.add_argument('--x_dim', type=int, default=200)
parser.add_argument('--a_dim', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=100) 

# Hyperparam tuning params
parser.add_argument('--exp', action='store_true')
parser.add_argument('--threshold', type=int, default=10000, help='Number of accesses after which performance is evaluated')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LQGEnv(x_dim = args.x_dim, u_dim = args.a_dim, rank = 5)
#stats = RunningStat(args.x_dim * args.a_dim)
stats = None
K0 = np.ones((args.a_dim, args.x_dim))*10 #np.random.randn(args.a_dim, args.x_dim)*0.01
test_perfs = lqr_ars(env, stats, args.stepsize, args.perturbation_length, 
        num_top_directions=args.num_top_directions, 
        num_directions=args.num_directions, 
        num_total_steps = args.n_accesses, K0 = None)