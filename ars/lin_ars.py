'''
Augmented random search for Linear Regression
Author: Anirudh Vemula
'''
import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam
from utils.ars import *

parser = argparse.ArgumentParser()
# Experiment parameters
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
# ARS parameters
parser.add_argument('--stepsize', type=float, default=0.03)
parser.add_argument('--num_directions', type=int, default=50)
parser.add_argument('--num_top_directions', type=int, default=20)
parser.add_argument('--perturbation_length', type=float, default=0.03)
# Hyperparam tuning params
parser.add_argument('--exp', action='store_true')
parser.add_argument('--threshold', type=int, default=100000, help='Number of accesses after which performance is evaluated')
# Debug parameters
parser.add_argument('--verbose', action='store_true')

# Parse arguments
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)
random.seed(args.seed)

# Initialize environment
env = LinReg(args.input_dim, args.num_directions, args.test_batch_size)

# Initialize parameters
w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

# Initialize stats
stats = RunningStat(args.input_dim)

# Log file
if not args.exp:
    g = open('data/linear-ars-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')

# Start
while True:
    # Sample directions
    directions = sample_directions(args.num_directions, args.input_dim+1)
    # Perturb parameters
    perturbed_ws = perturb_parameters(w, directions, args.perturbation_length)
    # Returns
    returns = np.zeros((2, args.num_directions))
    # mean and std
    mean = stats.mean.copy()
    std = stats.std.copy()
    # Get data
    x_all, y_all = env.reset()
    for d in range(args.num_directions):
        # Get data
        x, y = x_all[d].copy(), y_all[d].copy()
        x, y = x.reshape(1, -1), y.reshape(1, -1)
        # Normalize input
        stats.push_batch(x[:, 1:])
        x_norm = x.copy()
        x_norm[:, 1:] = (x[:, 1:] - mean) / std
        # For both +ve and -ve direction
        for posneg in range(2):
            # Perturbed params
            wp = perturbed_ws[posneg, d]
            # Get predictions
            yhat = x_norm.dot(wp)
            # Compute reward
            _, reward, _, _ = env.step(yhat)
            returns[posneg, d] = reward[d]
    # Get top directions and top returns
    top_directions, top_returns = get_top_directions_returns(returns.T, directions, args.num_top_directions)
    # Update params
    w = update_parameters(w, args.stepsize, top_returns, top_directions)

    # Test
    x, y = env.reset(test=True)
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - stats.mean) / stats.std
    yhat = x_norm.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    mse_loss = -np.mean(reward)
    if not args.exp:
        if args.verbose:
            print(env.get_num_accesses(), mse_loss)
        g.write(str(env.get_num_accesses())+','+str(mse_loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break

    # Check number of accesses for hyperparam tuning
    if args.exp:
        if env.get_num_accesses() > args.threshold:
            break

# For hyperparam tuning
if args.exp:
    g = open('data/hyperparam_tuning_results_'+str(args.seed), 'a')
    x, y = env.reset(test=True)
    x_norm = x.copy()
    x_norm[:, 1:] = (x[:, 1:] - stats.mean) / stats.std
    yhat = x_norm.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    loss = -np.mean(reward)
    mse_loss = loss
    g.write(str(args.stepsize)+','+str(args.num_directions)+','+str(args.num_top_directions)+','+str(args.perturbation_length)+','+str(mse_loss)+'\n')
