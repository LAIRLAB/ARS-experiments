import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam
from utils.ars import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
# ARS parameters
parser.add_argument('--stepsize', type=float, default=0.02)
parser.add_argument('--num_directions', type=int, default=100)
parser.add_argument('--num_top_directions', type=int, default=100)
parser.add_argument('--perturbation_length', type=float, default=0.05)
parser.add_argument('--test_batch_size', type=int, default=1000)
# Hyperparam tuning params
parser.add_argument('--exp', action='store_true')
parser.add_argument('--threshold', type=int, default=100000, help='Number of accesses after which performance is evaluated')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LinReg(args.input_dim, args.batch_size, args.test_batch_size)
# optim = Adam(args.input_dim+1, args.lr)
stats = RunningStat(args.input_dim+1)

w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

# Log file
g = open('data/linear-ars-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')
# for t in range(args.tsteps):
while True:    
    # Sample directions
    directions = sample_directions(args.num_directions, args.input_dim+1)
    # Perturb parameters
    perturbed_ws = perturb_parameters(w, directions, args.perturbation_length)
    # Returns
    returns = np.zeros((2, args.num_directions))
    # mean and std
    mean = stats.mean
    std = stats.std
    for d in range(args.num_directions):
        for posneg in range(2):
            wp = perturbed_ws[posneg, d]
            # Get data
            x, y = env.reset()
            # Normalize input
            stats.push_batch(x)
            x_norm = (x - mean) / std
            # Get predictions
            yhat = x_norm.dot(wp)
            _, reward, _, _ = env.step(yhat)
            returns[posneg, d] = reward[0]
    # Get top directions and top returns
    top_directions, top_returns = get_top_directions_returns(returns.T, directions, args.num_top_directions)
    # Update params
    w = update_parameters(w, args.stepsize, top_returns, top_directions)

    # Test
    # for i in range(args.test_batch_size):
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    mse_loss = -np.mean(reward)
    if not args.exp:        
        # print(env.get_num_accesses(), mse_loss)
        g.write(str(env.get_num_accesses())+','+str(mse_loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break

    # Check number of accesses for hyperparam tuning
    if args.exp:
        if env.get_num_accesses() > args.threshold:
            break

if args.exp:
    g = open('data/hyperparam_tuning_results', 'a')
    mse_loss = 0.
    for i in range(args.test_batch_size):
        x, y = env.reset()
        yhat = x.dot(w)
        _, reward, _, _ = env.step(yhat, test=True)
        loss = -reward[0]
        mse_loss += loss
    g.write(str(args.stepsize)+','+str(args.num_directions)+','+str(args.num_top_directions)+','+str(args.perturbation_length)+','+str(mse_loss / args.test_batch_size)+'\n')
