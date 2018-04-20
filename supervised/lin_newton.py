'''
Newton Descent for Linear Regression
Author: Anirudh Vemula
'''
import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam

parser = argparse.ArgumentParser()
# Experiment parameters
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1000)
# Debug parameters
parser.add_argument('--verbose', action='store_true')

# Parse arguments
args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
random.seed(args.seed)

# Initialize environment
env = LinReg(args.input_dim, args.batch_size, args.test_batch_size)

# Initialize parameters
w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

# Log file
g = open('data/linear-newton-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')

# Start
while True:
    # Training
    # Get data
    x, y = env.reset()
    # Compute predictions
    yhat = x.dot(w)
    # Compute rewards
    _, reward, _, info = env.step(yhat)
    loss = -np.mean(reward)
    # Get exact gradient
    grad = info['grad']
    # Compute hessian
    hessian = 0
    for i in range(args.batch_size):
        xi = x[i].reshape(-1, 1)
        hessian += xi.dot(xi.T)
    hessian *= (1./args.batch_size)
    # Add a small factor of identity matrix for inverse stability
    hessian += 1e-8 * np.eye(args.input_dim+1)
    # Compute hessian inverse
    hessian_inv = np.linalg.inv(hessian)
    # Newton update
    w = w - hessian_inv.dot(grad)

    # Test
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    loss = -np.mean(reward)
    if args.verbose:        
        print(env.get_num_accesses(), loss)

    g.write(str(env.get_num_accesses())+','+str(loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
