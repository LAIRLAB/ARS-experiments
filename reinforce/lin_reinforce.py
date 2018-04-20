'''
REINFORCE for Linear Regression
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
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
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
# Initialize Adam optimizer
optim = Adam(args.input_dim+1, args.lr)

# Initialize parameters
w = 5*np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

# Log file
g = open('data/linear-reinforce-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')

# Start
while True:    
    # Training
    # Get data
    x, y = env.reset()
    # Compute predictions
    pred = x.dot(w)
    # Define normal distribution
    mu_yhat = pred
    std_yhat = 0.5
    # Sample from the normal distribution
    yhat = np.random.normal(mu_yhat, std_yhat)
    # Compute rewards
    _, reward, _, _ = env.step(yhat)
    mse_loss = -np.mean(reward)
    # Compute REINFORCE gradient
    grad = -(1./args.batch_size) * (x.T.dot(reward*(yhat - mu_yhat)))
    # adam update
    w = optim.update(w, grad)

    # Test
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    mse_loss = -np.mean(reward)

    if args.verbose:
        print(env.get_num_accesses(), mse_loss)
        
    g.write(str(env.get_num_accesses())+','+str(mse_loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
