'''
Natural REINFORCE for Linear Regression
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
parser.add_argument('--lr', type=float, default=2.0)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=1000)
# Debug parameters
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--exp', action='store_true')

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
if not args.exp:    
    g = open('data/linear-naturalreinforce-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')

# learning rate
fixed_lr = args.lr

t = 0
# Start
while True:
    t += 1
    # Training
    # Get data
    x, y = env.reset()
    # Compute predictions
    pred = x.dot(w)
    # Define normal distribution
    mu_yhat = pred
    std_yhat = .5
    # Sample from normal distribution
    yhat = np.random.normal(mu_yhat, std_yhat)
    # compute rewards
    _, reward, _, _ = env.step(yhat)
    mse_loss = -np.mean(reward)
    # Compute gradient
    grad = -(1./args.batch_size) * (x.T.dot(reward*(yhat - mu_yhat)))
    # Compute fisher information matrix
    fim = x.T.dot(x) / args.batch_size
    # Add a small factor of identity matrix for inverse stability purposes
    fim += 1e-3 * np.eye(args.input_dim+1)
    # Compute descent direction by solving linear least squares problem
    descent_dir = np.linalg.lstsq(fim, grad, rcond=None)[0]
    fixed_lr = args.lr / np.sqrt(t)
    lr = np.sqrt(fixed_lr / descent_dir.dot(grad))
    # Update
    w = w - lr * descent_dir

    if not args.exp:        
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

if args.exp:
    g = open('data/hyperparam_tuning_results_reinforce_'+str(args.seed), 'a')
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    loss = -np.mean(reward)
    g.write(str(args.lr)+','+str(loss)+'\n')
