import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LinReg(args.input_dim, args.batch_size, args.test_batch_size)
optim = Adam(args.input_dim+1, args.lr)

w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

g = open('data/linear-newton-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')
lr = args.lr
while True:    
# for t in range(args.tsteps):
    # Training
    x, y = env.reset()
    yhat = x.dot(w)
    _, reward, _, info = env.step(yhat)
    loss = -np.mean(reward)
    grad = info['grad']
    hessian = 0
    for i in range(args.batch_size):
        xi = x[i].reshape(-1, 1)
        hessian += xi.dot(xi.T)
    hessian *= (1./args.batch_size)
    hessian += 1e-8 * np.eye(args.input_dim+1)
    hessian_inv = np.linalg.inv(hessian)
    # w = optim.update(w, hessian_inv.dot(grad))
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
