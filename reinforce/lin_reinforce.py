import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LinReg(args.input_dim, args.batch_size, args.test_batch_size)
optim = Adam(args.input_dim+1, args.lr)

w = 5*np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

g = open('data/linear-reinforce-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')
lr = args.lr
while True:    
# for t in range(args.tsteps):
    # Training
    x, y = env.reset()
    pred = x.dot(w)
    mu_yhat = pred
    std_yhat = 0.5 #0.01
    yhat = np.random.normal(mu_yhat, std_yhat)
    _, reward, _, _ = env.step(yhat)
    mse_loss = -np.mean(reward)
    grad = -(1./args.batch_size) * (x.T.dot(reward*(yhat - mu_yhat))) #* (1./(std_yhat**2))
    w = optim.update(w, grad)
    #w = w - lr*grad

    #lr = lr * 0.999

    # Test
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    mse_loss = -np.mean(reward)

    # print "mse_loss: {}".format(mse_loss)
    if args.verbose:
        print(env.get_num_accesses(), mse_loss)
        
    # print(env.get_num_accesses(), loss)
    g.write(str(env.get_num_accesses())+','+str(mse_loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
