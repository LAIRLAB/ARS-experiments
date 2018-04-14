import argparse
import numpy as np
import random
from envs.linreg.linreg import LinReg
import ipdb
from utils.adam import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--n_accesses', type=int, default=100000)
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LinReg(args.input_dim, args.batch_size)
optim = Adam(args.input_dim+1, args.lr)

w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

g = open('linear-sgd-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')
while True:    
# for t in range(args.tsteps):
    # Training
    x, y = env.reset()
    yhat = x.dot(w)
    _, reward, _, info = env.step(yhat)
    loss = -np.mean(reward)
    grad = info['grad']
    w = optim.update(w, grad)

    # Test
    x, y = env.reset()
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    loss = -np.mean(reward)
    # print(env.get_num_accesses(), loss)

    g.write(str(env.get_num_accesses())+','+str(loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
