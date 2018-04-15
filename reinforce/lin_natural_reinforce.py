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

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

env = LinReg(args.input_dim, args.batch_size, args.test_batch_size)
optim = Adam(args.input_dim+1, args.lr)

w = 5 * np.random.randn(args.input_dim+1) / np.sqrt(args.input_dim+1)

g = open('data/linear-naturalreinforce-'+str(args.seed)+'-'+str(args.input_dim)+'.csv', 'w')
t = 0
fixed_lr = args.lr
while True:    
    # Training
    t = t + 1
    x, y = env.reset()
    pred = x.dot(w)
    mu_yhat = pred
    std_yhat = .5  # 0.01
    yhat = np.random.normal(mu_yhat, std_yhat)
    _, reward, _, _ = env.step(yhat)
    mse_loss = -np.mean(reward)
    grad = -(1./args.batch_size) * (x.T.dot(reward*(yhat - mu_yhat))) # * (1./(std_yhat**2))
    fim = 0
    for i in range(args.batch_size):
        xi = x[i].reshape(-1, 1)
        fim += xi.dot(xi.T)
    fim *= (1./args.batch_size) # * (1./(std_yhat**2))
    fim += 1e-3 * np.eye(args.input_dim+1)
    descent_dir = np.linalg.lstsq(fim, grad, rcond=None)[0]
    lr = np.sqrt(fixed_lr / descent_dir.dot(grad))
    w = w - lr * descent_dir
    fixed_lr = fixed_lr *1. #0.99

    # Test
    x, y = env.reset(test=True)
    yhat = x.dot(w)
    _, reward, _, _ = env.step(yhat, test=True)
    loss = -np.mean(reward)
    print(t, env.get_num_accesses(), loss)

    g.write(str(env.get_num_accesses())+','+str(loss)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
