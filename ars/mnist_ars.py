from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from envs.mnist.mnist import MNIST
from utils.ars import *
import numpy as np
import random
import ipdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--tsteps', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
# ARS params
parser.add_argument('--stepsize', type=float, default=0.005, help='Stepsize for ARS')
parser.add_argument('--num_directions', type=int, default=10, help='Number of directions sampled for ARS')
parser.add_argument('--num_top_directions', type=int, default=5, help='Number of top direction used for ARS')
parser.add_argument('--perturbation_length', type=float, default=0.1, help='Perturbation length for ARS')

# Hyperparam tuning params
parser.add_argument('--exp', action='store_true')
parser.add_argument('--threshold', type=int, default=10000)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

env = MNIST(args.batch_size, args.test_batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = Net()

param_shapes = [x.data.numpy().shape for x in model.parameters()]
param_num_elements = [np.prod(x) for x in param_shapes]
num_params = np.sum(param_num_elements)

def get_parameters(m):
    if args.cuda:
        return [x.cpu().data.numpy() for x in m.parameters()]
    return [x.data.numpy() for x in m.parameters()]

def set_parameters(params):
    for ind_param, param in enumerate(model.parameters()):
        if args.cuda:
            param.data = torch.Tensor(params[ind_param]).cuda()
        else:
            param.data = torch.Tensor(params[ind_param])

stats = RunningStat(shape=[28, 28])

if args.cuda:
    model.cuda()

while True:    
# for t in range(args.tsteps):
    # Get parameters of the model and flatten them
    params = get_parameters(model)
    params = flatten_params(params)
    # Sample directions
    directions = sample_directions(args.num_directions, num_params)
    # Perturb parameters
    perturbed_params = perturb_parameters(params, directions, args.perturbation_length)
    # Returns
    returns = np.zeros((2, args.num_directions))
    # mean and std
    mean, std = stats.mean, stats.std
    for d in range(args.num_directions):
        for posneg in range(2):
            # Reconstruct params
            reconstructed_params = reconstruct_params(perturbed_params[posneg, d, :], param_shapes, param_num_elements)
            # Set params
            set_parameters(reconstructed_params)
            # Get data
            x, y = env.reset()
            # Normalize input
            stats.push_batch(x)
            x = x.numpy().squeeze()
            x = torch.Tensor((x - mean) / std)
            x = x.view(args.batch_size, 1, 28, 28)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            output = model(x)
            disb = torch.distributions.Categorical(output)
            actions = disb.sample()
            if args.cuda:
                _, rewards, _, _ = env.step(actions.cpu().data)
            else:
                _, rewards, _, _ = env.step(actions.data)
            returns[posneg, d] = np.mean(rewards.numpy())
    # Get top directions and returns
    top_directions, top_returns = get_top_directions_returns(returns.T, directions, args.num_top_directions)
    # Update params
    params = update_parameters(params, args.stepsize, top_returns, top_directions)
    # Reconstruct params
    reconstructed_params = reconstruct_params(params, param_shapes, param_num_elements)
    # Set parameters
    set_parameters(reconstructed_params)

    # Test
    model.eval()
    x, y = env.reset(test=True)
    x = x.view(args.test_batch_size, 1, 28, 28)
    if args.cuda:
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x, volatile=True), Variable(y, volatile=True)
    output = model(x)
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(y.data.view_as(pred)).long().sum()
    accuracy = correct / args.test_batch_size
    if not args.exp:        
        print(env.get_num_accesses(), accuracy)

    # Check number of accesses for hyperparam tuning
    if args.exp:
        if env.get_num_accesses() > args.threshold:
            break
    else:
        if env.get_num_accesses() >= args.n_accesses:
            break

if args.exp:
    g = open('data/hyperparam_tuning_results_mnist', 'a')
    model.eval()
    x, y = env.reset(test=True)
    x = x.view(args.test_batch_size, 1, 28, 28)
    if args.cuda:
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x, volatile=True), Variable(y, volatile=True)
    output = model(x)
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(y.data.view_as(pred)).long().sum()
    accuracy = correct / args.test_batch_size

    g.write(str(args.stepsize)+','+str(args.num_directions)+','+str(args.num_top_directions)+','+str(args.perturbation_length)+','+str(accuracy)+'\n')
