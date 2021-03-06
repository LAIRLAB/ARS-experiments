'''
Stochastic Gradient Descent for MNIST
Author: Anirudh Vemula

Heavily borrowed from https://github.com/pytorch/examples/blob/master/mnist/main.py
'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from envs.mnist.mnist import MNIST
import numpy as np
import random
import ipdb

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# Experiment parameters
parser.add_argument('--n_accesses', type=int, default=1000000)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--tsteps', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
# Debug parameters
parser.add_argument('--verbose', action='store_true')

# Parse arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Initialize environment
env = MNIST(args.batch_size, args.test_batch_size)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize model
model = Net()
if args.cuda:
    model.cuda()

# Initialize SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Log file
g = open('data/mnist-sgd-'+str(args.seed)+'.csv', 'w')

# Start
while True:    
    # Training
    model.train()
    # Get data
    x, y = env.reset()
    x = x.view(args.batch_size, 1, 28, 28)
    if args.cuda:
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)
    optimizer.zero_grad()
    # Get log softmax output
    output = model(x)
    # Increment accesses (used this function, since there is no reason for step)
    env.increment_accesses()
    # Compute negative log likelihood loss
    loss = F.nll_loss(output, y)
    # Compute gradients
    loss.backward()
    # SGD update
    optimizer.step()


    # Test
    model.eval()
    x, y = env.reset(test=True)
    x = x.view(args.test_batch_size, 1, 28, 28)
    if args.cuda:
        x, y = x.cuda(), y.cuda()
    x, y = Variable(x, volatile=True), Variable(y, volatile=True)
    output = model(x)
    # Compute predictions
    pred = output.data.max(1, keepdim=True)[1]
    # Compute accuracy
    correct = pred.eq(y.data.view_as(pred)).long().sum()
    accuracy = correct / args.test_batch_size
    if args.verbose:        
        print(env.get_num_accesses(), accuracy)
    g.write(str(env.get_num_accesses())+','+str(accuracy)+'\n')

    # Check termination
    if env.get_num_accesses() >= args.n_accesses:
        break
