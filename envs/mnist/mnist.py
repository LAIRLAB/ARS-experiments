import gym
import torch
from torchvision import datasets, transforms
import numpy as np


class MNIST(gym.Env):
    def __init__(self, train_batch_size, test_batch_size):
        super(MNIST, self).__init__()
        self.train_data = datasets.MNIST('data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        self.test_data = datasets.MNIST('data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        self.num_accesses = 0
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.reset()

    def step(self, actions, test=False):
        if not test:
            self.num_accesses += actions.shape[0]

        rewards = actions == self.y
        rewards = rewards.type(torch.FloatTensor)
        rewards = 2*rewards - 1

        return None, rewards, None, None


    def reset(self, test=False):
        if test:
            test_ind = torch.LongTensor(np.random.randint(self.test_length, size=self.test_batch_size))            
            self.x = self.test_data.test_data.index_select(0, test_ind).type(torch.FloatTensor)
            self.y = self.test_data.test_labels.index_select(0, test_ind)# .type(torch.FloatTensor)

            return self.x, self.y
        else:
            train_ind = torch.LongTensor(np.random.randint(self.train_length, size=self.train_batch_size))
            self.x = self.train_data.train_data.index_select(0, train_ind).type(torch.FloatTensor)
            self.y = self.train_data.train_labels.index_select(0, train_ind)# .type(torch.FloatTensor)

            return self.x, self.y

    def get_num_accesses(self):
        return self.num_accesses

    def increment_accesses(self):
        self.num_accesses += self.train_batch_size
