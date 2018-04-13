import gym
import numpy as np
import ipdb


class LinReg(gym.Env):
    def __init__(self, input_dim, batch_size):
        super(LinReg, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.w = np.random.randn(self.input_dim+1) / np.sqrt(self.input_dim+1)
        self.num_accesses = 0
        self.reset()

    def step(self, yhat, test=False):
        if not test:            
            self.num_accesses += self.batch_size
        # Compute squared loss
        loss = (yhat - self.y).T.dot(yhat - self.y) * (1./(2 * self.batch_size))
        grad = (1./self.batch_size) * self.x.T.dot(yhat - self.y)
        # Obs returned is none, reward is -loss, done is always True and info is None
        return None, -loss, True, {'grad': grad}

    def reset(self):
        self.x = np.random.randn(self.batch_size, self.input_dim+1)
        # Bias term should be 1
        self.x[:, 0] = 1
        self.y = self.x.dot(self.w) + np.random.randn()*0.01
        return self.x, self.y

    def get_num_accesses(self):
        return self.num_accesses
