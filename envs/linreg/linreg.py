import gym
import numpy as np
import ipdb


class LinReg(gym.Env):
    def __init__(self, input_dim, batch_size, test_batch_size):
        super(LinReg, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.w = np.random.randn(self.input_dim+1) / np.sqrt(self.input_dim+1)
        self.num_accesses = 0

        cov = np.random.randn(self.input_dim+1, self.input_dim+1)
        self.cov = cov.T.dot(cov)
        s,U = np.linalg.eig(self.cov)
        s[0:int(self.input_dim/2.)] *= 2.
        s[int(self.input_dim/2.):] /= 2.
        self.cov = U.dot(np.diag(s)).dot(U.T)

        self.reset()

    def step(self, yhat, test=False):
        if not test:            
            self.num_accesses += yhat.shape[0] # self.batch_size
        # Compute squared loss
        loss = ((yhat - self.y)**2) * (1./2)
        grad = (1./self.batch_size) * self.x.T.dot(yhat - self.y)
        # Obs returned is none, reward is -loss, done is always True and info is None
        return None, -loss, True, {'grad': grad}

    def reset(self, test=False):
        if not test:            
            #self.x = np.random.randn(self.batch_size, self.input_dim+1)
            self.x = np.random.multivariate_normal(np.zeros(self.input_dim+1), self.cov, self.batch_size)
            self.x /= np.sqrt(self.input_dim)
        else:
            #self.x = np.random.randn(self.test_batch_size, self.input_dim+1)
            self.x = np.random.multivariate_normal(np.zeros(self.input_dim+1), self.cov, self.batch_size)
            self.x /= np.sqrt(self.input_dim)
            
        # Bias term should be 1
        self.x[:, 0] = 1
        self.y = self.x.dot(self.w) + np.random.randn()*0.001
        return self.x, self.y

    def get_num_accesses(self):
        return self.num_accesses
