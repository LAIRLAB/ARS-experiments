import numpy as np


class Adam(object):
    def __init__(self, param_dim, alpha):
        self.param_dim = param_dim
        self.alpha = alpha
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.gamma = 1 - 1e-8

        self.m_t = np.zeros(param_dim)
        self.v_t = np.zeros(param_dim)
        self.t = 0

    def update(self, params, grad):
        self.t += 1
        self.beta_1_t = self.beta_1 * (self.gamma ** (self.t - 1))
        self.m_t = self.beta_1_t * self.m_t + (1 - self.beta_1_t) * grad
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (grad**2)

        m_cap = self.m_t / (1 - (self.beta_1**self.t))
        v_cap = self.v_t / (1 - (self.beta_2**self.t))
        
        params = params - (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)
        return params
