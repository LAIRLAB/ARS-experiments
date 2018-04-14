##
# @file   LQG.py
# @author Yibo Lin
# @date   Apr 2018
#

import numpy as np
import gym 
import gym.spaces.box as gym_box
from gym import utils
import scipy.linalg as LA
#from gym.envs.registration import register


def lqr_gain(A,B,Q,R):
  '''
  Arguments:
    State transition matrices (A,B)
    LQR Costs (Q,R)
  Outputs:
    K: optimal infinite-horizon LQR gain matrix given
  '''

  # solve DARE:
  M=LA.solve_discrete_are(A,B,Q,R)

  # K=(B'MB + R)^(-1)*(B'MA)
  return np.dot(LA.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))

def cost_inf_K(A,B,Q,R,K):
  '''
    Arguments:
      State transition matrices (A,B)
      LQR Costs (Q,R)
      Control Gain K
    Outputs:
      cost: Infinite time horizon LQR cost of static gain K
  '''
  cl_map = A+B.dot(K)
  if np.amax(np.abs(LA.eigvals(cl_map)))<(1.0-1.0e-6):
    cost = np.trace(LA.solve_discrete_lyapunov(cl_map.T,Q+np.dot(K.T,R.dot(K))))
  else:
    cost = float("inf")

  return cost

def cost_finite_K(A_true,B_true,Q,R,x0,T,K):
  '''
    Arguments:
      True Model state transition matrices (A_true,B_true)
      LQR Costs (Q,R)
      Initial State x0
      Time Horizon T
      Static Control Gain K
    Outputs:
      cost: finite time horizon LQR cost when control is static gain K on
      system (A_true,B_true)
  '''

  d,p = B_true.shape

  cost = 0
  x = x0
  for k in range(T):
    u = np.dot(K,x)
    x = A_true.dot(x)+B_true.dot(u)
    cost = cost+np.dot(x.T,Q.dot(x))+np.dot(u.T,R.dot(u))

  return cost.flatten()




#class LinearQuadGausEnv(mujoco_env.MujocoEnv, utils.EzPickle):
class LQGEnv(gym.Env):

    def __init__(self, x_dim, u_dim = 1, rank = 5):
        #utils.EzPickle.__init__(self)
        #mujoco_env.MujocoEnv.__init__(self, 'LQG.xml', 2)

        np.random.seed(1337) #fix A and B
        self.A = np.zeros((x_dim,x_dim))
        for i in range(x_dim):
            self.A[i,i] = 1.01
            k = 0
            while k <= rank:   #intuition: one server's heat slightly increases 5 other serves' temp
                pos = np.random.randint(x_dim)
                if pos != i:
                    self.A[i, pos] = 0.01
                    k = k + 1 
                                   
        self.B = np.random.rand(x_dim, u_dim) #u_dim many fans to reduce the temp of all servers. 
        self.Q = 1e-3*np.eye(x_dim)
        self.R = np.eye(u_dim)

        self.optimal_K = lqr_gain(A = self.A, B = self.B, Q = self.Q, R = self.R)

        self.x_dim = x_dim
        self.a_dim = u_dim

        self.observation_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (x_dim, ))
        self.action_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (u_dim, ))

        self.init_state_mean = np.ones(self.x_dim)*2
        self.init_state_cov = np.eye(self.x_dim)*0.1

        self.state = None
        self.noise_cov = np.eye(self.x_dim)*0.001 

        self.T = 50
        
        self.reset()

    def reset(self):
        self.state = np.random.multivariate_normal(mean = self.init_state_mean, cov = self.init_state_cov)
        self.t = 0
        return self.state

    def step(self, a): 

        cost = self.state.dot(self.Q).dot(self.state) + a.dot(self.R).dot(a)
        next_state = self.A.dot(self.state.reshape((self.x_dim, 1))) + self.B.dot(a.reshape((self.a_dim, 1)));
        next_state = next_state.reshape(self.x_dim)
        self.state = next_state + np.random.multivariate_normal(mean = np.zeros(self.x_dim), cov = self.noise_cov)

        done = False
        self.t += 1
        if self.t >= self.T:
            done = True
        return self.state, -cost, done, None

    def seed(self, seed):
        np.random.seed(seed)

    def render(self, mode='human', close=False):
        pass 
