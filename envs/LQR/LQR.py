import numpy as np
import gym
import gym.spaces.box as gym_box
from gym import utils
import scipy.linalg as LA
from IPython import embed

def finite_LQR_solver(A,B,Q,R,T, x_0):
  x_dim = A.shape[0]
  a_dim = B.shape[1]
  P = np.zeros((x_dim, x_dim))    
  
  for i in range(T):
    P = (A.T.dot(P).dot(A) + Q 
         - A.T.dot(P).dot(B).dot(LA.inv(B.T.dot(P).dot(B)+R)).dot(B.T).dot(P).dot(A))
    
  return x_0.dot(P).dot(x_0)
  
def finite_K_cost(A, B, Q, R, K, T, x_0):
  x_dim = A.shape[0]
  a_dim = B.shape[1]
  P = np.zeros((x_dim,x_dim))
  total_c = 0
  x = x_0
  for i in range(T):
    u = K.dot(x)
    total_c += x.dot(Q).dot(x) + u.dot(R).dot(u)
    x = A.dot(x) + B.dot(u)
  return total_c

class LQREnv(gym.Env):

    def __init__(self, x_dim, u_dim=1, rank=5, seed=100, T=10):
      super(LQREnv, self).__init__()
      np.random.seed(seed)
      
      self.A = np.zeros((x_dim, x_dim))
      tmpA = np.random.randn(x_dim, x_dim)
      A = tmpA.T.dot(tmpA)
      s, U = np.linalg.eig(A)
      s = s / np.linalg.norm(s)
      self.A = U.dot(np.diag(s)).dot(U.T)
      
      self.B = np.ones((x_dim, u_dim))
      self.Q = np.eye(x_dim)
      self.R = np.eye(u_dim) * 1000  # FIX: Changed the cost to 1000
      
      self.x_dim = x_dim
      self.a_dim = u_dim
      
      self.observation_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (x_dim, ))
      self.action_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (u_dim, ))
      
      self.init_state = np.random.randn(x_dim)
      
      self.state = self.init_state.copy()
      self.noise_cov = np.eye(self.x_dim)*0.01
      
      self.T = T
      
      self.optimal_cost = finite_LQR_solver(self.A,self.B, self.Q,self.R, self.T, 
                                            self.init_state)
      
      self.reset()
      
    def reset(self):
      self.state = self.init_state.copy()
      self.t = 0
      return self.state

    def step(self, a): 
      
      cost = self.state.dot(self.Q).dot(self.state) + a.dot(self.R).dot(a)
      next_state = self.A.dot(self.state.reshape((self.x_dim, 1))) + self.B.dot(a.reshape((self.a_dim, 1)));
      next_state = next_state.reshape(self.x_dim)
      
      done = False
      self.t += 1
      if self.t >= self.T:
        done = True
      return self.state, -cost, done, None
      
    def seed(self, seed):
      np.random.seed(seed)

    def render(self, mode='human', close=False):
      pass 

    def evaluate_policy(self, K):
      cost_for_K = finite_K_cost(self.A,self.B,self.Q, self.R, K, 
                                 self.T, self.init_state)
      
      return cost_for_K    
