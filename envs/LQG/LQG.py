import numpy as np
import gym 
import gym.spaces.box as gym_box
from gym import utils
import scipy.linalg as LA
from IPython import embed

def finite_LQR_solver(A,B,Q,R,T, x_0,Cov_0, noise_cov):
  x_dim = A.shape[0]
  a_dim = B.shape[1]
  P = np.zeros((x_dim, x_dim))    

  for i in range(T):
    P = (A.T.dot(P).dot(A) + Q 
         - A.T.dot(P).dot(B).dot(LA.inv(B.T.dot(P).dot(B)+R)).dot(B.T).dot(P).dot(A))
    
  return x_0.dot(P).dot(x_0) + np.trace(Cov_0.dot(P))

def finite_K_cost(A, B, Q, R, K, T, x_0, Cov_0):
  x_dim = A.shape[0]
  a_dim = B.shape[1]
  P = np.zeros((x_dim,x_dim))
  total_c = 0
  x_mean = x_0
  x_cov = Cov_0
  for i in range(T):
    u_mean = K.dot(x_mean)
    u_cov = K.dot(x_cov).dot(K.T)
    total_c += x_mean.dot(Q).dot(x_mean) + np.trace(x_cov.dot(Q)) + u_mean.dot(R).dot(u_mean) + np.trace(u_cov.dot(R))
    x_mean = A.dot(x_mean) + B.dot(u_mean)
    x_cov = A.dot(x_cov).dot(A.T) + B.dot(u_cov).dot(B.T)
  return total_c


class LQGEnv(gym.Env):

    def __init__(self, x_dim, u_dim = 1, rank = 5, seed = 100, T = 10):
      super(LQGEnv, self).__init__()
      np.random.seed(seed) #fix A and B
      tmpA = np.random.randn(x_dim,x_dim)
      A = tmpA.T.dot(tmpA)
      s,U = np.linalg.eig(A)
      s = s / np.linalg.norm(s)*.9
      #embed()
      s[int(x_dim/2.):] *= 0.1
      self.A = U.dot(np.diag(s)).dot(U.T)
      
      self.B = np.ones((x_dim, u_dim))
      self.Q = np.eye(x_dim) #/np.sqrt(x_dim)
      self.R = np.eye(u_dim)*0.01
      
      self.x_dim = x_dim
      self.a_dim = u_dim
      
      self.observation_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (x_dim, ))
      self.action_space = gym_box.Box(low = -np.inf, high = np.inf, shape = (u_dim, ))
      
      self.init_state_mean = 1*np.ones(self.x_dim)*1/np.sqrt(self.x_dim)
      self.init_state_cov = np.eye(self.x_dim)/self.x_dim
      
      self.state = None
      self.noise_cov = np.eye(self.x_dim)*0.01
      
      self.T = T
      
      self.optimal_cost = finite_LQR_solver(self.A,self.B, self.Q,self.R, self.T, 
                                            self.init_state_mean,self.init_state_cov, self.noise_cov)
      
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

    def evaluate_policy(self, K):
      cost_for_K = finite_K_cost(self.A,self.B,self.Q, self.R, K, 
                                 self.T, self.init_state_mean, 
                                 self.init_state_cov)
      return cost_for_K    


