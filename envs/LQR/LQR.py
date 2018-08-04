import numpy as np
import gym
import gym.spaces.box as gym_box
from gym import utils
import scipy.linalg as LA
from IPython import embed
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
        #P = Q + K.T.dot(R).dot(K) + (A + B.dot(K)).T.dot(P).dot(A+B.dot(K))
        x_mean = A.dot(x_mean) + B.dot(u_mean)
        x_cov = A.dot(x_cov).dot(A.T) + B.dot(u_cov).dot(B.T)
    return total_c #x_0.dot(P).dot(x_0) + np.trace(Cov_0.dot(P))

class LQREnv(gym.Env):

    def __init__(self, x_dim, u_dim=1, rank=5, seed=100, T=10):
        super(LQREnv, self).__init__()
        np.random.seed(seed)
        
        self.A = np.zeros((x_dim, x_dim))
        tmpA = np.random.randn(x_dim, x_dim)
        A = tmpA.T.dot(tmpA)
        s, U = np.linalg.eig(A)
        s = s / np.linalg.norm(s) * .9
        s[int(x_dim/2.):] *= 0.1
        self.A = U.dot(np.diag(s)).dot(U.T)

        self.B = np.ones((x_dim, u_dim))
        self.Q = np.eye(x_dim)
        self.R = np.eye(u_dim) * 0.01

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
        # @avemula: Removing the noise in dynamics. Completely deterministic dynamics
        # self.state = next_state + np.random.multivariate_normal(mean = np.zeros(self.x_dim), cov = self.noise_cov)

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
        
        #cost_finite_K(self.A, self.B, self.Q, self.R, 
        #    self.init_state_mean, self.T, K)
        return cost_for_K    
