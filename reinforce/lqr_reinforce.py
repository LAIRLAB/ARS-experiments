import numpy as np 
import argparse
from envs.LQG.LQG import LQGEnv
from utils.adam import Adam
#from IPython import embed


class Trajectory(object):

    def __init__(self):
        self.xs = []
        self.acts = []
        self.rews = []

        self.c_rew = 0.0

def rollout_one_traj(env, K, explore_mag = 0.1, test = False):
    traj = Trajectory()
    x = env.reset()
    done = False
    while done is False:
        a = K.dot(x)
        if test is False:
            a += explore_mag*np.random.randn(env.a_dim)
        traj.xs.append(x)
        traj.acts.append(a)
        x, r, done, _ = env.step(a)
        traj.rews.append(r)
    
    traj.c_rew = np.sum(traj.rews)
    return traj

def roll_out(env, batch_size, K, explore_mag = 0.1, test = False):
    total_steps = 0
    trajs = []
    while total_steps < batch_size:
        traj = rollout_one_traj(env, K, explore_mag=explore_mag, test = test)
        traj.c_rew = np.sum(traj.rews)
        total_steps += len(traj.xs)
        trajs.append(traj)
    
    return trajs

def evaluation(env, batch_size, K):
    trajs = roll_out(env, batch_size, K, test = True)
    c_rews  = [trajs[i].c_rew for i in range(len(trajs))]
    return np.mean(c_rews)

def processing_batch(trajs, gamma = 0.99):
    gamma_seqs = np.array([gamma**i for i in range(0,100)])
    xs = []
    acts = []
    ctgs = []
    avg_rew = 0.0
    for traj_id in range(len(trajs)):
        traj = trajs[traj_id]
        xs = xs + traj.xs
        acts = acts + traj.acts
        rews = traj.rews
        avg_rew += np.sum(np.array(rews)*gamma_seqs[0:len(rews)])
        qs = [np.sum(rews[i:]*gamma_seqs[0:len(rews[i:])]) for i in range(len(rews))]
        ctgs = ctgs + qs 

    #a constant baseline: the mean of the traj cummulative reward:
    baseline = avg_rew / (len(trajs)*1.)
    xs = np.array(xs)
    acts = np.array(acts)
    ctgs = np.array(ctgs)
    advs = ctgs - baseline
    return xs, acts, advs, ctgs
        
def policy_gradient_adam_linear_policy(env, optimizer, explore_mag = 0.1, 
            batch_size = 100, max_iter = 100, K0 = None, Natural = False):

    a_dim = env.a_dim
    x_dim = env.x_dim 
    if K0 is None:
        K0 = 0.0 * np.random.randn(a_dim, x_dim)
    
    K = K0 
    baseline = 0.0 

    #evalue the optimal K:
    optimal_perf = evaluation(env = env, batch_size = batch_size*2, K = env.optimal_K)
    print "optimal K's performance is {}".format(optimal_perf)

    test_perfs = []
    for e in range(max_iter):

        #evaluation on the current K:
        perf = evaluation(env = env, batch_size = batch_size*2, K=K)
        print "at epoch {}, current K's avg cummulative reward is {}".format(e, perf)
        test_perfs.append(perf)
        num_steps = 0
        #rollout:
        trajs = roll_out(env = env, batch_size = batch_size, K = K, 
            explore_mag = explore_mag, test = False)
        #process batch data:
        xs,acts,advs,ctgs = processing_batch(trajs)
        #compute gradient:
        mean_acts = xs.dot(K.T)
        d_acts = acts - mean_acts
        weighted_d_acts = np.matmul(d_acts[:,:,np.newaxis], advs[:,np.newaxis, np.newaxis])
        weighted_d_acts = np.reshape(weighted_d_acts, (weighted_d_acts.shape[0], env.a_dim))
        weighted_d_acts_s = np.matmul(weighted_d_acts[:,:,np.newaxis], xs[:, np.newaxis,:])
        gradient = np.mean(weighted_d_acts_s,axis=0)

        new_flatten_param = optimizer.update(K.reshape(env.x_dim*env.a_dim), 
            -gradient.reshape(env.x_dim*env.a_dim))
        
        K = new_flatten_param.reshape(env.a_dim, env.x_dim)
    
    return test_perfs


parser = argparse.ArgumentParser()
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--x_dim', type=int, default=100)
parser.add_argument('--a_dim', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--iters', type=int, default=100)

args = parser.parse_args()
np.random.seed(args.seed)

env = LQGEnv(x_dim = args.x_dim, u_dim = args.a_dim, rank = 5)
optimizer = Adam(args.x_dim*args.a_dim, args.lr)

K0 = np.random.randn(args.a_dim, args.x_dim)*0.01
test_perfs = policy_gradient_adam_linear_policy(env, explore_mag=0.1,
        optimizer = optimizer, batch_size=args.batch_size, max_iter=args.iters, 
        K0 = K0)



