import numpy as np 
import argparse
from envs.LQG.LQG import LQGEnv
from envs.linreg.linreg import LinReg
from utils.adam import Adam
from utils.ars import *
from IPython import embed
import random


class Trajectory(object):

    def __init__(self):
        self.xs = []
        self.acts = []
        self.acts_mean = []
        self.rews = []

        self.c_rew = 0.0

def rollout_one_traj(env, K, explore_mag = 0.1, test = False, stats = None):
    traj = Trajectory()
    x = env.reset()
    done = False
    x_orginal = []
    while done is False:
        if stats is None:
            mean_a = K.dot(x)
        else:
            mean_a = K.dot((x - stats.mean)/stats.std)

        if test is False:
            a = mean_a + explore_mag*np.random.randn(env.a_dim)
        else:
            a = np.copy(mean_a)
            
        traj.xs.append(x)
        traj.acts.append(a)
        traj.acts_mean.append(mean_a)
        x, r, done, _ = env.step(a)
        traj.rews.append(r)

    traj.c_rew = np.sum(traj.rews)
    return traj

def roll_out(env, batch_size, K, explore_mag = 0.1, test = False, stats = None):
    total_steps = 0
    trajs = []
    while total_steps < batch_size:
        traj = rollout_one_traj(env, K, explore_mag=explore_mag, test = test, stats = stats)
        traj.c_rew = np.sum(traj.rews)
        total_steps += len(traj.xs)
        trajs.append(traj)
    
    return trajs

def evaluation(env, batch_size, K, stats = None):
    trajs = roll_out(env, batch_size, K, test = True, stats = stats)
    c_rews  = [trajs[i].c_rew for i in range(len(trajs))]
    return np.mean(c_rews)

def processing_batch(trajs, gamma = 0.99, stats = None):
    gamma = 1.
    gamma_seqs = np.array([gamma**i for i in range(0,1000)])
    xs = []
    acts = []
    acts_mean = []
    ctgs = []
    avg_rew = 0.0
    for traj_id in range(len(trajs)):
        traj = trajs[traj_id]
        xs = xs + traj.xs
        acts = acts + traj.acts
        acts_mean = acts_mean + traj.acts_mean
        rews = traj.rews
        #avg_rew += np.sum(np.array(rews)*gamma_seqs[0:len(rews)])
        avg_rew += traj.c_rew
        qs = [np.sum(rews[i:]*gamma_seqs[0:len(rews[i:])]) for i in range(len(rews))]
        #qs = [np.sum(rews[i:]) for i in range(len(rews))]
        #embed()
        ctgs = ctgs + list(np.ones(len(rews))*traj.c_rew) #ctgs + qs 

    #a constant baseline: the mean of the traj cummulative reward:
    baseline = avg_rew / (len(trajs)*1.)
    xs = np.array(xs)
    acts_mean = np.array(acts_mean)
    acts = np.array(acts)
    ctgs = np.array(ctgs)
    advs = ctgs #- baseline

    adv_mean = np.mean(advs)
    adv_std = np.std(advs)
    #advs = (advs - adv_mean) / adv_std

    if stats is not None:
        stats.push_batch(xs)
        xs = (xs - stats.mean)/stats.std

    return xs, acts, acts_mean, advs, ctgs

def policy_gradient_adam_linear_policy(env, optimizer, explore_mag = 0.1, 
            batch_size = 100, max_iter = 100, K0 = None, Natural = False, kl = 1e-3, stats = None, sigma = None):

    a_dim = env.a_dim
    x_dim = env.x_dim 
    if K0 is None:
        K0 = 0.0 * np.random.randn(a_dim, x_dim)
    if sigma is None:
        sigma = 0. #std exp(sigma) = 0

    K = K0
    sigma = sigma
    print (sigma)
    baseline = 0.0 

    #evalue the optimal K:
    #optimal_perf = evaluation(env = env, batch_size = batch_size*2, K = env.optimal_K)
    print ("optimal K's performance is {}".format(-env.optimal_cost))

    test_perfs = []
    #for e in range(max_iter):
    e = 0
    while True:
        #evaluation on the current K:
        #perf = evaluation(env = env, batch_size = batch_size, K=K, stats = stats)
        perf = env.evaluate_policy(K)
        info = (e, e*batch_size, sigma, optimizer.alpha, perf)
        print (info)
        if abs(perf - env.optimal_cost)/env.optimal_cost < 0.05:
            return e*batch_size
            break

        #print "at epoch {}, current K's avg cummulative reward is {}".format(e, perf)
        test_perfs.append(perf)
        num_steps = 0
        #rollout:
        trajs = roll_out(env = env, batch_size = batch_size, K = K, 
            explore_mag = np.exp(sigma), test = False, stats = stats)
        #process batch data:
        xs,acts,acts_mean, advs,ctgs = processing_batch(trajs, gamma = 1., stats = stats)
        #compute gradient:
        mean_acts = acts_mean
        d_acts = acts - mean_acts
        
        #d_acts = d_acts.reshape(d_acts.shape[0])
        #grad = (xs.T.dot(ctgs*d_acts))/batch_size
        #gradient = grad.reshape(1, env.x_dim)

        weighted_d_acts = np.matmul(d_acts[:,:,np.newaxis], advs[:,np.newaxis, np.newaxis])
        weighted_d_acts = np.reshape(weighted_d_acts, (weighted_d_acts.shape[0], env.a_dim))
        gradient = weighted_d_acts.T.dot(xs)/(xs.shape[0])
        gradient *= 1./(np.exp(sigma)**2)

        weighted_d_acts_square = np.mean(np.sum(d_acts**2, axis = 1)*advs)/(np.exp(sigma)**2)
        concatenated_grad = np.concatenate([gradient.reshape(env.x_dim*env.a_dim), [weighted_d_acts_square]])

        #if np.linalg.norm(gradient) > 100.:
        #    gradient = 100.*gradient/np.linalg.norm(gradient)

        if Natural is True:
            JJp = xs.T.dot(xs)/xs.shape[0]
            #JJp = np.mean(np.matmul(xs[:,:,np.newaxis], xs[:,np.newaxis,:]),axis=0)
            descent_dir = (np.linalg.lstsq(JJp+np.eye(env.x_dim)*1e-3, gradient.T, rcond = None)[0]).T
            #descent_dir = np.linalg.inv(JJp+1e-3*np.eye(env.x_dim)).dot(gradient.T).T
            lr = np.sqrt(kl/(gradient.flatten().dot(descent_dir.flatten())))
            K = K + lr * descent_dir
            #embed()
        else:
            #K = K + kl * gradient
            #new_flatten_param = optimizer.update(K.reshape(env.x_dim*env.a_dim), 
            #   -gradient.reshape(env.x_dim*env.a_dim))
            #K = new_flatten_param.reshape(env.a_dim, env.x_dim)
            new_flatten_param = optimizer.update(np.concatenate([K.reshape(env.x_dim*env.a_dim),[sigma]]), -concatenated_grad)
            K = new_flatten_param[0:-1].reshape(env.a_dim, env.x_dim)
            sigma = new_flatten_param[-1]


            #optimizer.alpha /= (e+1)**0.1
        if e*batch_size > 1e5:
            #break
            return e*batch_size
        
        e = e + 1

    return test_perfs

'''
parser = argparse.ArgumentParser()
parser.add_argument('--tsteps', type=int, default=100)
parser.add_argument('--x_dim', type=int, default=500)
parser.add_argument('--a_dim', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128) 
#in every epoch, we generate batch_size # of steps (batch_size/T = num of trajectories)
parser.add_argument('--iters', type=int, default=100)
args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)

env = LQGEnv(x_dim = args.x_dim, u_dim = args.a_dim, rank = 5, seed = args.seed)
#env = LinReg(args.x_dim, batch_size = 1)
optimizer = Adam(args.x_dim*args.a_dim, args.lr)

stats = RunningStat(args.x_dim * args.a_dim)
stats = None

K0 = 0.01*np.ones((args.a_dim, args.x_dim)) #np.random.randn(args.a_dim, args.x_dim)*0.5
#K0 = np.linalg.pinv(env.B).dot(np.eye(env.x_dim) - env.A)
#K0 = 5*np.random.randn(args.a_dim, args.x_dim) / np.sqrt(args.x_dim*args.a_dim)
test_perfs = policy_gradient_adam_linear_policy(env, explore_mag=0.1,
        optimizer = optimizer, batch_size=args.batch_size, max_iter=args.iters, 
        K0 = K0, Natural = False, kl = args.lr, stats = stats)
'''


