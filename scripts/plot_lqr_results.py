import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--H_start', type=int, default=10, help="Horizon length to start with")
parser.add_argument('--H_end', type=int, default=200, help="Horizon length to end with")
parser.add_argument('--H_bin', type=int, default=20, help="Horizon length spacing at which experiments are done (or bin size)")
parser.add_argument('--episode', action='store_true')
args = parser.parse_args()

ars_filename = "ars_result_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
exact_filename = "exact_result_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"
reinforce_filename = "lqr_reinforce_cross_H_" + str(args.H_start)+"_"+str(args.H_end)+"_"+str(args.H_bin)+".p"

ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))
reinforce_results = pickle.load(open(reinforce_filename, 'rb'))

mean_ars = np.mean(np.array(ars_results), axis=1)
mean_exact = np.mean(np.array(exact_results), axis=1)
mean_reinforce = np.mean(np.array(reinforce_results), axis=1)

# 10 random seeds were used
std_ars = np.std(np.array(ars_results), axis=1) / np.sqrt(10)
std_exact = np.std(np.array(exact_results), axis=1) / np.sqrt(10)
std_reinforce = np.std(np.array(reinforce_results), axis=1) / np.sqrt(10)

H = list(range(args.H_start, args.H_end+args.H_bin, args.H_bin))

if args.episode:
    mean_ars = np.divide(mean_ars, H)
    std_ars = np.divide(std_ars, H)
    mean_reinforce = np.divide(mean_reinforce, H)
    std_reinforce = np.divide(std_reinforce, H)
    mean_exact = np.divide(mean_exact, H)
    std_exact = np.divide(std_exact, H)

plt.plot(H, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_ars - std_ars), np.minimum(1e8, mean_ars + std_ars), facecolor='red', alpha=0.2)

plt.plot(H, mean_exact, color='blue', label='ExAct', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_exact - std_exact), np.minimum(1e8, mean_exact + std_exact), facecolor='blue', alpha=0.2)

plt.plot(H, mean_reinforce, color='green', label='REINFORCE', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_reinforce - std_reinforce), np.minimum(1e8, mean_reinforce + std_reinforce), facecolor='green', alpha=0.2)

plt.xlabel('Horizon length')
if not args.episode:    
    plt.ylabel('Number of samples')
    plt.title('Number of samples needed to reach 5% of optimal performance')
else:    
    plt.ylabel('Number of episodes')
    plt.title('Number of episodes needed to reach 5% of optimal performance')

plt.legend()

plt.show()
