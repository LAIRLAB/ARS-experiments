import numpy as np
import matplotlib.pyplot as plt
import pickle

ars_results = pickle.load(open('ars_result_cross_H_10_160.p', 'rb'))
exact_results = pickle.load(open('exact_result_cross_H_10_160.p', 'rb'))
#reinforce_results = pickle.load(open('lqr_reinforce_cross_H_10_160.p', 'rb'))

mean_ars = np.mean(np.array(ars_results), axis=1)
mean_exact = np.mean(np.array(exact_results), axis=1)
#mean_reinforce = np.mean(np.array(reinforce_results), axis=1)

std_ars = np.std(np.array(ars_results), axis=1)
std_exact = np.std(np.array(exact_results), axis=1)
#std_reinforce = np.std(np.array(reinforce_results), axis=1)

# H = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
H = [5, 10, 20, 40]
mean_ars = mean_ars[0:4]
std_ars = std_ars[0:4] / 4
mean_exact = mean_exact[0:4]
std_exact = std_exact[0:4] / 4

#mean_ars = np.divide(mean_ars, H)
#std_ars = np.divide(std_ars, H)

#mean_reinforce = np.divide(mean_reinforce, H)
#std_reinforce = np.divide(std_reinforce, H)

#mean_exact = np.divide(mean_exact, H)
#std_exact = np.divide(std_exact, H)

plt.plot(H, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_ars - std_ars), np.minimum(1e8, mean_ars + std_ars), facecolor='red', alpha=0.2)

plt.plot(H, mean_exact, color='blue', label='ExAct', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_exact - std_exact), np.minimum(1e8, mean_exact + std_exact), facecolor='blue', alpha=0.2)

#plt.plot(H, mean_reinforce, color='green', label='REINFORCE', linewidth=2)
#plt.fill_between(H, np.maximum(0, mean_reinforce - std_reinforce), np.minimum(1e6, mean_reinforce + std_reinforce), facecolor='green', alpha=0.2)

plt.xlim([0, 30])
plt.xlabel('Horizon length')
plt.ylabel('Number of samples')
#plt.ylabel('Number of episodes')
plt.title('Number of samples needed to reach 5% of optimal performance')
#plt.title('Number of episodes needed to reach 5% of optimal performance')

plt.legend()

plt.show()
