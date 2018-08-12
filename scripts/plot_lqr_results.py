import numpy as np
import matplotlib.pyplot as plt
import pickle

ars_results = pickle.load(open('ars_result_cross_H_10_160.p', 'rb'))
exact_results = pickle.load(open('exact_result_cross_H_10_160.p', 'rb'))
reinforce_results = pickle.load(open('lqr_reinforce_cross_H_10_160.p', 'rb'))

mean_ars = np.mean(np.array(ars_results), axis=1)
mean_exact = np.mean(np.array(exact_results), axis=1)
mean_reinforce = np.mean(np.array(reinforce_results), axis=1)

std_ars = np.std(np.array(ars_results), axis=1)
std_exact = np.std(np.array(exact_results), axis=1)
std_reinforce = np.std(np.array(reinforce_results), axis=1)

H = [10, 20, 40, 60, 80, 100, 120, 140, 160]

# mean_ars = np.divide(mean_ars, H)
# std_ars = np.divide(std_ars, H)

# mean_reinforce = np.divide(mean_reinforce, H)
# std_reinforce = np.divide(std_reinforce, H)

# mean_exact = np.divide(mean_exact, H)
# std_exact = np.divide(std_exact, H)

plt.plot(H, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_ars - std_ars/2), np.minimum(1e6, mean_ars + std_ars/2), facecolor='red', alpha=0.2)

plt.plot(H, mean_exact, color='blue', label='EXACT', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_exact - std_exact/2), np.minimum(1e6, mean_exact + std_exact/2), facecolor='blue', alpha=0.2)

plt.plot(H, mean_reinforce, color='green', label='REINFORCE', linewidth=2)
plt.fill_between(H, np.maximum(0, mean_reinforce - std_reinforce/2), np.minimum(1e6, mean_reinforce + std_reinforce/2), facecolor='green', alpha=0.2)

plt.show()