import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

alphas = np.load('alphas.npy')
recalls_per_alpha = np.load('recalls_per_alpha.npy')
balanced_acc_per_alpha = np.load('balanced_acc_per_alpha.npy')

plt.figure()
plt.title("Recall rate")
plt.plot(alphas, recalls_per_alpha)
plt.xscale('log')

plt.figure()
plt.title("Balanced accuracy")
plt.plot(alphas, balanced_acc_per_alpha)
plt.xscale('log')

plt.show()