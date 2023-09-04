import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

alphas = np.load('alphas.npy')

recalls = np.load('recall.npy')
recalls_std = np.load('recall_std.npy')

balaccs = np.load('balacc.npy')
balaccs_std = np.load('balacc_std.npy')

falsepos = np.load('falsepos.npy')
falsepos_std = np.load('falsepos_std.npy')

fig_recall, ax_recall = plt.subplots()
fig_balaccs, ax_balaccs = plt.subplots()
fig_falsepos, ax_falsepos = plt.subplots()

elems = ['recalls, balaccs, falsepos']

ax_recall.errorbar(alphas, recalls)
ax_recall.set_xlabel('Hyperparameter Alpha')
ax_recall.set_ylabel('Recall')
ax_recall.set_title('Recall Rate')
ax_recall.set_xscale('log')

ax_balaccs.errorbar(alphas, balaccs)
ax_balaccs.set_xlabel('Hyperparameter Alpha')
ax_balaccs.set_ylabel('Balanced Accuracy')
ax_balaccs.set_title('Balanced Accuracy')
ax_balaccs.set_xscale('log')

ax_falsepos.errorbar(alphas, falsepos)
ax_falsepos.set_xlabel('Hyperparameter Alpha')
ax_falsepos.set_ylabel('FPR')
ax_falsepos.set_title('False Positive Rate')
ax_falsepos.set_xscale('log')

plt.show()

"""
plt.figure()
plt.title("Recall Rate")
plt.xlabel("Hyperparameter alpha")
plt.ylabel("Recall")
plt.plot(alphas, recalls)
plt.xscale('log')

plt.figure()
plt.title("Balanced Accuracy")
plt.xlabel("Hyperparameter alpha")
plt.ylabel("Balanced accuracy")
plt.plot(alphas, balaccs)
plt.xscale('log')

plt.figure()
plt.title("False Positive Rate")
plt.xlabel("Hyperparameter alpha")
plt.ylabel("FPR")
plt.plot(alphas, falsepos)
plt.xscale('log')

plt.show()
"""