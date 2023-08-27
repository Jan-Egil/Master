import numpy as np

alphas = np.load('alphas.npy')

recalls = np.load('recall.npy')
recalls_std = np.load('recall_std.npy')

balaccs = np.load('balacc.npy')
balaccs_std = np.load('balacc_std.npy')

falsepos = np.load('falsepos.npy')
falsepos_std = np.load('falsepos_std.npy')

idx = np.argmax(balaccs)

print(f"balacc: {balaccs[idx]} ± {balaccs_std[idx]}")
print(f"recall: {recalls[idx]} ± {recalls_std[idx]}")
print(f"falsepos: {falsepos[idx]} ± {falsepos_std[idx]}")
print("alpha: %e" % alphas[idx])