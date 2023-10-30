import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sys import platform, exit
from time import perf_counter

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sns


def threshold_scanner(C, thresholds):
    model = LogisticRegression(C=C, class_weight='balanced', max_iter=10000)

    master_df_path = "/scratch/feats_CLtS/master_df/master_trainable_fsim_6feat.h5"

    master_df = pd.read_hdf(master_df_path, key=f"final_feats")
    master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

    num_feats = master_df['averaged_feats'][0].shape[0]
    num_imgs = len(master_df.index)

    array_feats = np.zeros((num_imgs, num_feats))
    substorm_onset = np.zeros(num_imgs)
    trainable = np.zeros(num_imgs)
    timestamps_list = []

    for i in range(num_imgs):
        array_feats[i] = master_df['averaged_feats'][i]
        substorm_onset[i] = master_df['substorm_onset'][i]
        trainable[i] = master_df['trainable'][i]
        timestamps_list.append(master_df['timestamp'][i])
    timestamps = np.array(timestamps_list)

    n_samples = len(substorm_onset)
    indices = np.arange(n_samples)
    idxs_train, idxs_test = train_test_split(indices, test_size=0.2, shuffle=False)
    train_idxs_filtered = []
    test_idxs_filtered = []
    for train_idx in idxs_train:
        if trainable[train_idx] == 1:
            train_idxs_filtered.append(train_idx)
    for test_idx in idxs_test:
        if trainable[test_idx] == 1:
            test_idxs_filtered.append(test_idx)

    num_imgs_train = len(train_idxs_filtered)
    num_imgs_test = len(test_idxs_filtered)

    X_train = np.zeros((num_imgs_train, num_feats*30))
    X_test = np.zeros((num_imgs_test, num_feats*30))
    Y_train = substorm_onset[train_idxs_filtered]
    Y_test = substorm_onset[test_idxs_filtered]

    for i, train_idx in enumerate(train_idxs_filtered):
        X_train[i] = array_feats[train_idx-29:train_idx+1].flatten()
    for j, test_idx in enumerate(test_idxs_filtered):
        X_test[j] = array_feats[test_idx-29:test_idx+1].flatten()


    clf = model.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)


    balaccs = np.zeros_like(thresholds)
    recalls = np.zeros_like(thresholds)
    FPRs = np.zeros_like(thresholds)

    for idx, threshold in enumerate(thresholds):
        Y_pred_new = (clf.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        balaccs[idx] = balanced_accuracy_score(Y_test, Y_pred_new)
        recalls[idx] = recall_score(Y_test, Y_pred_new)
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_new).ravel()
        FPRs[idx] = fp/(fp+tn)

    best_balacc_idx = np.argmax(balaccs)
    Y_pred_best = (clf.predict_proba(X_test)[:, 1] >= thresholds[best_balacc_idx]).astype(int)
    return balaccs, recalls, FPRs


sns.set_theme()

"""
N = 100
M = 200
thresholds = np.linspace(0,1,N)
Cs = np.logspace(-8,8,M)

balaccs_array = np.zeros([M,N])
recalls_array = np.zeros_like(balaccs_array)
fprs_array = np.zeros_like(balaccs_array)

for idx, C in enumerate(tqdm(Cs)):
    balaccs_array[idx], recalls_array[idx], fprs_array[idx] = threshold_scanner(C, thresholds)

np.save('balaccs_array.npy', balaccs_array)
np.save('recalls_array.npy', recalls_array)
np.save('fprs_array.npy', fprs_array)
np.save('thresholds_array.npy', thresholds)
np.save('Cs_array.npy', Cs)
"""

balaccs_array = np.load('balaccs_array.npy')
recalls_array = np.load('recalls_array.npy')
fprs_array = np.load('fprs_array.npy')
thresholds = np.load('thresholds_array.npy')
Cs = np.load('Cs_array.npy')

print(balaccs_array.shape)

plt.figure()
plt.contourf(thresholds, Cs, balaccs_array)
plt.colorbar(cmap='Greys').set_label('Balanced Accuracy', fontsize='large')
plt.yscale('log')
plt.xlabel("Threshold", fontsize='large')
plt.ylabel("Regularization Hyperparameter C", fontsize='large')
plt.title("Balanced Accuracy\nHyperparameter Plot", fontsize='large')

plt.figure()
plt.contourf(thresholds, Cs, recalls_array)
plt.colorbar().set_label('Recall', fontsize='large')
plt.yscale('log')
plt.xlabel("Threshold", fontsize='large')
plt.ylabel("Regularization Hyperparameter C", fontsize='large')
plt.title("Recall\nHyperparameter Plot", fontsize='large')

plt.figure()
plt.contourf(thresholds, Cs, fprs_array)
plt.colorbar().set_label('False Positive Rate', fontsize='large')
plt.yscale('log')
plt.xlabel("Threshold", fontsize='large')
plt.ylabel("Regularization Hyperparameter C", fontsize='large')
plt.title("False Positive Rate\nHyperparameter Plot", fontsize='large')

plt.show()