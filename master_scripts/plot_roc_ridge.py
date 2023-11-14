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

sns.set_theme()

os.chdir('master_data')

best_alpha_path = "Ridge_CLtS_1bins_6.csv"
df = pd.read_csv(best_alpha_path)
alphas = np.array(df.alphas)
balaccs = np.array(df.balanced_accuracy)
best_balacc_idx = np.argmax(balaccs)


alpha = alphas[best_balacc_idx]
model = RidgeClassifier(alpha=alpha, class_weight='balanced')

del df
del alphas
del balaccs
del best_balacc_idx
os.chdir('..')

master_df_path = "/scratch/feats_CLtS/master_df/master_trainable_fsim_6feat.h5"

master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

num_feats = master_df['averaged_feats'][0].shape[0]
num_imgs = len(master_df.index)

array_feats = np.zeros((num_imgs, num_feats))
substorm_onset = np.zeros(num_imgs)
trainable = np.zeros(num_imgs)
timestamps_list = []
print(master_df.columns)

for i in tqdm(range(num_imgs)):
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


print(X_train.shape)
clf = model.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

thresholds = np.linspace(0,1,200)
balaccs = np.zeros_like(thresholds)
recalls = np.zeros_like(thresholds)
FPRs = np.zeros_like(thresholds)

d = clf.decision_function(X_test)

Y_pred_new = (d >= 0).astype(int)
print(set(Y_pred_new - Y_pred))

for idx, threshold in enumerate(thresholds):
    new_threshold = (threshold*2)-1
    Y_pred_new = (d >= new_threshold).astype(int)
    balaccs[idx] = balanced_accuracy_score(Y_test, Y_pred_new)
    recalls[idx] = recall_score(Y_test, Y_pred_new)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_new).ravel()
    FPRs[idx] = fp/(fp+tn)

best_balacc_idx = np.argmax(balaccs)
print(balaccs[best_balacc_idx])
print(recalls[best_balacc_idx])
print(FPRs[best_balacc_idx])
print(thresholds[best_balacc_idx])
Y_pred_best = (d >= (thresholds[best_balacc_idx]*2)-1).astype(int)

tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_best).ravel()
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

plt.figure()
plt.plot(thresholds, balaccs, label="Balanced Accuracy")
plt.xlabel("Threshold", fontsize='large')
plt.ylabel("Balanced Accuracy", fontsize='large')
plt.title("Balanced Accuracy for Ridge Classifier\n6 features, 1 min bins, varying threshold", fontsize='large')
plt.legend()

plt.figure()
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, FPRs, label="False Positive Rate")
plt.xlabel("Threshold", fontsize='large')
plt.ylabel("Recall / FPR", fontsize='large')
plt.title("Recall / FPR for Ridge Classifier\n6 features, 1 min bins, varying threshold", fontsize='large')
plt.legend()

RocCurveDisplay.from_estimator(clf, X_test, Y_test)
plt.plot([0,1], [0,1], "--", label="Random Chance")
plt.legend()
plt.title(f"ROC Curve for Ridge Classifier\n6 features, 1 min bins", fontsize='large')
plt.xlabel("False Positive Rate",fontsize='large')
plt.ylabel("True Positive Rate", fontsize='large')

ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_best, normalize='true')
plt.grid()

plt.show()
