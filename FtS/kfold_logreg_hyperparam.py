import os
import numpy as np
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, balanced_accuracy_score

"""
----------------------------------
"""

numfeats = 35 #4, 6 and 35 exists
minbin = 1 #1 for 1min-bins, 5 for 5min-bins
wegonshuffle = False

k = 5
kfold = KFold(n_splits=k, shuffle=wegonshuffle)


"""
---------------------------------
"""

if not numfeats in {4,6,35}:
    exit(f"\nWoops, {numfeats} number of features does not exist!\n")
if not minbin in {1,5}:
    exit(f"\nWoops, {minbin} binning interval does not exist!\n")

if platform == "win32":
    master_df_path = f"master_trainable_fsim_{numfeats}feat"
else:
    master_df_path = f"/scratch/feats_FtS/master_df/master_trainable_fsim_{numfeats}feat"

if minbin == 1:
    master_df_path += ".h5"
elif minbin == 5:
    master_df_path += "_5min.h5"


master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)


train_score_list = []
test_score_list = []

recall_list = []
balanced_acc_list = []
false_positive_list = []

fitting_time_list = []
predicting_time_list = []
total_time_list = []

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


#n_samples = len(substorm_onset)
#indices = np.arange(n_samples)
#idxs_train, idxs_test = train_test_split(indices, test_size=0.2)

alphas = np.logspace(-8,-2,1000)

recalls = np.zeros_like(alphas)
recalls_std = np.zeros_like(alphas)

balaccs = np.zeros_like(alphas)
balaccs_std = np.zeros_like(alphas)

falsepos = np.zeros_like(alphas)
falsepos_std = np.zeros_like(alphas)

for alphaidx, alpha in enumerate(tqdm(alphas)):
    model = LogisticRegression(class_weight='balanced', C=alpha) 
    for idxs_train, idxs_test in kfold.split(array_feats):
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

        X_train = np.zeros((num_imgs_train, num_feats*8))
        X_test = np.zeros((num_imgs_test, num_feats*8))
        Y_train = substorm_onset[train_idxs_filtered]
        Y_test = substorm_onset[test_idxs_filtered]

        for i, train_idx in enumerate(train_idxs_filtered):
            X_train[i] = array_feats[train_idx-7:train_idx+1].flatten()
        for j, test_idx in enumerate(test_idxs_filtered):
            X_test[j] = array_feats[test_idx-7:test_idx+1].flatten()


        #print("Started classifying")

        start_fitting = perf_counter()
        clf = model.fit(X_train, Y_train)
        stop_fitting = perf_counter()
        fitting_time = stop_fitting-start_fitting

        #print("Started predicting")

        start_predicting = perf_counter()
        Y_pred = clf.predict(X_test)
        stop_predicting = perf_counter()
        predicting_time = stop_predicting-start_predicting
        total_time = fitting_time+predicting_time

        score = clf.score(X_train, Y_train)
        score2 = clf.score(X_test, Y_test)

        model_recall = recall_score(Y_test, Y_pred)

        balanced_acc = balanced_accuracy_score(Y_test, Y_pred)

        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
        model_fpr = fp/(fp+tn)

        train_score_list.append(score*100)
        test_score_list.append(score2*100)
        recall_list.append(model_recall*100)
        balanced_acc_list.append(balanced_acc*100)
        false_positive_list.append(model_fpr*100)

        fitting_time_list.append(fitting_time)
        predicting_time_list.append(predicting_time)
        total_time_list.append(total_time)

    recalls[alphaidx] = np.mean(recall_list)
    recalls_std[alphaidx] = np.std(recall_list)

    balaccs[alphaidx] = np.mean(balanced_acc_list)
    balaccs_std[alphaidx] = np.std(balanced_acc_list)

    falsepos[alphaidx] = np.mean(false_positive_list)
    falsepos_std[alphaidx] = np.std(false_positive_list)

os.chdir('data')

np.save('alphas.npy', alphas)

np.save('recall.npy', recalls)
np.save('recall_std.npy', recalls_std)

np.save('balacc.npy', balaccs)
np.save('balacc_std.npy', balaccs_std)

np.save('falsepos.npy', falsepos)
np.save('falsepos_std.npy', falsepos_std)