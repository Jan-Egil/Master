import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import precision_score, recall_score

if platform == "win32":
    master_df_path = "master_trainable_fsim.h5"
else:
    master_df_path = "/scratch/feats_FtS/master_df/master_trainable_fsim.h5"
master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

# Step 1: Extract into different arrays

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

# Step 2: split training and testing data in smart manner
# Tip 1: 30/70 split
# Tip 2: k-fold cross-validation

k = 5
kfold = KFold(n_splits=k, shuffle=True)
ridge_classifier = RidgeClassifier()

for train_idxs, test_idxs in kfold.split(array_feats):
    train_idxs_filtered = []
    test_idxs_filtered = []
    print(test_idxs)

    for train_idx in train_idxs:
        if trainable[train_idx] == 1:
            train_idxs_filtered.append(train_idx)
    for test_idx in test_idxs:
        if trainable[test_idx] == 1:
            test_idxs_filtered.append(test_idx)
    
    num_imgs_train = len(train_idxs_filtered)
    num_imgs_test = len(test_idxs_filtered)

    X_train = np.zeros((num_imgs_train, num_feats*30))
    X_test = np.zeros((num_imgs_test, num_feats*30))
    Y_train = substorm_onset[train_idxs_filtered]
    Y_test = substorm_onset[test_idxs_filtered]

    i = 0
    j = 0
    for train_idx in train_idxs_filtered:
        X_train[i] = np.array(array_feats[train_idx-29:train_idx+1]).flatten()
        i += 1
    for test_idx in test_idxs_filtered:
        X_test[j] = np.array(array_feats[test_idx-29:test_idx+1]).flatten()
        j += 1
    del i, j

    clf = ridge_classifier.fit(X_train, Y_train)
    score = clf.score(X_train, Y_train)
    score2 = clf.score(X_test, Y_test)
    print(score, score2)
    #precision = precision_score(Y_test, Y_pred)
    #print(precision)

# Print the amount of data with onset
tot = 0
real = 0
for elem in substorm_onset:
    tot += 1
    if elem == 1:
        real += 1

print(real/tot)
# After print: roughly 1.2% of data reports an onset