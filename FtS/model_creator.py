import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.svm import SVC

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
# Tip 2: k-fold cross-validation (Went with this)
# NB: Make sure to deal with imbalanced dataset!

y_or_n = input("Want to do simple Ridge? [Y/n] ")

if y_or_n == "Y" or y_or_n == "y":
    k = 5
    kfold = KFold(n_splits=k, shuffle=True)
    ridge_classifier = RidgeClassifier(class_weight='balanced')

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
        for i, train_idx in enumerate(train_idxs_filtered):
            X_train[i] = array_feats[train_idx-29:train_idx+1].flatten()
        for j, test_idx in enumerate(test_idxs_filtered):
            X_test[j] = array_feats[test_idx-29:test_idx+1].flatten()
            


        clf = ridge_classifier.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        score = clf.score(X_train, Y_train)
        score2 = clf.score(X_test, Y_test)
        print(f"Train: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")
        #precision = precision_score(Y_test, Y_pred)
        #print(precision)
        idxs = []
        for idx, val in enumerate(Y_test):
            if val == 1:
                idxs.append(idx)
        Y_pred_subset = Y_pred[idxs]
        Y_test_subset = Y_test[idxs]
        substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
        print(f"\n{substorm_accuracy}\n\n")

y_or_n = input("Want to do Ridge with the new and improved cross validation? [Y/n] (unfinished) ")
if y_or_n == "Y" or y_or_n == "y":
    k = 5
    ridge_classifier = RidgeClassifierCV(class_weight='balanced', cv=k)
    
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
        for i, train_idx in enumerate(train_idxs_filtered):
            X_train[i] = array_feats[train_idx-29:train_idx+1].flatten()
        for j, test_idx in enumerate(test_idxs_filtered):
            X_test[j] = array_feats[test_idx-29:test_idx+1].flatten()
            


        clf = ridge_classifier.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        score = clf.score(X_train, Y_train)
        score2 = clf.score(X_test, Y_test)
        print(f"Train: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")
        #precision = precision_score(Y_test, Y_pred)
        #print(precision)
        idxs = []
        for idx, val in enumerate(Y_test):
            if val == 1:
                idxs.append(idx)
        Y_pred_subset = Y_pred[idxs]
        Y_test_subset = Y_test[idxs]
        substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
        print(f"\n{substorm_accuracy}\n\n")


y_or_n = input("Want to do simple SVM? [Y/n] ")

if y_or_n == "Y" or y_or_n == "y":
    n_samples = len(substorm_onset)
    indices = np.arange(n_samples)
    idxs_train, idxs_test = train_test_split(indices, test_size=0.3)

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

    SVM_classifier = SVC(class_weight='balanced')


    print("Started classifying")
    clf = SVM_classifier.fit(X_train, Y_train)
    print("Started predicting")
    Y_pred = clf.predict(X_test)
    score = clf.score(X_train, Y_train)
    score2 = clf.score(X_test, Y_test)
    print(f"Train: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")

    idxs = []
    for idx, val in enumerate(Y_test):
        if val == 1:
            idxs.append(idx)
    Y_pred_subset = Y_pred[idxs]
    Y_test_subset = Y_test[idxs]
    substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
    print(f"\n{int(substorm_accuracy*1000)/10}%\n\n")

y_or_n = input("Want to do SVM with month-separated train-test-split? [Y/n] ")

if y_or_n == "Y" or y_or_n == "y":
    months = [12]#months = [10, 11, 12, 1, 2]
    for month_to_train in months:
        n_samples = len(substorm_onset)
        indices = np.arange(n_samples)
        idxs_train = []
        idxs_test = []
        for i in indices:
            month = timestamps_list[i].month
            if month == month_to_train:
                idxs_test.append(i)
            else:
                idxs_train.append(i)
        
        #idxs_train, idxs_test = train_test_split(indices, test_size=0.3)

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

        SVM_classifier = SVC(class_weight='balanced')
        #SVM_classifier = RidgeClassifier(class_weight='balanced')

        print("Started classifying")
        clf = SVM_classifier.fit(X_train, Y_train)
        print("Started predicting")
        Y_pred = clf.predict(X_test)
        score = clf.score(X_train, Y_train)
        score2 = clf.score(X_test, Y_test)
        print(f"month: {month_to_train}")
        print(f"Train: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")

        idxs = []
        for idx, val in enumerate(Y_test):
            if val == 1:
                idxs.append(idx)
        Y_pred_subset = Y_pred[idxs]
        Y_test_subset = Y_test[idxs]
        substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
        print(f"\n{int(substorm_accuracy*1000)/10}%\n\n")
    
y_or_n = input("Do you want to use the Gaussian Process classifier? (NB: Needs a lot of RAM) [Y/n] ")

if y_or_n == "Y" or y_or_n == "y":
    n_samples = len(substorm_onset)
    indices = np.arange(n_samples)
    idxs_train, idxs_test = train_test_split(indices, test_size=0.5)

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

    GP_classifier = GaussianProcessClassifier()

    print("Started classifying")
    clf = GP_classifier.fit(X_train, Y_train)
    print("Started predicting")
    Y_pred = clf.predict(X_test)
    score = clf.score(X_train, Y_train)
    score2 = clf.score(X_test, Y_test)
    print(f"Train: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")

    idxs = []
    for idx, val in enumerate(Y_test):
        if val == 1:
            idxs.append(idx)
    Y_pred_subset = Y_pred[idxs]
    Y_test_subset = Y_test[idxs]
    substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
    print(f"\n{int(substorm_accuracy*1000)/10}%\n\n")

"""
# Print the amount of data with onset
tot = 0
real = 0
for elem in substorm_onset:
    tot += 1
    if elem == 1:
        real += 1

print(real/tot)
# After print: roughly 1.2% of data reports an onset
"""