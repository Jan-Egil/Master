import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, exit
from time import perf_counter

from sklearn.linear_model import RidgeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score

# Just to get away the error of the "missing model"
if 0 == 1:
    model = "lol"


if platform == "win32":
    master_df_path = "master_trainable_fsim.h5"
else:
    master_df_path = "/scratch/feats_FtS/master_df/master_trainable_fsim_4feat_5min.h5"
master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

"""
----------------------------------
"""

months_to_train = [10, 11, 12, 1, 2]
#month_to_train = 11

# The models (uncomment the one to try out)

model = RidgeClassifier(class_weight='balanced')       # Wait
#model = GaussianProcessClassifier()                    # Wait (Can't run due to matrix size)
#model = SVC(class_weight='balanced')                   # Wait
#model = SVC(class_weight='balanced', kernel="linear", max_iter=400)   # Wait
#model = GaussianNB()                                   # Wait
#model = MLPClassifier()                                # Wait
#model = KNeighborsClassifier()                         # Wait
#model = AdaBoostClassifier()                           # Wait
#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight='balanced') # Wait


"""
---------------------------------
"""

train_score_list = []
test_score_list = []
recall_list = []
balanced_acc_list = []
false_positive_list = []

fitting_time_list = []
predicting_time_list = []
total_time_list = []

try:
    model
except NameError:
    exit("\nWoops, you forgot to uncomment a model!\n")

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

for month_to_train in months_to_train:
    idxs_train = []
    idxs_test = []
    for i in indices:
        month = timestamps_list[i].month
        if month == month_to_train:
            idxs_test.append(i)
        else:
            idxs_train.append(i)

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


    print("Started classifying")

    start_fitting = perf_counter()
    clf = model.fit(X_train, Y_train)
    stop_fitting = perf_counter()
    fitting_time = stop_fitting-start_fitting

    print("Started predicting")

    start_predicting = perf_counter()
    Y_pred = clf.predict(X_test)
    stop_predicting = perf_counter()
    predicting_time = stop_predicting-start_predicting
    total_time = fitting_time+predicting_time

    score = clf.score(X_train, Y_train)
    score2 = clf.score(X_test, Y_test)
    print(f"\nTrain: {int(score*1000)/10}%\n Test: {int(score2*1000)/10}%\n")

    idxs = []
    for idx, val in enumerate(Y_test):
        if val == 1:
            idxs.append(idx)
    Y_pred_subset = Y_pred[idxs]
    Y_test_subset = Y_test[idxs]
    substorm_accuracy = accuracy_score(Y_test_subset, Y_pred_subset)
    print(f"\n{int(substorm_accuracy*1000)/10}%\n\n")

    balanced_acc = balanced_accuracy_score(Y_test, Y_pred)
    print(f"Balanced accuracy: {balanced_acc}\n")

    print(f"\ntime spent..\nFitting: {fitting_time}\nPredicting: {predicting_time}\nTotal: {total_time}\n\n")

    train_score_list.append(score*100)
    test_score_list.append(score2*100)
    recall_list.append(substorm_accuracy*100)
    balanced_acc_list.append(balanced_acc*100)

    fitting_time_list.append(fitting_time)
    predicting_time_list.append(predicting_time)
    total_time_list.append(total_time)

print(f"training score: {np.mean(train_score_list)} ± {np.std(train_score_list)}")
print(f"test score: {np.mean(test_score_list)} ± {np.std(test_score_list)}")
print(f"Recall score: {np.mean(recall_list)} ± {np.std(recall_list)}")
print(f"Fitting time: {np.mean(fitting_time_list)} ± {np.std(fitting_time_list)}")
print(f"Predicting time: {np.mean(predicting_time_list)} ± {np.std(predicting_time_list)}")
print(f"total time: {np.mean(total_time_list)} ± {np.std(total_time_list)}\n\n")