import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, exit
from time import perf_counter
import matplotlib.pyplot as plt

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

# Just to get away the vscode-error of the "missing model"
if 0 == 1:
    model = "lol"

"""
----------------------------------
"""

numfeats = 35 #4, 6 and 35 exists
minbin = 1 #1 for 1min-bins, 5 for 5min-bins
lon_lat_sep = 10 #10 and 15 exists
wegonshuffle = False

alpha = 8.6e+2

# The models (uncomment the one to try out)
model = RidgeClassifier(alpha=alpha, class_weight='balanced')       # Decently bad, but fast
#model = GaussianProcessClassifier()                    # Wait (Can't run due to matrix size)
#model = SVC(class_weight='balanced')                   # Wait
#model = SVC(class_weight='balanced', kernel="linear", max_iter=1000)   # Wait
#model = GaussianNB()                                   # Wait
#model = MLPClassifier()                                # Wait
#model = KNeighborsClassifier()                         # Wait
#model = AdaBoostClassifier()                           # Shit and slow
#model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight='balanced') # Wait
#model = LogisticRegression(class_weight='balanced')

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

if lon_lat_sep == 10:
    master_df_path += "_10deg"
elif lon_lat_sep == 15:
    pass

if minbin == 1:
    master_df_path += ".h5"
elif minbin == 5:
    master_df_path += "_5min.h5"


master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

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


#n_samples = len(substorm_onset)
#indices = np.arange(n_samples)
#idxs_train, idxs_test = train_test_split(indices, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(array_feats, substorm_onset, shuffle=wegonshuffle)

start_fitting = perf_counter()
clf = model.fit(X_train, Y_train)
stop_fitting = perf_counter()
fitting_time = stop_fitting-start_fitting

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

RocCurveDisplay.from_estimator(clf, X_test, Y_test)
plt.plot([0,1], [0,1], "--")
plt.title(f"ROC Curve for Linear SVM Classifier\n{numfeats} features, {minbin} minute bins", fontsize='x-large')
plt.show()

