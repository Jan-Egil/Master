import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import os

from Data_Classifier import Data_Classifier

os.chdir('../..')
base_dir = os.getcwd()

classifier1 = 'SVM'
classifier2 = 'LDA'

path_feats = base_dir + '/data/auroral_feat.h5'
path_classification = base_dir + '/data/classifications.csv'

with h5py.File(path_feats, 'r') as f:
    features = f['Logits'][:]

df = pd.read_csv(path_classification, header=16)
aurora_binary = np.array(df['class2'])
aurora_class = np.array(df['class6'])

dims_array = np.arange(3,30)

prcnt2_array_classifier1 = np.zeros(dims_array.shape[0])
prcnt6_array_classifier1 = np.zeros(dims_array.shape[0])
prcnt2_array_classifier2 = np.zeros(dims_array.shape[0])
prcnt6_array_classifier2 = np.zeros(dims_array.shape[0])

precision2_array_classifier1 = np.zeros(dims_array.shape[0])
precision6_array_classifier1 = np.zeros(dims_array.shape[0])
precision2_array_classifier2 = np.zeros(dims_array.shape[0])
precision6_array_classifier2 = np.zeros(dims_array.shape[0])

recall2_array_classifier1 = np.zeros(dims_array.shape[0])
recall6_array_classifier1 = np.zeros(dims_array.shape[0])
recall2_array_classifier2 = np.zeros(dims_array.shape[0])
recall6_array_classifier2 = np.zeros(dims_array.shape[0])

f1_2_array_classifier1 = np.zeros(dims_array.shape[0])
f1_6_array_classifier1 = np.zeros(dims_array.shape[0])
f1_2_array_classifier2 = np.zeros(dims_array.shape[0])
f1_6_array_classifier2 = np.zeros(dims_array.shape[0])

std_2_array_classifier1 = np.zeros(dims_array.shape[0])
std_6_array_classifier1 = np.zeros(dims_array.shape[0])
std_2_array_classifier2 = np.zeros(dims_array.shape[0])
std_6_array_classifier2 = np.zeros(dims_array.shape[0])

iters = 1

"""
Classifier 1
"""

for i, dims in enumerate(tqdm(dims_array)):
    prcntsum2 = 0
    prcntsum6 = 0
    prec2sum = 0
    prec6sum = 0
    rec2sum = 0
    rec6sum = 0
    f12sum = 0
    f16sum = 0
    std2 = np.zeros(iters)
    std6 = np.zeros(iters)
    for j in range(iters):
        DC2 = Data_Classifier(features, aurora_binary)
        DC6 = Data_Classifier(features, aurora_class)

        DC2.scale()
        DC6.scale()

        DC2.decompose(dims=dims)
        DC6.decompose(dims=dims)

        pred2, test2 = DC2.classify(classifier=classifier1)
        pred6, test2 = DC6.classify(classifier=classifier1)

        prcnt2 = DC2.accuracy()
        prcnt6 = DC6.accuracy()
        prcntsum2 += prcnt2
        prcntsum6 += prcnt6
        std2[j] = prcnt2
        std6[j] = prcnt6

        prec2 = DC2.precision()
        prec6 = DC6.precision()
        prec2sum += prec2
        prec6sum += prec6

        rec2 = DC2.recall()
        rec6 = DC6.recall()
        rec2sum += rec2
        rec6sum += rec6

        f12 = DC2.F1()
        f16 = DC6.F1()
        f12sum += f12
        f16sum += f16
    prcnt2_array_classifier1[i] = prcntsum2/iters
    prcnt6_array_classifier1[i] = prcntsum6/iters
    std_2_array_classifier1[i] = np.std(std2)
    std_6_array_classifier1[i] = np.std(std6)
    precision2_array_classifier1[i] = prec2sum/iters
    precision6_array_classifier1[i] = prec6sum/iters
    recall2_array_classifier1[i] = rec2sum/iters
    recall6_array_classifier1[i] = rec6sum/iters
    f1_2_array_classifier1[i] = f12sum/iters
    f1_6_array_classifier1[i] = f16sum/iters

"""
Classifier 2
"""
for i, dims in enumerate(tqdm(dims_array)):
    prcntsum2 = 0
    prcntsum6 = 0
    prec2sum = 0
    prec6sum = 0
    rec2sum = 0
    rec6sum = 0
    f12sum = 0
    f16sum = 0
    std2 = np.zeros(iters)
    std6 = np.zeros(iters)
    for j in range(iters):
        DC2 = Data_Classifier(features, aurora_binary)
        DC6 = Data_Classifier(features, aurora_class)

        DC2.scale()
        DC6.scale()

        DC2.decompose(dims=dims)
        DC6.decompose(dims=dims)

        pred2, test2 = DC2.classify(classifier=classifier2)
        pred6, test2 = DC6.classify(classifier=classifier2)

        prcnt2 = DC2.accuracy()
        prcnt6 = DC6.accuracy()
        prcntsum2 += prcnt2
        prcntsum6 += prcnt6
        std2[j] = prcnt2
        std6[j] = prcnt6

        prec2 = DC2.precision()
        prec6 = DC6.precision()
        prec2sum += prec2
        prec6sum += prec6

        rec2 = DC2.recall()
        rec6 = DC6.recall()
        rec2sum += rec2
        rec6sum += rec6

        f12 = DC2.F1()
        f16 = DC6.F1()
        f12sum += f12
        f16sum += f16
    prcnt2_array_classifier2[i] = prcntsum2/iters
    prcnt6_array_classifier2[i] = prcntsum6/iters
    std_2_array_classifier2[i] = np.std(std2)
    std_6_array_classifier2[i] = np.std(std6)
    precision2_array_classifier2[i] = prec2sum/iters
    precision6_array_classifier2[i] = prec6sum/iters
    recall2_array_classifier2[i] = rec2sum/iters
    recall6_array_classifier2[i] = rec6sum/iters
    f1_2_array_classifier2[i] = f12sum/iters
    f1_6_array_classifier2[i] = f16sum/iters

plt.figure()
plt.plot(dims_array, prcnt2_array_classifier1, label="SVM, 2 class")
plt.plot(dims_array, prcnt6_array_classifier1, label="SVM, 6 class")
plt.plot(dims_array, prcnt2_array_classifier2, label="LDA, 2 class")
plt.plot(dims_array, prcnt6_array_classifier2, label="LDA, 6 class")
plt.xlabel("Dimensions")
plt.ylabel("% Accuracy")
plt.title(f"% Accuracy vs dimension using PCA feature projection.\n{iters} iterations per dimension")
plt.legend()

plt.figure()
plt.plot(dims_array, precision2_array_classifier1, label="SVM, 2 class")
plt.plot(dims_array, precision6_array_classifier1, label="SVM, 6 class")
plt.plot(dims_array, precision2_array_classifier2, label="LDA, 2 class")
plt.plot(dims_array, precision6_array_classifier2, label="LDA, 6 class")
plt.xlabel("Dimensions")
plt.ylabel("Precision (tp/(tp+fp)")
plt.title(f"Precision vs dimension using PCA feature projection.\n{iters} iterations per dimension")
plt.legend()

plt.figure()
plt.plot(dims_array, recall2_array_classifier1, label="SVM, 2 class")
plt.plot(dims_array, recall6_array_classifier1, label="SVM, 6 class")
plt.plot(dims_array, recall2_array_classifier2, label="LDA, 2 class")
plt.plot(dims_array, recall6_array_classifier2, label="LDA, 6 class")
plt.xlabel("Dimensions")
plt.ylabel("Recall (tp/(tp+fn)")
plt.title(f"Recall vs dimension using PCA feature projection.\n{iters} iterations per dimension")
plt.legend()

plt.figure()
plt.plot(dims_array, f1_2_array_classifier1, label="SVM, 2 class")
plt.plot(dims_array, f1_6_array_classifier1, label="SVM, 6 class")
plt.plot(dims_array, f1_2_array_classifier2, label="LDA, 2 class")
plt.plot(dims_array, f1_6_array_classifier2, label="LDA, 6 class")
plt.xlabel("Dimensions")
plt.ylabel("F1 score (precision*recall/(precision+recall)")
plt.title(f"F1 score vs dimension using PCA feature projection.\n{iters} iterations per dimension")
plt.legend()

plt.figure()
plt.plot(dims_array, std_2_array_classifier1, label="SVM, 2 class")
plt.plot(dims_array, std_6_array_classifier1, label="SVM, 6 class")
plt.plot(dims_array, std_2_array_classifier2, label="LDA, 2 class")
plt.plot(dims_array, std_6_array_classifier2, label="LDA, 6 class")
plt.xlabel("Dimensions")
plt.ylabel("STD accuracy")
plt.title(f"STD vs dimension using PCA feature projection.\n{iters} iterations per dimension")
plt.legend()

plt.show()