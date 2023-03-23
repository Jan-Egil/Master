import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from tqdm import tqdm

from Data_Classifier import Data_Classifier

sns.set_theme()

dims = 3
classifier1 = 'SVM'
classifier2 = 'LDA'

path_feats = '/scratch/oath_v1.1/features/auroral_feat.h5'
path_classification = '/scratch/oath_v1.1/classifications/classifications.csv'

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

iters = 1

"""
Classifier 1
"""
for i, dims in enumerate(tqdm(dims_array)):
    sum2 = 0
    sum6 = 0
    prec2sum = 0
    prec6sum = 0
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
        sum2 += prcnt2
        sum6 += prcnt6

        prec2 = DC2.precision()
        prec6 = DC6.precision()
        prec2sum += prec2
        prec6sum += prec6
    prcnt2_array_classifier1[i] = sum2/iters
    prcnt6_array_classifier1[i] = sum6/iters

"""
Classifier 2
"""
for i, dims in enumerate(tqdm(dims_array)):
    sum2 = 0
    sum6 = 0
    prec2sum = 0
    prec6sum = 0
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
        sum2 += prcnt2
        sum6 += prcnt6

        prec2 = DC2.precision()
        prec6 = DC6.precision()
        prec2sum += prec2
        prec6sum += prec6
    prcnt2_array_classifier2[i] = sum2/iters
    prcnt6_array_classifier2[i] = sum6/iters
    precision2_array_classifier2[i] = prec2sum/iters
    precision6_array_classifier2[i] = prec6sum/iters

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

plt.show()