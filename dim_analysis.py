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
prcnt2_array = np.zeros(dims_array.shape[0])
prcnt6_array = np.zeros(dims_array.shape[0])

iters = 10

for i, dims in enumerate(tqdm(dims_array)):
    sum2 = 0
    sum6 = 0
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
    prcnt2_array[i] = sum2/iters
    prcnt6_array[i] = sum6/iters

plt.plot(dims_array, prcnt2_array, label="SVM, 2 class")
plt.plot(dims_array, prcnt6_array, label="SVM, 6 class")

prcnt2_array = np.zeros(dims_array.shape[0])
prcnt6_array = np.zeros(dims_array.shape[0])

for i, dims in enumerate(tqdm(dims_array)):
    sum2 = 0
    sum6 = 0
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
    prcnt2_array[i] = sum2/iters
    prcnt6_array[i] = sum6/iters

plt.plot(dims_array, prcnt2_array, label="LDA, 2 class")
plt.plot(dims_array, prcnt6_array, label="LDA, 6 class")

plt.xlabel("Dimensions")
plt.ylabel("% Accuracy")
plt.title(f"% Accuracy vs dimension using PCA feature projection.\n{iters} iterations per dimension")

plt.legend()
plt.show()