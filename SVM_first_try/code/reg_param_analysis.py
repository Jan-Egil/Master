import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from tqdm import tqdm

from Data_Classifier import Data_Classifier


path_feats = '/scratch/oath_v1.1/features/auroral_feat.h5'
path_classification = '/scratch/oath_v1.1/classifications/classifications.csv'

with h5py.File(path_feats, 'r') as f:
    features = f['Logits'][:]

df = pd.read_csv(path_classification, header=16)
aurora_binary = np.array(df['class2'])
aurora_class = np.array(df['class6'])

dims_array = np.array([3,5,10,15,20])
reg_param_array = np.logspace(start,stop,50)
result2 = np.zeros((dims_array.shape[0],reg_param_array.shape[0]))
result6 = np.zeros((dims_array.shape[0],reg_param_array.shape[0]))

for i, reg_param in enumerate(tqdm(reg_param_array)):
    for j, dims in enumerate(dims_array):
        DC2 = Data_Classifier(features, aurora_binary)
        DC6 = Data_Classifier(features, aurora_class)

        DC2.scale()
        DC6.scale()

        DC2.decompose(dims=dims)
        DC6.decompose(dims=dims)

        pred2, test2 = DC2.classify(classifier='SVM')
        pred6, test6 = DC6.classify(classifier='SVM')

        prcnt2 = ...
        prcnt6 = ...