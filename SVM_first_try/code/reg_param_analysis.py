import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from tqdm import tqdm
import os

from Data_Classifier import Data_Classifier

os.chdir('../..')
base_dir = os.getcwd()

path_feats = base_dir + '/data/SVM_first_try/auroral_feat.h5'
path_classification = base_dir + '/data/SVM_first_try/classifications.csv'

with h5py.File(path_feats, 'r') as f:
    features = f['Logits'][:]

df = pd.read_csv(path_classification, header=16)
aurora_binary = np.array(df['class2'])
aurora_class = np.array(df['class6'])

iters = 5

dims_array = np.array([30])#,5,10,15,20])
C_param_array = np.logspace(-5,0,10)
gamma_param_array = np.logspace(-3,0,10)
errorresult2 = np.zeros((dims_array.shape[0],C_param_array.shape[0],gamma_param_array.shape[0]))
errorresult6 = np.zeros((dims_array.shape[0],C_param_array.shape[0],gamma_param_array.shape[0]))


for i, C_reg_param in enumerate(C_param_array):
    for j, gamma_reg_param in enumerate(gamma_param_array):
        for k, dims in enumerate(dims_array):
            errorarray2 = np.zeros(iters)
            errorarray6 = np.zeros(iters)
            for iter in range(iters):
                DC2 = Data_Classifier(features, aurora_binary)
                DC6 = Data_Classifier(features, aurora_class)

                DC2.scale()
                DC6.scale()

                DC2.decompose(dims=dims)
                DC6.decompose(dims=dims)

                pred2, test2 = DC2.classify(classifier='SVM', C = C_reg_param, gamma = gamma_reg_param)
                pred6, test6 = DC6.classify(classifier='SVM', C = C_reg_param, gamma = gamma_reg_param)

                errorarray2[iter] = 100-DC2.accuracy()
                errorarray6[iter] = 100-DC6.accuracy()
                print(f"{i+1}/{C_param_array.shape[0]}, {j+1}/{gamma_param_array.shape[0]}, {k+1}/{dims_array.shape[0]}, {iter+1}/{iters}")
            errorresult2[k,j,i] = np.mean(errorarray2)
            errorresult6[k,j,i] = np.mean(errorarray6)

np.save('errorresult2.npy', errorresult2)
np.save('errorresult6.npy', errorresult6)
np.save('C_param_array.npy', C_param_array)
np.save('gamma_param_array.npy', gamma_param_array)
np.save('dims_array.npy', dims_array)

"""
for i, dims in enumerate(tqdm(dims_array)):
    plt.figure()
    plt.title(f"Binary, {dims} dimensions")
    plt.contourf(C_param_array, gamma_param_array, errorresult2[i])
    plt.axis('scaled')
    plt.colorbar()

plt.show()
"""
