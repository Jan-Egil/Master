import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform, argv
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# num_feats = 1000
# num_elems = 100000
# X_array = np.random.rand(num_feats, num_elems)
# print(X_array.shape)

# model = KMeans(n_clusters=6, n_init='auto')
# X_new = model.fit_predict(X_array.T)
# print(X_new.shape)
# #clf = model.fit(X_array.T)
# #X_new = clf.predict(X_array.T)
# print(X_new.shape)

# for i in range(6):
#     print(f"num {i}: {X_new[X_new==i].shape}")

path = "/scratch/feats_CLtS/reduced_feats/clustered_feats.h5"

df = pd.read_hdf(path, key=f'reduced_feats')
df.sort_values(by='timestamp', inplace=True, ignore_index=True)
df.reset_index()

num_pics = len(df.index)

for i in tqdm(range(num_pics)):
    df['feat_reduced'][i] = np.exp(-df['feat_reduced'][i])/np.sum(np.exp(-df['feat_reduced'][i]))

print(np.sum(df['feat_reduced'][0]))

# X_array = 10*np.random.rand(6)

# print(X_array)
# print(1/X_array)
# X_temp = np.exp(-X_array)
# X_probs = X_temp/np.sum(X_temp)
# print(X_probs)
# print(np.sum(X_probs))