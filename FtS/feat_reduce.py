import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

array_feats_first = np.random.random((100,1000))
loc = ['fsim' for i in range(100)]
timestamps = [datetime.now() for i in range(100)]

dict = {"features": [array_feats_first[0]],
        "timestamp": timestamps[0],
        "loc": loc[0]}
df = pd.DataFrame(dict, dtype=object)


for i in tqdm(range(1,100)):
    dict = {"features": [array_feats_first[i]],
            "timestamp": timestamps[i],
            "loc": loc[i]}
    df2 = pd.DataFrame(dict)
    df = pd.concat([df, df2], ignore_index=True)
    df.reset_index()

del df2

# 0th: Define the paths
"""
if platform == 'win32':
    curr_path = os.getcwd()
    temp_dir = "\\tempdir"
    full_path = curr_path+temp_dir
    cdfpath = full_path + "\\temp.cdf"
    save_path = ...
else:
    curr_path = os.getcwd()
    temp_dir = "/tempdir"
    full_path = curr_path+temp_dir
    cdfpath = full_path + "/temp.cdf"
    save_path = "/scratch/feats_FtS/"

# 1st: Gather all features in one large array-like structure
# Tip: first make one big pandas dataframe, then turn that into array
print("Extracting the data from file")
df = pd.read_hdf(save_path+f"iter_0_loc_fsim.h5", key='features')

for iter in tqdm(range(1,1686)):
    df2 = pd.read_hdf(save_path+f"iter_{iter}_loc_fsim.h5", key='features')
    df = pd.concat([df, df2], ignore_index=True)
    df.reset_index()
del df2
print("Data extraction done!\n")
"""

# Define and extract features-column into its own array
num_points = len(df.index)
num_feats = 1000
array_feats = np.zeros((num_points, num_feats))

for i in range(num_points):
    features = df['features'][i]
    array_feats[i] = features
del features

# 2nd: Apply scaling and PCA to reduce said dimensions. (35 dims is a nice start)
# HEADS UP! One should maybe train-test-split first before scaling and reducing to be statistically correct.
# This is sort of cheating. Should look closer into this.

print("Scaling the data..")
sc = StandardScaler()
array_feats = sc.fit_transform(array_feats)
print("Scaling done!\n")

print("Reducing dimensionality of data..")
reducer = PCA(n_components=35)
array_feats = reducer.fit_transform(array_feats)
print("Reduction done!\n")

print(array_feats[0].shape)

# 3rd: Connect the right features with the right locatons and timestamps in a new dataframe



# 4th: Save dataframe to file. Delete variable
# Tip: Try to plot features against timestamps. See if you find some structure in the madness


# 5th: Extract dataframe from file


# 6th: Combine the features into bins of 1 minutes each. (Average of each feature on their own?)


# 7th: Use location and timestamp-data together with onset-data to determine whether or not there has been an onset.


# 8th: Create dataframe with timestamp (in 1 minute iters), whether or not there will be an onset in the next 15 mins, and the reduced features in bins


# 9th: Save dataframe to file. Delete variabele.


# 10th: 