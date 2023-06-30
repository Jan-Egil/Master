import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import platform

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 0th: Define the paths

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

df = pd.read_hdf(save_path+f"iter_0_loc_fsim.h5", key='features')

for iter in tqdm(range(1,1686)):
    df2 = pd.read_hdf(save_path+f"iter_{iter}_loc_fsim.h5", key='features')
    df = pd.concat([df, df2], ignore_index=True)
    df.reset_index()
del df2

print(df['features'])



# 2nd: Apply scaling and PCA to reduce said dimensions. (35 dims is a nice start)


# 3rd: Make sure to connect the right features with the right locatons and timestamps in a new dataframe


# 4th: Save dataframe to file. Delete variable
# Tip: Try to plot features against timestamps. See if you find some structure in the madness


# 5th: Extract dataframe from file


# 6th: Combine the features into bins of 1 minutes each. (Average of each feature on their own?)


# 7th: Use location and timestamp-data together with onset-data to determine whether or not there has been an onset.


# 8th: Create dataframe with timestamp (in 1 minute iters), whether or not there will be an onset in the next 15 mins, and the reduced features in bins


# 9th: Save dataframe to file. Delete variabele.


# 10th: 