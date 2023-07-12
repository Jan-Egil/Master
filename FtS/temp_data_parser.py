import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

path = "/scratch/feats_FtS/"

df = pd.read_hdf(path+f"iter_0_loc_fsim.h5", key='features')

for iter in tqdm(range(1,1686)):
    df2 = pd.read_hdf(path+f"iter_{iter}_loc_fsim.h5", key='features')
    #print(df2)
    df = pd.concat([df, df2], ignore_index=True)
    df.reset_index()
print(df)
print(df.columns)