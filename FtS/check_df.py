import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sys import platform
from tqdm import tqdm

if platform == "win32":
    master_df_path = "master_trainable_fsim.h5"
else:
    master_df_path = "/scratch/feats_FtS/master_df/master_trainable_fsim_35feat_5min.h5"
master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

months = [10, 11, 12, 1, 2]

num_elems = len(master_df.index)
print(master_df)
print(master_df.columns)

print(set(master_df['trainable']))
print(master_df['substorm_onset'][450:500])


substorm_dict = {}

for i in tqdm(range(num_elems)):
    if master_df['trainable'][i] == 0:
        continue
    month = str(master_df['timestamp'][i].month)
    substorm = master_df['substorm_onset'][i]
    
    if substorm == 0:
        substorm_key = month + "no"
    else:
        substorm_key = month + "yes"

    if substorm_key in substorm_dict:
        substorm_dict[substorm_key] += 1
    else:
        substorm_dict[substorm_key] = 1


for month in months:
    if str(month)+"yes" in substorm_dict:
        num_sub = substorm_dict[str(month) + "yes"]
    else:
        num_sub = 0
    num_tot = num_sub + substorm_dict[str(month) + "no"]
    prcnt = int((num_sub/num_tot)*10000)/100
    print(f"month: {month}\nnum substorms: {num_sub}\nnum total: {num_tot}\n percentage: {prcnt}%\n")
