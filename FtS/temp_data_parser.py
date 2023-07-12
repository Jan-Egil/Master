import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

master_df_path = "/scratch/feats_FtS/master_df/master_fsim.h5"
master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

# Extract data into individual arrays

num_feats = master_df['averaged_feats'][0].shape[0]
num_imgs = len(master_df.index)

array_feats = np.zeros((num_imgs, num_feats))
substorm_onset = np.zeros(num_imgs)
timestamps = np.zeros(num_imgs)
print(master_df.columns)

for i in tqdm(range(num_imgs)):
    array_feats[i] = master_df['averaged_feats'][i]
    substorm_onset[i] = master_df['substorm_onset'][i]
    #timestamps[i] = master_df['timestamp'][i]


array_feats_for_plotting = array_feats.T

for i in tqdm(range(num_imgs)):
    if substorm_onset[i] == 1:
        for j in range(30,35):
            y_data = array_feats_for_plotting[j][i-60:i+60]
            plt.plot(y_data, label=f"feature {j}")
        plt.axvline(60+15, color='b', linestyle="--")
        plt.axvline(60, color='b', linestyle='--')
        break
#plt.legend()
plt.grid()
plt.show()

# Plot each feature 2 hour before substorm, until 2 hour after substorm, for first substorm.
