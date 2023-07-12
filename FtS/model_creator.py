import os
import numpy as np
import pandas as pd
from sys import platform

master_df_path = "/scratch/feats_FtS/master_df/master_fsim.h5"
master_df = pd.read_hdf(master_df_path, key=f"final_feats")
master_df.sort_values(by='timestamp', inplace=True, ignore_index=True)

# Step 1: Extract into different arrays

# Step 2: split training and testing data in smart manner
# Tip 1: 30/70 split
# Tip 2: k-fold cross-validation