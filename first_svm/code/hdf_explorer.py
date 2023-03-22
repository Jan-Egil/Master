import h5py
import numpy as np

path_feats = '/scratch/oath_v1.1/features/auroral_feat.h5'

with h5py.File(path_feats, 'r') as f:
    features = f['Logits'][:]

print(features.shape)