import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path_feats_dir = '/scratch/asim/'
path_feats_list = ['201011.hdf', '201012.hdf', '201101.hdf', '201102.hdf']
path_feats = path_feats_dir + path_feats_list[0]

out_path = '/scratch/newfeats'

num_feats = 10

# First create all dataframe-keys
with h5py.File(path_feats_dir + path_feats_list[0], 'r') as f:
    pandacolumns = list(f['asim']['axis0'][0:16])
    for i in range(num_feats):
        pandacolumns.append(f'feat_{i}')
    df = pd.DataFrame(columns=pandacolumns)

# Then create reduced features
num_pics_tot = 0
num_pics_in_each = [0]

for path_file_name in path_feats_list:
    path_feat = path_feats_dir + path_file_name
    with h5py.File(path_feat, 'r') as f:
        print(f['asim']['block1_values'][0].shape)
        num_pics = f['asim']['block1_values'].shape[0]
        num_pics_tot = num_pics_tot + num_pics
        num_pics_in_each.append(num_pics)

features = np.zeros([num_pics_tot, 1000], dtype=np.float32)

for i, path_file_name in enumerate(path_feats_list):
    path_feat = path_feats_dir + path_file_name
    with h5py.File(path_feat, 'r') as f:
        num_pics = f['asim']['block1_values'].shape[0]
        for j in tqdm(range(num_pics)):
            features[j+num_pics_in_each[i]] = f['asim']['block1_values'][i]

sc = StandardScaler()
reducer = PCA(num_feats)
features = sc.fit_transform(features)
features = reducer.fit_transform(features)

print(features.shape)

np.save('/scratch/newfeats/features', features)


"""
# Then insert content into dataframe
with h5py.File(path_feats, 'r') as f:
    keys = list(f['asim'].keys())
    num_pics = f['asim'][keys[1]].shape[0]
    key = keys[4]
    print(keys)
    print(key)
    print(f['asim'][key][:])
    print(f['asim'][key].shape)
    day = ...
    sec = ...
    location = ...
    wvl = ...
    fl = ...
    url = ...
    filename = ...
    minimum = ...
    maximum = ...
    median = ...
    mean = ...
    variance = ...
    skewness = ...
    kurtosis = ...
    entropy = ...
    energy = ...
    features = ...
    features_reduced = ...
    for i in range(num_feats):
        ...#df[f'feat_{i}'] = features_reduced[i]
"""


# Finally, create hdf-file with content
# Use pd.df.to_hdf