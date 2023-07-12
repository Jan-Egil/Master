import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import h5py
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path_feats_dir = '/scratch/asim/'
path_feats_list = ['201011.hdf', '201012.hdf', '201101.hdf', '201102.hdf']
path_feats = path_feats_dir + path_feats_list[0]

out_path = '/scratch/newfeats'

num_feats = 10

# First create all dataframe-keys
with h5py.File(path_feats_dir + path_feats_list[0], 'r') as f:
    #pandacolumns = list(f['asim']['axis0'][0:16])
    pandacolumns = []
    for i in range(num_feats):
        pandacolumns.append(f'feat_{i}')
    pandacolumns.append('datetime')
    pandacolumns.append('substorm')
    pandacolumns.append('location')
    df = pd.DataFrame(columns=pandacolumns)

# Then create reduced features
num_pics_tot = 0
num_pics_in_each = [0]

for path_file_name in path_feats_list:
    path_feat = path_feats_dir + path_file_name
    with h5py.File(path_feat, 'r') as f:
        print(f['asim']['block2_values'][:])
        num_pics_in_file = f['asim']['block1_values'].shape[0]
        num_pics_tot = num_pics_tot + num_pics_in_file
        num_pics_in_each.append(num_pics_in_file)

print(num_pics_tot)
# Then create (and fill) arrays to insert into dataframe
features = np.zeros([num_pics_tot, 1000], dtype=np.float32)
datetimes = np.zeros(num_pics_tot, dtype='datetime64[s]')
substorm_onset = np.zeros(num_pics_tot, dtype=np.bool_)
locations = []

for i, path_file_name in enumerate(tqdm(path_feats_list)):
    path_feat = path_feats_dir + path_file_name
    year = str(path_feat[14:18])
    month = str(path_feat[18:20])
    with h5py.File(path_feat, 'r') as f:
        num_pics_in_file = f['asim']['block1_values'].shape[0]
        for j in tqdm(range(num_pics_in_file)):
            features[j+num_pics_in_each[i]] = f['asim']['block1_values'][j]
            
            date = str(f['asim']['block2_values'][j][0]).zfill(2)
            second = str(datetime.timedelta(seconds=int(f['asim']['block3_values'][j]))).zfill(8)
            datetime_of_pic = np.datetime64(f"{year}-{month}-{date}T{second}")
            datetimes[j+num_pics_in_each[i]] = datetime_of_pic
            #locations.append(...)

sc = StandardScaler()
reducer = PCA(num_feats)
features = sc.fit_transform(features)
features = reducer.fit_transform(features)

print(features.shape)

np.save('/scratch/newfeats/features', features)

# Then insert content into dataframe

for i in range(num_feats):
    df[f'feat_{i}'] = features[:, i]
df['datetime'] = datetimes
df['substorm'] = substorm_onset
df['location'] = ...
print(df)


# Finally, create hdf-file with content
# Use pd.df.to_hdf