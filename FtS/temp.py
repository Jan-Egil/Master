import h5py
import pandas as pd

path_feats_dir = '/scratch/asim/'
path_feats_list = ['201011.hdf', '201012.hdf', '201101.hdf', '201102.hdf']
path_feats = path_feats_dir + path_feats_list[0]

df2 = pd.read_hdf(path_feats)
print(df2)
"""
with h5py.File(path_feats, 'r') as f:
    df = pd.read_hdf(f['asim']['block5_values'])
"""