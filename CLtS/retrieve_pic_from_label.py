import numpy as np
import pandas as pd
from tqdm import tqdm

# First find which images are the closest to k-means-point.
# Then extract datetime

save_path_reduced = "/scratch/feats_CLtS/reduced_feats/clustered_feats.h5"
df = pd.read_hdf(save_path_reduced, key=f'reduced_feats')
df.sort_values(by='timestamp', inplace=True, ignore_index=True)
df.reset_index()

n_pic_per_label = 3
n_labels = len(df['feat_reduced'][0])
n_imgs = len(df.index)

"""
best_dist_list_full = []
best_idx_list_full = []

for label in tqdm(range(n_labels)):
    best_dist_list = []
    best_idx_list = []
    for n_best in tqdm(range(n_pic_per_label)):
        best_dist = 0
        for img_label in tqdm(range(n_imgs)):
            if img_label in best_idx_list:
                continue
            dist = df['feat_reduced'][img_label][label]
            if dist > best_dist:
                best_dist = dist
                best_idx = img_label
        best_dist_list.append(best_dist)
        best_idx_list.append(best_idx)
    print(best_dist_list)
    print(best_idx_list)
    print("----------")
    best_dist_list_full.append(best_dist_list)
    best_idx_list_full.append(best_idx_list)

best_dist_array = np.array(best_dist_list_full)
best_idx_array = np.array(best_idx_list_full)

np.save('best_dist_array.npy', best_dist_array)
np.save('best_idx_array.npy', best_idx_array)
"""

best_dist_array = np.load('best_dist_array.npy')
best_idx_array = np.load('best_idx_array.npy')

print(best_dist_array)
print(best_idx_array)
print(best_idx_array.shape)
n_labels = best_idx_array.shape[0]
n_best = best_idx_array.shape[1]

