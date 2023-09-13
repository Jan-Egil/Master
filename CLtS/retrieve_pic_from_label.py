import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from PIL import Image

os.environ["CDF_LIB"] = "/uio/hume/student-u58/janeod/Downloads/cdf39_0-dist-all/cdf39_0-dist/lib"
from spacepy import pycdf

# First find which images are the closest to k-means-point.
# Then extract datetime

save_path_reduced = "/scratch/feats_CLtS/reduced_feats/clustered_feats.h5"
df = pd.read_hdf(save_path_reduced, key=f'reduced_feats')
df.sort_values(by='timestamp', inplace=True, ignore_index=True)
df.reset_index()

n_pic_per_label = 3
n_labels = len(df['feat_reduced'][0])
n_imgs = len(df.index)


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

#np.save('best_dist_array.npy', best_dist_array)
#np.save('best_idx_array.npy', best_idx_array)


#best_dist_array = np.load('best_dist_array.npy')
#best_idx_array = np.load('best_idx_array.npy')

cdf_path = 'temp.cdf'
key_img = f"thg_asf_fsim"

n_labels = best_idx_array.shape[0]
n_best = best_idx_array.shape[1]

for label_num in tqdm(range(n_labels)):
    labelarray_idxs = best_idx_array[label_num]
    for rank in tqdm(range(n_best)):
        idx = labelarray_idxs[rank]
        rel_datetime = df['timestamp'][idx]
        year = rel_datetime.year
        month = str(rel_datetime.month).zfill(2)
        day = str(rel_datetime.day).zfill(2)
        hour = str(rel_datetime.hour).zfill(2)

        print(f"day: {day},")
        cdf_url = f"http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/fsim/{year}/{month}/thg_l1_asf_fsim_{year}{month}{day}{hour}_v01.cdf"
        r = requests.get(cdf_url)
        a = open(cdf_path, 'wb')
        a.write(r.content)
        a.close()
        

        cdf = pycdf.CDF("temp.cdf")
        num_imgs = cdf[key_img].shape[0]

        for i in range(num_imgs):
            img_datetime = cdf[f'{key_img}_epoch'][i]
            #print(img_datetime)
            if img_datetime != rel_datetime:
                continue

            img_array = cdf[key_img][i]
            crop_percent = 12
            num_pixels = int(img_array.shape[0]*(crop_percent/100))
            temp_img = Image.fromarray(img_array)
            (left, upper, right, lower) = (num_pixels, num_pixels, img_array.shape[0]-num_pixels, img_array.shape[0]-num_pixels)
            temp_img2 = temp_img.crop((left, upper, right, lower))
            img_array = np.asarray(temp_img2)

            # Rescale to range [0,1]
            # First subtract 1st percentile
            percentile1 = np.percentile(img_array, 1)
            img_array = img_array - percentile1

            # Then divide by 99th percentile
            percentile99 = np.percentile(img_array, 99)
            img_array = img_array/percentile99

            # Then set everything under 0 to 0, and everything over 1 to 1
            img_array[img_array > 1] = 1
            img_array[img_array < 0] = 0
            
            img = Image.fromarray(img_array*255)
            img = img.resize((224,224))
            img = img.convert("RGB")
            img.save(f'label_{label_num}_rank_{rank}.png')
        cdf.close()
        os.remove("temp.cdf")

print("Script finished!")